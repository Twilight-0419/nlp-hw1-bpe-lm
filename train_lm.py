import argparse
import math
import os
import time

import numpy as np
import torch

from basics.model import LSTMLM, TransformerLM
from basics.optimizer import AdamW, Cross_Entropy_Loss, Gridient_Clipping, Lr_Cos_Schedule
from basics.training import get_batch, load_checkpoint, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Train a Transformer or LSTM language model on tokenized data.")

    p.add_argument("--train-data", type=str, required=True, help="Path to tokenized training data (.npy or raw binary).")
    p.add_argument("--val-data", type=str, default=None, help="Path to tokenized validation data.")
    p.add_argument("--data-dtype", type=str, default="uint16", help="dtype of raw binary token files.")

    p.add_argument("--model", type=str, default="transformer", choices=["transformer", "lstm"])
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--use-rope", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--theta", type=float, default=10000.0)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--cosine-cycle-iters", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--val-interval", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def load_tokens(path, dtype):
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")


def build_model(args, device, dtype):
    if args.model == "lstm":
        return LSTMLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            device=device,
            dtype=dtype,
        )
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device,
        dtype=dtype,
        use_rope=args.use_rope,
        theta=args.theta,
    )


@torch.no_grad()
def evaluate(model, data, args, device):
    model.eval()
    total = 0.0
    for _ in range(args.val_batches):
        x, y = get_batch(data, args.batch_size, args.context_length, device)
        logits = model(x)
        total += Cross_Entropy_Loss(y, logits).item()
    model.train()
    return total / args.val_batches


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cosine_cycle_iters = args.cosine_cycle_iters if args.cosine_cycle_iters is not None else args.max_iters

    train_data = load_tokens(args.train_data, args.data_dtype)
    val_data = load_tokens(args.val_data, args.data_dtype) if args.val_data else None
    assert len(train_data) > args.context_length

    model = build_model(args, device, dtype)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={num_params:,} device={device} start_iter={start_iter}", flush=True)

    model.train()
    tokens_since_log = 0
    last_log_time = time.time()
    running_loss = 0.0
    running_steps = 0

    for it in range(start_iter, args.max_iters):
        lr = Lr_Cos_Schedule(it, args.lr, args.min_lr, args.warmup_iters, cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        optimizer.zero_grad()
        logits = model(x)
        loss = Cross_Entropy_Loss(y, logits)
        loss.backward()
        if args.grad_clip > 0:
            Gridient_Clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += loss.item()
        running_steps += 1
        tokens_since_log += args.batch_size * args.context_length

        if (it + 1) % args.log_interval == 0:
            now = time.time()
            elapsed = now - last_log_time
            token_throughput = tokens_since_log / elapsed if elapsed > 0 else 0.0
            print(
                f"iter {it + 1}/{args.max_iters} | lr {lr:.2e} | "
                f"train loss {running_loss / running_steps:.4f} | "
                f"{token_throughput:.0f} tok/s | {elapsed:.1f}s",
                flush=True,
            )
            running_loss = 0.0
            running_steps = 0
            tokens_since_log = 0
            last_log_time = now

        if val_data is not None and (it + 1) % args.val_interval == 0:
            val_loss = evaluate(model, val_data, args, device)
            print(
                f"iter {it + 1}/{args.max_iters} | val loss {val_loss:.4f} | "
                f"val ppl {math.exp(val_loss):.2f}",
                flush=True,
            )

        if (it + 1) % args.save_interval == 0:
            path = os.path.join(args.checkpoint_dir, f"ckpt_{it + 1}.pt")
            save_checkpoint(model, optimizer, it + 1, path)
            print(f"saved checkpoint to {path}", flush=True)

    final_path = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"saved final checkpoint to {final_path}", flush=True)


if __name__ == "__main__":
    main()