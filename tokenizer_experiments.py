import argparse
import os
import time

import numpy as np

from basics.tokenizer import Tokenizer

EOS = "<|endoftext|>"
EOS_BYTES = EOS.encode("utf-8")
TRAIN_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VALID_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
VOCAB_PATH = "output/vocab.json"
MERGES_PATH = "output/merges.txt"


def load_tokenizer():
    return Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, [EOS])


def sample_documents(path, num_docs=10, prefix_bytes=50_000_000, seed=0):
    with open(path, "rb") as f:
        raw = f.read(prefix_bytes)
    docs = raw.split(EOS_BYTES)[:-1]
    docs = [doc for doc in docs if doc.strip()]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(docs), size=num_docs, replace=False)
    return [docs[i] for i in indices]


def compression_stats(tokenizer, docs):
    total_bytes = sum(len(doc) for doc in docs)
    total_tokens = sum(len(tokenizer.encode(doc.decode("utf-8"))) for doc in docs)
    return total_bytes, total_tokens, total_bytes / total_tokens


def measure_throughput(tokenizer, path, sample_bytes=20_000_000):
    with open(path, "rb") as f:
        raw = f.read(sample_bytes)
    cut = raw.rfind(EOS_BYTES)
    raw = raw[:cut + len(EOS_BYTES)]
    text = raw.decode("utf-8", errors="ignore")
    start = time.time()
    ids = tokenizer.encode(text)
    elapsed = time.time() - start
    return len(raw), len(ids), elapsed


def encode_file(tokenizer, path, out_bin, block_size=8 * 1024 * 1024):
    total_tokens = 0
    total_bytes = 0
    with open(path, "rb") as f, open(out_bin, "wb") as out:
        buffer = b""
        while True:
            block = f.read(block_size)
            if not block:
                if buffer:
                    ids = tokenizer.encode(buffer.decode("utf-8", errors="ignore"))
                    np.asarray(ids, dtype=np.uint16).tofile(out)
                    total_tokens += len(ids)
                    total_bytes += len(buffer)
                break
            buffer += block
            cut = buffer.rfind(EOS_BYTES)
            if cut < 0:
                continue
            cut += len(EOS_BYTES)
            part, buffer = buffer[:cut], buffer[cut:]
            ids = tokenizer.encode(part.decode("utf-8", errors="ignore"))
            np.asarray(ids, dtype=np.uint16).tofile(out)
            total_tokens += len(ids)
            total_bytes += len(part)
    return total_tokens, total_bytes


def bin_to_npy(bin_path, npy_path, num_tokens, block=20_000_000):
    out = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint16, shape=(num_tokens,))
    src = np.memmap(bin_path, dtype=np.uint16, mode="r")
    for start in range(0, num_tokens, block):
        end = min(start + block, num_tokens)
        out[start:end] = src[start:end]
    out.flush()
    del out, src


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--train-data", type=str, default=TRAIN_PATH)
    parser.add_argument("--valid-data", type=str, default=VALID_PATH)
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args()
    if not args.stats and not args.encode:
        args.stats = args.encode = True

    tokenizer = load_tokenizer()

    if args.stats:
        docs = sample_documents(args.train_data)
        total_bytes, total_tokens, ratio = compression_stats(tokenizer, docs)
        print(f"sampled_docs={len(docs)} bytes={total_bytes} tokens={total_tokens} bytes_per_token={ratio:.4f}")
        for i, doc in enumerate(docs):
            ids = tokenizer.encode(doc.decode("utf-8"))
            print(f"doc{i}: bytes={len(doc)} tokens={len(ids)} bytes_per_token={len(doc) / len(ids):.4f}")

        raw_bytes, ids_count, elapsed = measure_throughput(tokenizer, args.train_data)
        throughput = raw_bytes / elapsed
        print(f"throughput_bytes_per_s={throughput:.0f} elapsed_s={elapsed:.2f} tokens={ids_count}")
        print(f"pile_825GB_seconds={825e9 / throughput:.0f} pile_825GB_days={825e9 / throughput / 86400:.2f}")

    if args.encode:
        os.makedirs(args.out_dir, exist_ok=True)
        for name, path in (("valid", args.valid_data), ("train", args.train_data)):
            bin_path = os.path.join(args.out_dir, f"{name}_tokens.bin")
            npy_path = os.path.join(args.out_dir, f"{name}_tokens.npy")
            start = time.time()
            num_tokens, num_bytes = encode_file(tokenizer, path, bin_path)
            elapsed = time.time() - start
            bin_to_npy(bin_path, npy_path, num_tokens)
            os.remove(bin_path)
            print(
                f"{name}: tokens={num_tokens} bytes={num_bytes} time_s={elapsed:.1f} "
                f"throughput_MB_s={num_bytes / elapsed / 1e6:.3f} "
                f"bytes_per_token={num_bytes / num_tokens:.4f} saved={npy_path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
