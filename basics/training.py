import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional,Tuple



def get_batch(
        x,batch_size,context_length,device_str
):
    num_starts = len(x) - context_length
    starts = np.random.randint(0, num_starts, size=batch_size)
    indices = starts[:, None] + np.arange(context_length)[None, :]

    inputs = torch.as_tensor(x[indices].astype(np.int64), dtype=torch.long, device=device_str)
    targets = torch.as_tensor(x[indices + 1].astype(np.int64), dtype=torch.long, device=device_str)
    return inputs, targets


def save_checkpoint(model, optimizer, iteration, out):
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(src, model, optimizer):
    obj = torch.load(src, weights_only=False)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]