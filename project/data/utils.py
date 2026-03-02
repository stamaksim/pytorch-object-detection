import os
import random

import numpy as np
import torch


def collate_fn(batch):
    return tuple(zip(*batch))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def select_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, outdir: str, filename: str = "best.pt") -> str:
    ensure_dir(outdir)
    checkpoint_path = os.path.join(outdir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path
