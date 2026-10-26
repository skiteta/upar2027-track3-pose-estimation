"""Epoch-boundary recovery without pretending legacy checkpoints saved RNG state."""
from __future__ import annotations

import random
import numpy as np
import torch


def restore_training(checkpoint, model, optimizer, scheduler, scaler, *, steps_per_epoch):
    epoch = int(checkpoint["epoch"])
    if epoch < 0:
        raise ValueError("negative checkpoint epoch")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    else:
        # Original loop calls scheduler.step() exactly once per minibatch,
        # including any AMP-skipped updates. Keep optimizer's saved LR/momentum.
        scheduler.last_epoch = epoch * steps_per_epoch
        scheduler._step_count = scheduler.last_epoch + 1
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    if "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    rng = checkpoint.get("rng_state")
    if rng is not None:
        random.setstate(rng["python"])
        n = rng["numpy"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng["cuda"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    return epoch


def training_checkpoint(epoch, model, optimizer, scheduler, scaler):
    n = np.random.get_state()
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "worker_rng_note": "Persistent DataLoader worker RNG is not captured; a process restart changes worker augmentation streams.",
    }
