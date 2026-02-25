from pathlib import Path

import torch


def save_checkpoint(path: str, model, optimizer, epoch: int):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        path_obj,
    )


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu"):
    # Keep explicit mode for forward compatibility with upcoming torch defaults.
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state


def find_latest_checkpoint(output_root: str):
    root = Path(output_root)
    if not root.exists():
        return None
    ckpts = sorted(root.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        return None
    return str(ckpts[0])
