"""Training helpers shared by train.py and evaluate.py.

MODIFIED(both): NEW. Adds a CSV loss log, atomic checkpoints that also carry
the RNG state, a way to find the newest checkpoint for ``--resume``, and seed
handling. Kept free of project imports so it can be unit-tested directly.
"""
import csv
import os
import random
import re

import numpy as np
import torch

LOSS_COLUMNS = (
    "step",
    "total_loss",
    "mel_loss",
    "mel_postnet_loss",
    "pitch_loss",
    "energy_loss",
    "duration_loss",
)

_CHECKPOINT_RE = re.compile(r"^(\d+)\.pth\.tar$")


def set_seed(seed):
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_state(tensor):
    """Return an RNG tensor as a CPU tensor, leaving non-tensors untouched.

    MODIFIED(both): ``load_checkpoint`` maps a checkpoint onto the training
    device, which also moves the saved RNG ByteTensors to CUDA. PyTorch only
    accepts CPU ByteTensors in ``set_rng_state``, so resuming from a GPU-saved
    checkpoint raised before any step ran. This is what broke the English
    restart at step 30,000 and was patched Kaggle-side only.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().to("cpu")
    return tensor


def restore_rng_state(state):
    """Restore a state written by :func:`capture_rng_state`; ignore missing keys."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_cpu_state(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_cpu_state(s) for s in state["cuda"]])


def append_loss_csv(path, step, losses):
    """Append one loss row, writing the header only when the file is new."""
    if len(losses) != len(LOSS_COLUMNS) - 1:
        raise ValueError(
            "expected {} losses, got {}".format(len(LOSS_COLUMNS) - 1, len(losses))
        )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if needs_header:
            writer.writerow(LOSS_COLUMNS)
        writer.writerow(
            [int(step)] + ["{:.6f}".format(float(value)) for value in losses]
        )


def save_checkpoint(path, model_state, optimizer_state, step):
    """Atomically save a resumable checkpoint (model, optimizer, step, RNG)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "step": int(step),
        "model": model_state,
        "optimizer": optimizer_state,
        "rng_state": capture_rng_state(),
    }
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device):
    """Load a trusted local checkpoint on ``device``.

    ``weights_only=False`` is required because the RNG state includes NumPy
    arrays; these files are produced by this repository, never downloaded.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("checkpoint not found: {}".format(path))
    return torch.load(path, map_location=device, weights_only=False)


def find_latest_checkpoint(ckpt_path):
    """Return the highest numbered ``<step>.pth.tar`` in ``ckpt_path`` or None."""
    if not os.path.isdir(ckpt_path):
        return None
    steps = []
    for name in os.listdir(ckpt_path):
        match = _CHECKPOINT_RE.match(name)
        if match:
            steps.append(int(match.group(1)))
    return max(steps) if steps else None


def checkpoint_path(ckpt_path, step):
    return os.path.join(ckpt_path, "{}.pth.tar".format(step))
