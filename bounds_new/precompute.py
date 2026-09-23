"""Model-specific precomputations: compute and save `c_z` and `const_z` (memmapable `.npy`).

`c_z` shape: (N, H, num_vars) — coefficient of each placement var on each local hidden neuron.
`const_z` shape: (N, H) — constant term per local hidden neuron.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import json
import torch

from . import io as io_mod
from bounds_new import builder as bldr


def _find_state_param(state: dict, name: str):
    # try direct key
    if name in state:
        return state[name]
    # try module prefix
    mname = "module." + name
    if mname in state:
        return state[mname]
    # fallback: substring match
    for k in state.keys():
        if k.endswith(name) or name in k:
            return state[k]
    raise KeyError(f"Parameter '{name}' not found in checkpoint state_dict keys")


def precompute_model_cz(
    checkpoint_path: str | Path,
    height: int = 20,
    width: int = 20,
    goal_idx: int = 399,
    obstacles: list[int] | None = None,
    detection_probability=None,
    force: bool = False,
):
    """Compute and save `c_z` and `const_z` for a model checkpoint.

    Returns model_dir (Path) where files were written and the numpy arrays.
    """
    checkpoint_path = Path(checkpoint_path)
    model_hash = io_mod.model_hash_from_file(checkpoint_path)

    # ensure static precompute exists and get output dir
    model_dir = io_mod.model_output_dir(height, width, obstacles, detection_probability, model_hash)

    c_z_path = model_dir / "c_z.npy"
    const_z_path = model_dir / "const_z.npy"

    if not force and c_z_path.exists() and const_z_path.exists():
        c_z = np.load(c_z_path, mmap_mode="r")
        const_z = np.load(const_z_path, mmap_mode="r")
        return model_dir, c_z, const_z

    # load model state
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        # likely a nn.Module
        try:
            state = ckpt.state_dict()
        except Exception as e:
            raise RuntimeError("Unsupported checkpoint format") from e

    # extract local hidden weights and bias
    W_loc_t = _find_state_param(state, "local_hidden_weight")
    B_loc_t = _find_state_param(state, "local_hidden_bias")

    W_loc = W_loc_t.cpu().numpy()
    B_loc = B_loc_t.cpu().numpy()

    # build C,d (memmap-capable) via builder
    C, d = bldr.build_local_input(height=height, width=width, goal_idx=goal_idx, obstacles=obstacles, detection_probability=detection_probability)

    # shapes: W_loc (N, H, K), C (N, K, num_vars)
    # compute c_z (N, H, num_vars)
    c_z = np.einsum("ihk,ikj->ihj", W_loc, C)

    # compute const_z = w @ d_i + bias
    const_z = np.einsum("ihk,ik->ih", W_loc, d) + B_loc

    # save as float32
    io_mod.atomic_save_npy(c_z_path, c_z.astype(np.float32))
    io_mod.atomic_save_npy(const_z_path, const_z.astype(np.float32))

    meta = {
        "checkpoint": str(checkpoint_path),
        "model_hash": model_hash,
        "c_z_shape": list(c_z.shape),
        "const_z_shape": list(const_z.shape),
    }
    with open(model_dir / "precompute.meta.json", "w") as f:
        json.dump(meta, f)

    return model_dir, c_z, const_z
