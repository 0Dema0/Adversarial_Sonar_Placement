"""I/O helpers for bounds_new: atomic saves, model directories, hashes.

All outputs are placed under `tight_bounds/<static_subdir>/models/model_<model_hash>/`.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import numpy as np
import os


def model_hash_from_file(path: Path) -> str:
    path = Path(path)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:10]


def model_output_dir(height: int, width: int, obstacles: list[int] | None, detection_probability, model_hash: str) -> Path:
    """Return (and create) output directory for this model.

    Directory layout: `tight_bounds/model_<model_hash>/`.
    This places both local and global bounds together under a model-scoped folder.
    """
    out = Path("tight_bounds") / f"model_{model_hash}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_bounds(path: str | Path, out: dict[tuple[int, int], tuple[float, float]], meta: dict | None = None) -> None:
    """Save bounds mapping to `path` (.npz).

    - `out` maps `(cell_idx, neuron_idx)` -> `(L, U)`.
    - `meta` is optional metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    items = sorted(out.items())
    cells = np.array([k[0] for k, _ in items], dtype=int)
    neurons = np.array([k[1] for k, _ in items], dtype=int)
    L = np.array([v[0] for _, v in items], dtype=float)
    U = np.array([v[1] for _, v in items], dtype=float)

    save_dict = dict(cell=cells, neuron=neurons, L=L, U=U)
    if meta:
        for k, v in meta.items():
            try:
                save_dict[str(k)] = v
            except Exception:
                pass

    # write to a temporary file in the same directory using a file object
    tmp = path.parent / (path.name + ".tmp")
    try:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
    except Exception:
        pass
    with open(tmp, "wb") as f:
        # write archive directly to the file object (avoids numpy appending extensions)
        np.savez(f, **save_dict)
    os.replace(str(tmp), str(path))


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.parent / (path.name + ".tmp")
    # remove leftover tmp if exists
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass
    # write using file object so numpy does not append '.npy' to the filename
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(str(tmp), str(path))


def atomic_save_npz(path: Path, arrays: dict, meta: dict | None = None) -> None:
    tmp = path.parent / (path.name + ".tmp")
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass
    # write archive directly to file object to avoid numpy appending extensions
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(str(tmp), str(path))
    if meta is not None:
        # write meta.json next to the NPZ
        meta_file = path.parent / (path.stem + ".meta.json")
        with open(meta_file, "w") as f:
            json.dump(meta, f)
