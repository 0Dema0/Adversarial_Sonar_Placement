#!/usr/bin/env python3
"""Compare builder (C,d) local features against an existing parts directory sample.

Usage:
  python -m bounds_new.check_features_vs_parts --parts data/evaluating_dataset_parts --index 0

The script loads one sample from the parts directory, calls the builder to
reconstruct local features as `d + C @ x` (x = sonar_placement), and prints a
per-key summary of maximum absolute differences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


LOCAL_KEYS = [
    "cost",
    "obstacle_placement",
    "sonar_placement",
    "axial_coordinates",
    "cartesian_coordinates",
    "distance_to_goal",
    "known_sonar_mask",
    "unknown_sonar_mask",
    "neighbor_cost",
    "neighbor_obstacle",
    "neighbor_sonar",
    "neighbor_mask",
    "ring_cost",
    "ring_obstacle_count",
    "ring_obstacle_density",
    "ring_sonar_count",
    "ring_sonar_density",
]


def infer_grid_dims_from_axial(axial: np.ndarray) -> tuple[int, int]:
    """Infer (height, width) from axial coordinates array shape (N,2)."""
    r_vals = axial[:, 1]
    uniq = np.unique(r_vals)
    height = int(uniq.max() + 1)
    counts = [int((r_vals == r).sum()) for r in uniq]
    width = int(max(counts)) if counts else int(np.sqrt(axial.shape[0]))
    # fallback if product doesn't match N
    N = axial.shape[0]
    if height * width != N:
        if int(np.sqrt(N)) ** 2 == N:
            width = int(np.sqrt(N))
            height = int(np.sqrt(N))
        else:
            width = int(N // height)
    return height, width


def load_sample_parts(parts_dir: Path, idx: int):
    parts_dir = Path(parts_dir)
    meta = json.load(open(parts_dir / "meta.json", "r"))

    sample = {}
    for key in list(meta.get("keys", {}).keys()):
        p = parts_dir / f"{key}.npy"
        if not p.exists():
            continue
        arr = np.load(p, mmap_mode="r")
        sample[key] = arr[idx]

    return sample


def build_local_matrix_from_sample(sample: dict) -> tuple[np.ndarray, dict]:
    """Concatenate local keys from sample into a (N, K) local matrix and return slice map."""
    pieces = []
    slices = {}
    pos = 0
    for key in LOCAL_KEYS:
        if key not in sample:
            raise KeyError(f"Key '{key}' not found in parts sample")
        arr = np.asarray(sample[key])
        if arr.ndim == 1:
            arr2 = arr.reshape(-1, 1)
        else:
            arr2 = arr
        pieces.append(arr2.astype(np.float64))
        w = arr2.shape[1]
        slices[key] = (pos, pos + w)
        pos += w

    local = np.concatenate(pieces, axis=1)
    return local, slices


def main():
    p = argparse.ArgumentParser(description="Compare builder local features with dataset parts")
    p.add_argument("--parts", required=True, help="Path to parts directory (meta.json + .npy files)")
    p.add_argument("--index", type=int, default=0, help="Sample index to compare")
    p.add_argument("--tol", type=float, default=1e-6, help="Tolerance for reporting small differences")
    args = p.parse_args()

    parts_dir = Path(args.parts)
    if not parts_dir.exists():
        raise FileNotFoundError(parts_dir)

    sample = load_sample_parts(parts_dir, args.index)

    # infer dims and sample parameters
    axial = np.asarray(sample["axial_coordinates"])  # (N,2)
    N = axial.shape[0]
    height, width = infer_grid_dims_from_axial(axial)
    goal_idx = int(np.asarray(sample.get("goal_index", 0)))

    # obstacles list from sample
    obst_mask = np.asarray(sample.get("obstacle_placement", np.zeros(N, dtype=bool))).astype(bool)
    obstacles = list(np.where(obst_mask)[0])

    # ring radius from sample ring_cost last dim
    ring_cost = np.asarray(sample["ring_cost"])  # (N, R)
    R = ring_cost.shape[1]

    print(f"parts: N={N}, height={height}, width={width}, goal={goal_idx}, obstacles={len(obstacles)}, R={R}")

    # build C,d using bounds_new builder
    from bounds_new.builder import build_local_input
    from environment.utils import detection_probability as detect_fn

    C, d = build_local_input(height=height, width=width, goal_idx=goal_idx, obstacles=obstacles, detection_probability=detect_fn)

    # build local matrix from parts
    local_parts, slices = build_local_matrix_from_sample(sample)

    # build x (sonar placement) and reconstruct local via d + C @ x
    x_mask = np.asarray(sample["sonar_placement"]).astype(bool)
    x = x_mask.astype(float).reshape(-1)

    # sanity shapes
    if C.shape[0] != N:
        raise RuntimeError(f"C first dim {C.shape[0]} != N {N}")
    if d.shape[0] != N:
        raise RuntimeError(f"d first dim {d.shape[0]} != N {N}")

    # reconstruct
    local_builder = d + np.einsum("nik,k->ni", C, x)

    if local_parts.shape != local_builder.shape:
        print("SHAPE MISMATCH: parts local", local_parts.shape, "builder local", local_builder.shape)

    diffs = np.abs(local_builder - local_parts)

    # treat non-finite as NaN for reporting
    diffs = np.where(np.isfinite(diffs), diffs, np.nan)

    per_col_max = np.nanmax(diffs, axis=0)
    overall_max = float(np.nanmax(per_col_max))
    print(f"overall max abs diff = {overall_max:.6g}")

    # per-key summary
    print("Per-key max absolute differences:")
    for key in LOCAL_KEYS:
        s, e = slices[key]
        key_max = float(np.nanmax(per_col_max[s:e]))
        print(f"  {key:25s}: max_abs_diff={key_max:.6g}  cols={s}:{e}")

    # show detailed examples for keys that exceed tolerance
    bad_keys = [k for k in LOCAL_KEYS if float(np.nanmax(per_col_max[slices[k][0]:slices[k][1]])) > args.tol]
    if bad_keys:
        print("\nDetailed top differences for keys exceeding tol={}:".format(args.tol))
        for key in bad_keys:
            s, e = slices[key]
            # per-cell max diff within key
            cell_max = np.nanmax(diffs[:, s:e], axis=1)
            order = np.argsort(-cell_max)
            print(f"\nKey '{key}' top diffs:")
            for i in order[:10]:
                print(f"  cell {i:4d}: diff={cell_max[i]:.6g}")

    else:
        print("All keys within tolerance.")


if __name__ == "__main__":
    main()
