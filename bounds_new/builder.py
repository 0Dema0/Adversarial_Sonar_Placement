"""Build linear mapping C and constant d from placement variables to local features.

This version targets the local-feature ordering used by `dataset`/`trainer` and
uses the project's cached precomputes when available.

It will save / load `C.npy` and `d.npy` under `best_bounds/precomp/<hash>`
using the same naming scheme as `dataset/static_cache._cache_subdir`.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pathlib import Path
import os
import json
import datetime

from environment.utils import detection_probability
import dataset.static_cache as sc


def build_local_input(
    height: int = 20,
    width: int = 20,
    goal_idx: int = 399,
    obstacles: list[int] | None = None,
    detection_probability=detection_probability
) -> tuple[npt.NDArray, npt.NDArray]:
    """Construct C and d for the dataset/trainer local features.

    Ordering matches `trainer.data_builder.PartsDataset.local_features`.

    Returns:
      C: np.ndarray shape (N, local_in, num_vars) mapping x -> linear contributions
      d: np.ndarray shape (N, local_in) baseline constants (no sonars)
    """
    # compute cache subdir using same id function as dataset.static_cache
    cache_root = Path("tight_bounds/precomp")
    det_fn = detection_probability if detection_probability is not None else (lambda x: None)
    subdir = sc._cache_subdir(cache_root, height, width, obstacles, det_fn)

    C_path = subdir / "C.npy"
    d_path = subdir / "d.npy"

    # try loading cached arrays first (memory-mapped)
    if subdir.exists() and C_path.exists() and d_path.exists():
        try:
            C_load = np.load(C_path, mmap_mode="r")
            d_load = np.load(d_path, mmap_mode="r")
            return C_load, d_load
        except Exception:
            # fallback to rebuild if load fails
            pass

    # build StaticFeatures (will also ensure visibility/base caches exist under data/precomp)
    static = sc.build_static(height=height, width=width, obstacle_indices=obstacles, detection_probability=detection_probability)
    raw_map = static.raw_map
    total_cells = raw_map.total_cells
    neigh_map = raw_map.neighbor_map

    contrib = static.base_contributions.T.copy()
    
    obstacle_set = set(int(x) for x in static.obstacles)
    R = int(static.ring_obstacle_count.shape[1])

    ring_indices_by_cell = [[list(raw_map.rings[i][r]) if len(raw_map.rings[i]) > r else [] for r in range(1, R + 1)] for i in range(total_cells)]

    # compute local input dimension according to PartsDataset.local_features
    local_in = (
        1  # cost
        + 1  # obstacle_placement
        + 1  # sonar_placement
        + 2  # axial_coordinates
        + 2  # cartesian_coordinates
        + 1  # distance_to_goal
        + 1  # known_sonar_mask
        + 1  # unknown_sonar_mask
        + 6  # neighbor_cost
        + 6  # neighbor_obstacle
        + 6  # neighbor_sonar
        + 6  # neighbor_mask
        + R * 5  # ring_cost, ring_obstacle_count, ring_obstacle_density, ring_sonar_count, ring_sonar_density
    )

    num_vars = total_cells

    C = np.zeros((total_cells, local_in, num_vars), dtype=float)
    d = np.zeros((total_cells, local_in), dtype=float)

    p = 0
    # 1) cost (linear via contrib)
    # baseline (no sonars) -> zero for non-obstacles; obstacles use sentinel
    d[:, p] = 0.0
    C[:, p, :] = contrib.copy()
    # zero placement columns that correspond to obstacle placements
    for j in obstacle_set:
        if 0 <= int(j) < total_cells:
            C[:, p, int(j)] = 0.0
    # obstacle targets: set large constant and remove placement dependence
    for i_obs in obstacle_set:
        if 0 <= int(i_obs) < total_cells:
            d[int(i_obs), p] = 100.0
            C[int(i_obs), p, :] = 0.0
    p += 1

    # 2) obstacle_placement (constant)
    for i in range(total_cells):
        d[i, p] = 1.0 if i in obstacle_set else 0.0
        C[i, p, :] = 0.0
    p += 1

    # 3) sonar_placement (linear: x_i)
    for i in range(total_cells):
        d[i, p] = 0.0
        if i in obstacle_set:
            C[i, p, i] = 0.0
        else:
            C[i, p, i] = 1.0
    p += 1

    # 4) axial_coordinates (constant, 2)
    for i in range(total_cells):
        d[i, p:p + 2] = raw_map.axial_coordinates[i].astype(float)
    C[:, p:p + 2, :] = 0.0
    p += 2

    # 5) cartesian_coordinates (constant, 2)
    for i in range(total_cells):
        d[i, p:p + 2] = raw_map.cartesian_coordinates[i].astype(float)
    C[:, p:p + 2, :] = 0.0
    p += 2

    # 6) distance_to_goal (constant)
    # static.goal_distance is (N, N) hex distances; take column `goal_idx`
    goal_idx = int(goal_idx)
    dist_col = static.goal_distance[:, goal_idx]
    for i in range(total_cells):
        d[i, p] = float(dist_col[i])
    C[:, p, :] = 0.0
    p += 1

    # 7) known_sonar_mask, 8) unknown_sonar_mask (both treated as constants here)
    # These are not linear functions of placement variables; keep baseline zeros.
    for i in range(total_cells):
        d[i, p] = 0.0
    C[:, p, :] = 0.0
    p += 1
    for i in range(total_cells):
        d[i, p] = 0.0
    C[:, p, :] = 0.0
    p += 1

    # 9) neighbor_cost (6) -- linear via contrib of neighbor target
    for i in range(total_cells):
        neighbors = neigh_map[i]
        for k in range(6):
            if k < len(neighbors):
                nb = int(neighbors[k])
                if nb in obstacle_set:
                    d[i, p + k] = 100.0
                    C[i, p + k, :] = 0.0
                else:
                    d[i, p + k] = 0.0
                    if 0 <= nb < total_cells:
                        C[i, p + k, :] = contrib[nb, :].copy()
                        for j in obstacle_set:
                            if 0 <= int(j) < total_cells:
                                C[i, p + k, int(j)] = 0.0
                    else:
                        C[i, p + k, :] = 0.0
            else:
                d[i, p + k] = 0.0
                C[i, p + k, :] = 0.0
    p += 6

    # 10) neighbor_obstacle (6) -- constant
    for i in range(total_cells):
        arr = static.neighbor_obstacle[i].astype(float)
        d[i, p:p + 6] = arr
        C[i, p:p + 6, :] = 0.0
    p += 6

    # 11) neighbor_sonar (6) -- linear: depends on neighbor placement
    for i in range(total_cells):
        neighbors = neigh_map[i]
        for k in range(6):
            if k < len(neighbors):
                nb = int(neighbors[k])
                d[i, p + k] = 0.0
                if nb in obstacle_set:
                    C[i, p + k, :] = 0.0
                else:
                    C[i, p + k, nb] = 1.0
            else:
                d[i, p + k] = 0.0
                C[i, p + k, :] = 0.0
    p += 6

    # 12) neighbor_mask (6) -- constant
    for i in range(total_cells):
        arr = static.neighbor_mask[i].astype(float)
        d[i, p:p + 6] = arr
        C[i, p:p + 6, :] = 0.0
    p += 6

    # 13) ring_cost (R) -- linear mean over ring (exclude obstacle targets in mean)
    for i in range(total_cells):
        for r_idx in range(R):
            inds = ring_indices_by_cell[i][r_idx]
            nonobs = [int(t) for t in inds if t not in obstacle_set]
            d[i, p + r_idx] = 0.0
            if len(nonobs) == 0:
                C[i, p + r_idx, :] = 0.0
            else:
                C[i, p + r_idx, :] = np.sum(contrib[nonobs, :], axis=0) / float(len(nonobs))
                for j in obstacle_set:
                    if 0 <= int(j) < total_cells:
                        C[i, p + r_idx, int(j)] = 0.0
    p += R

    # 14) ring_obstacle_count (R) -- constant
    for i in range(total_cells):
        d[i, p:p + R] = static.ring_obstacle_count[i].astype(float)
        C[i, p:p + R, :] = 0.0
    p += R

    # 15) ring_obstacle_density (R) -- constant
    for i in range(total_cells):
        d[i, p:p + R] = static.ring_obstacle_density[i].astype(float)
        C[i, p:p + R, :] = 0.0
    p += R

    # 16) ring_sonar_count (R) -- linear (sum of placements in ring)
    for i in range(total_cells):
        for r_idx in range(R):
            inds = ring_indices_by_cell[i][r_idx]
            d[i, p + r_idx] = 0.0
            if not inds:
                C[i, p + r_idx, :] = 0.0
            else:
                for j in inds:
                    if j in obstacle_set:
                        C[i, p + r_idx, j] = 0.0
                    else:
                        C[i, p + r_idx, j] = 1.0
    p += R

    # 17) ring_sonar_density (R) -- linear mean over ring (denom = len(inds))
    for i in range(total_cells):
        for r_idx in range(R):
            inds = ring_indices_by_cell[i][r_idx]
            d[i, p + r_idx] = 0.0
            if len(inds) > 0:
                denom = float(len(inds))
                factor = 1.0 / denom
                for j in inds:
                    if j in obstacle_set:
                        C[i, p + r_idx, j] = 0.0
                    else:
                        C[i, p + r_idx, j] = factor
    p += R

    # Attempt to save C and d to cache (float32 to save space)
    try:
        subdir_tmp = subdir.with_suffix('.tmp')
        if subdir_tmp.exists():
            try:
                for pth in subdir_tmp.iterdir():
                    pth.unlink()
                subdir_tmp.rmdir()
            except Exception:
                pass
        subdir_tmp.mkdir(parents=True, exist_ok=True)

        np.save(subdir_tmp / "C.npy", C.astype(np.float32))
        np.save(subdir_tmp / "d.npy", d.astype(np.float32))

        meta = {
            "height": int(height),
            "width": int(width),
            "obstacle_indices": None if obstacles is None else list(map(int, obstacles)),
            "detection_id": getattr(detection_probability, "__name__", str(detection_probability)),
            "created": datetime.datetime.now().isoformat(),
            "local_in": int(local_in),
            "num_vars": int(num_vars),
            "dtype": "float32",
        }
        with open(subdir_tmp / "meta.json", "w") as f:
            json.dump(meta, f)

        # atomic replace
        if subdir.exists():
            for pth in subdir.iterdir():
                try:
                    pth.unlink()
                except Exception:
                    pass
            try:
                subdir.rmdir()
            except Exception:
                pass
        os.replace(str(subdir_tmp), str(subdir))
    except Exception:
        # best-effort: ignore caching errors
        pass

    return C, d


def build_global_context(
    height: int = 20,
    width: int = 20,
    goal_idx: int | None = None,
    obstacles: list[int] | None = None,
    sonar_number: int | None = None,
    detection_probability=detection_probability,
) -> npt.NDArray:
    """Build trainer-compatible global context vector.

    Returns concatenated vector matching `PartsDataset.global_features` order:
    `[goal_onehot (N), obstacle_number (1), sonar_number (1), start_as_known_number (1), start_as_unknown_number (1)]`.

    This helper does not cache results; it reuses `dataset.static_cache.build_static`
    to obtain map geometry and obstacle info.
    """
    static = sc.build_static(height=height, width=width, obstacle_indices=obstacles, detection_probability=detection_probability)
    total_cells = static.raw_map.total_cells

    # goal one-hot
    goal_vec = np.zeros((total_cells,), dtype=np.float32)
    if goal_idx is not None:
        gi = int(goal_idx)
        if 0 <= gi < total_cells:
            goal_vec[gi] = 1.0

    # scalar globals
    obst_num = float(len(static.obstacles))
    snum = float(0 if sonar_number is None else int(sonar_number))
    unk = 0
    known = 0

    tail = np.asarray([obst_num, snum, known, float(unk)], dtype=np.float32)

    return np.concatenate([goal_vec, tail])
