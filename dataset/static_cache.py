from environment.features import StaticFeatures
from environment.map import RawMap, ObstacleMap
from environment.utils import detection_probability

from pathlib import Path
import hashlib
import json
import os
import numpy as np


def _sanitize(s: str) -> str:
    # keep safe filename characters
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _cache_subdir(cache_root: Path, height: int, width: int, obstacle_indices: list[int] | None, detection_fn) -> Path:
    # detection id: try module + name, fallback to hash of repr
    try:
        det_id = f"{detection_fn.__module__}.{detection_fn.__name__}"
    except Exception:
        det_id = hashlib.sha1(repr(detection_fn).encode()).hexdigest()[:10]

    det_id = _sanitize(det_id)

    if obstacle_indices is None:
        obs_hash = "none"
    else:
        obs_repr = ",".join(map(str, obstacle_indices))
        obs_hash = hashlib.sha1(obs_repr.encode()).hexdigest()[:10]

    name = f"static_h{height}_w{width}_o{obs_hash}_d{det_id}"
    return cache_root / name


def build_static(
    height: int,
    width: int,
    obstacle_indices: list[int] | None = None,
    detection_probability=detection_probability,
    cache_dir: str | Path | None = "data/precomp",
    force_rebuild: bool = False,
) -> StaticFeatures:
    """Build StaticFeatures, optionally loading/saving a cache for heavy arrays.

    If `cache_dir` is provided, this function will try to load precomputed
    `visibility_map` and `base_contributions` from disk and attach them to
    the returned `StaticFeatures`. If the cache is missing or `force_rebuild`
    is True, it computes them and writes them to disk for future runs.
    """
    raw_map = RawMap(width=width, height=height)

    obstacle_map = ObstacleMap(raw_map, obstacle_indices=obstacle_indices)

    if cache_dir is None:
        return StaticFeatures(raw_map=raw_map, obstacle_map=obstacle_map, detection_probability=detection_probability)

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    subdir = _cache_subdir(cache_root, height, width, obstacle_indices, detection_probability)
    base_path = subdir / "base_contributions.npy"
    vis_path = subdir / "visibility_map.npy"
    meta_path = subdir / "meta.json"

    if subdir.exists() and not force_rebuild and base_path.exists() and vis_path.exists():
        # load as memory-mapped arrays to avoid copying into each worker
        try:
            vis = np.load(vis_path, mmap_mode="r")
            base = np.load(base_path, mmap_mode="r")
            static = StaticFeatures(
                raw_map=raw_map,
                obstacle_map=obstacle_map,
                detection_probability=detection_probability,
                visibility_map=vis,
                base_contributions=base,
            )
            return static
        except Exception:
            # fall through to rebuild on any error
            pass

    # compute and save
    static = StaticFeatures(raw_map=raw_map, obstacle_map=obstacle_map, detection_probability=detection_probability)

    # atomic write: write to tmp files then replace
    subdir_tmp = subdir.with_suffix(".tmp")
    if subdir_tmp.exists():
        try:
            # cleanup leftover tmp
            for p in subdir_tmp.iterdir():
                p.unlink()
            subdir_tmp.rmdir()
        except Exception:
            pass

    subdir_tmp.mkdir(parents=True, exist_ok=True)
    try:
        np.save(subdir_tmp / "visibility_map.npy", static.visibility_map)
        np.save(subdir_tmp / "base_contributions.npy", static.base_contributions)
        meta = {
            "height": int(height),
            "width": int(width),
            "obstacle_indices": None if obstacle_indices is None else list(map(int, obstacle_indices)),
            "detection_id": getattr(detection_probability, "__name__", str(detection_probability)),
        }
        with open(subdir_tmp / "meta.json", "w") as f:
            json.dump(meta, f)

        # atomic replace
        if subdir.exists():
            # remove old and replace
            for p in subdir.iterdir():
                p.unlink()
            subdir.rmdir()
        os.replace(str(subdir_tmp), str(subdir))
    except Exception:
        # best-effort cleanup
        try:
            for p in subdir_tmp.iterdir():
                p.unlink()
            subdir_tmp.rmdir()
        except Exception:
            pass

    return static