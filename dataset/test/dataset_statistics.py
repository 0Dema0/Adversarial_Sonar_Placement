"""
This module contains functions to compute statistics for the test dataset.
"""
import json
import numpy as np
from pathlib import Path


def compute_statistics(parts_dir: str, keys: list[str] | None = None):
    """
    Compute summary statistics for specified keys in a dataset parts folder.
    
    Args:
        parts_dir: Path to the dataset parts directory containing .npy files and meta.json
        keys: List of keys to compute statistics for. If None, computes for all keys.
    
    Returns:
        Dictionary with statistics for each key, including min/max value locations and
        corresponding sample indices. If sonar_placement is available, active sonar
        positions are also reported for the min/max samples.
    """
    parts_path = Path(parts_dir)
    meta_path = parts_path / "meta.json"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {parts_dir}")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    available_keys = list(meta.get("keys", {}).keys())
    
    if keys is None:
        keys = available_keys
    else:
        invalid_keys = set(keys) - set(available_keys)
        if invalid_keys:
            raise ValueError(f"Invalid keys: {invalid_keys}. Available keys: {available_keys}")
    
    sonar_arr = None
    sonar_path = parts_path / "sonar_placement.npy"
    if sonar_path.exists():
        sonar_arr = np.load(sonar_path, mmap_mode="r")

    stats = {}
    for key in keys:
        npy_path = parts_path / f"{key}.npy"
        if not npy_path.exists():
            print(f"Warning: {key}.npy not found, skipping")
            continue
        
        arr = np.load(npy_path, mmap_mode="r")
        
        # Flatten array for global statistics and keep arg locations.
        flat = arr.flatten()
        min_flat_idx = int(np.argmin(flat))
        max_flat_idx = int(np.argmax(flat))
        min_index = tuple(int(i) for i in np.unravel_index(min_flat_idx, arr.shape))
        max_index = tuple(int(i) for i in np.unravel_index(max_flat_idx, arr.shape))

        min_sample_idx = int(min_index[0]) if len(min_index) > 0 else 0
        max_sample_idx = int(max_index[0]) if len(max_index) > 0 else 0
        
        stats[key] = {
            "mean": float(np.mean(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "min_index": min_index,
            "max_index": max_index,
            "min_sample_idx": min_sample_idx,
            "max_sample_idx": max_sample_idx,
        }

        if sonar_arr is not None and sonar_arr.ndim >= 2:
            if 0 <= min_sample_idx < sonar_arr.shape[0]:
                stats[key]["min_sample_sonar_indices"] = np.flatnonzero(
                    sonar_arr[min_sample_idx]
                ).astype(int).tolist()
            if 0 <= max_sample_idx < sonar_arr.shape[0]:
                stats[key]["max_sample_sonar_indices"] = np.flatnonzero(
                    sonar_arr[max_sample_idx]
                ).astype(int).tolist()
    
    return stats


def print_statistics(stats: dict):
    """Pretty print statistics dictionary."""
    for key, values in stats.items():
        print(f"\n{key}:")
        print(f"  Mean: {values['mean']:.6f}")
        print(f"  Min:  {values['min']:.6f}")
        print(f"  Max:  {values['max']:.6f}")
        print(f"  Min index: {values['min_index']} (sample {values['min_sample_idx']})")
        print(f"  Max index: {values['max_index']} (sample {values['max_sample_idx']})")

        if "min_sample_sonar_indices" in values:
            print(
                f"  Min sample sonar active indices: {values['min_sample_sonar_indices']}"
            )
        if "max_sample_sonar_indices" in values:
            print(
                f"  Max sample sonar active indices: {values['max_sample_sonar_indices']}"
            )

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute statistics for dataset parts.")
    parser.add_argument("--parts_dir", type=str, required=True, help="Path to the dataset parts directory")
    parser.add_argument("--keys", type=str, nargs="+", default=None, help="Keys to compute statistics for (default: all keys)")
    
    args = parser.parse_args()
    
    stats = compute_statistics(args.parts_dir, args.keys)
    print_statistics(stats)

if __name__ == "__main__":
    main()