import argparse
import json
from pathlib import Path

import numpy as np


LOCAL_FEATURES = [
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="parts directory")
    root = Path(parser.parse_args().data)

    with open(root / "meta.json", "r") as f:
        meta = json.load(f)

    keys = meta["keys"]

    # Dataset dimensions
    first = np.load(root / f"{LOCAL_FEATURES[0]}.npy", mmap_mode="r")
    n_samples = first.shape[0]
    n_cells = first.shape[1]

    # Calculate number of output channels
    total_channels = 0

    for key in LOCAL_FEATURES:
        shape = keys[key]["shape"]
        width = shape[-1] if len(shape) > 2 else 1
        total_channels += width

    output_shape = (n_samples, n_cells, total_channels)

    print(f"Output shape: {output_shape}")
    print(f"Total channels: {total_channels}")
    print("Dtype: float32")
    print("Creating memory-mapped output...")

    output_path = root / "local_features.npy"

    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=output_shape,
    )

    # Process samples in chunks
    chunk_size = 1000

    channel = 0

    for key in LOCAL_FEATURES:
        print(f"\nWriting {key}...")

        arr = np.load(
            root / f"{key}.npy",
            mmap_mode="r",
        )

        if arr.ndim == 2:
            width = 1
        else:
            width = arr.shape[2]

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)

            chunk = arr[start:end]

            if chunk.ndim == 2:
                chunk = chunk[:, :, None]

            # Convert to float32 before writing
            output[start:end, :, channel:channel + width] = (
                chunk.astype(np.float32, copy=False)
            )

            if start % (chunk_size * 100) == 0:
                print(
                    f"  samples {start:,} / {n_samples:,}"
                )

        channel += width

    output.flush()

    print("\nDone!")
    print(f"File: {output_path}")
    print(f"Shape: {output.shape}")
    print(f"Dtype: {output.dtype}")


if __name__ == "__main__":
    main()