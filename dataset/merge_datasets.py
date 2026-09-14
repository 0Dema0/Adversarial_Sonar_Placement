import json
from pathlib import Path

import numpy as np


def _merge_memmap_pair(parts1: Path, parts2: Path, output: Path, name: str, n1: int, n2: int):
    src1 = parts1 / name
    src2 = parts2 / name
    out_path = output / name

    if not src1.exists() or not src2.exists():
        return

    mm1 = np.load(src1, mmap_mode="r")
    mm2 = np.load(src2, mmap_mode="r")

    if mm1.shape[0] < n1:
        raise ValueError(
            f"{name}: first dataset only has capacity {mm1.shape[0]}, but needs {n1}"
        )

    if mm2.shape[0] < n2:
        raise ValueError(
            f"{name}: second dataset only has capacity {mm2.shape[0]}, but needs {n2}"
        )

    if mm1.shape[1:] != mm2.shape[1:]:
        raise ValueError(f"{name}: shape mismatch: {mm1.shape} vs {mm2.shape}")

    total = n1 + n2
    new_shape = (total,) + mm1.shape[1:]

    print(f"{name}: {mm1.shape} + {mm2.shape} -> {new_shape}")

    out_mm = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=mm1.dtype,
        shape=new_shape,
    )

    out_mm[:n1] = mm1[:n1]
    out_mm[n1:n1 + n2] = mm2[:n2]
    out_mm.flush()

    del mm1
    del mm2
    del out_mm


def merge_dataset(parts1: Path, parts2: Path, output: Path, N1: int, N2: int):
    TOTAL = N1 + N2

    with open(parts1 / "meta.json", "r") as f:
        meta1 = json.load(f)

    with open(parts2 / "meta.json", "r") as f:
        meta2 = json.load(f)

    if int(meta1["processed"]) != N1:
        raise ValueError(
            f"First dataset says processed={meta1['processed']}, expected {N1}"
        )

    if int(meta2["processed"]) != N2:
        raise ValueError(
            f"Second dataset says processed={meta2['processed']}, expected {N2}"
        )

    keys1 = set(meta1["keys"])
    keys2 = set(meta2["keys"])

    if keys1 != keys2:
        raise ValueError(
            f"Dataset keys don't match.\n"
            f"Only in first: {keys1 - keys2}\n"
            f"Only in second: {keys2 - keys1}"
        )

    if output.exists():
        raise FileExistsError(
            f"Output directory already exists: {output}\n"
            f"Delete it first if you want to recreate it."
        )

    output.mkdir(parents=True)

    merged_keys = {}

    for key in sorted(keys1):
        k1 = meta1["keys"][key]
        k2 = meta2["keys"][key]

        dtype1 = np.dtype(k1["dtype"])
        dtype2 = np.dtype(k2["dtype"])

        shape1 = tuple(k1["shape"])
        shape2 = tuple(k2["shape"])

        if dtype1 != dtype2:
            raise ValueError(f"{key}: dtype mismatch: {dtype1} vs {dtype2}")

        if shape1[1:] != shape2[1:]:
            raise ValueError(f"{key}: shape mismatch: {shape1} vs {shape2}")

        if shape1[0] < N1:
            raise ValueError(
                f"{key}: first dataset only has capacity {shape1[0]}, but needs {N1}"
            )

        if shape2[0] < N2:
            raise ValueError(
                f"{key}: second dataset only has capacity {shape2[0]}, but needs {N2}"
            )

        new_shape = (TOTAL,) + shape1[1:]
        print(f"{key}: {shape1} + {shape2} -> {new_shape}")

        out_mm = np.lib.format.open_memmap(
            output / f"{key}.npy",
            mode="w+",
            dtype=dtype1,
            shape=new_shape,
        )

        mm1 = np.lib.format.open_memmap(parts1 / f"{key}.npy", mode="r")
        mm2 = np.lib.format.open_memmap(parts2 / f"{key}.npy", mode="r")

        out_mm[:N1] = mm1[:N1]
        out_mm[N1:N1 + N2] = mm2[:N2]
        out_mm.flush()

        del mm1
        del mm2
        del out_mm

        merged_keys[key] = {
            "shape": list(new_shape),
            "dtype": str(dtype1),
        }

    if (parts1 / "local_features.npy").exists() and (parts2 / "local_features.npy").exists():
        _merge_memmap_pair(parts1, parts2, output, "local_features.npy", N1, N2)
    elif (parts1 / "local_features.npy").exists() or (parts2 / "local_features.npy").exists():
        print(
            "Skipping local_features.npy merge because only one source dataset contains it."
        )

    merged_meta = {
        "total": TOTAL,
        "processed": TOTAL,
        "keys": merged_keys,
        "created": meta1.get("created"),
    }

    with open(output / "meta.json", "w") as f:
        json.dump(merged_meta, f, indent=2)

    return output


if __name__ == "__main__":
    parts1 = Path("data/best_dataset_complete_parts")
    parts2 = Path("data/best_dataset_1unknown_complete_parts")
    output = Path("data/best_dataset_known_and_unknown_complete_parts")

    N1 = 112205
    N2 = 112205

    merge_dataset(parts1, parts2, output, N1, N2)

    print()
    print("=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"First dataset:  {N1:,} samples")
    print(f"Second dataset: {N2:,} samples")
    print(f"Total:          {N1 + N2:,} samples")
    print(f"Output:         {output}")