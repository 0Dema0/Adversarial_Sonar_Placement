from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

"""Inspect .npz dataset files and print each entry's name and type.

Usage:
  python dataset/check_dataset.py path/to/file.npz
  python dataset/check_dataset.py path/to/folder/  # lists all .npz in folder
"""

def inspect_npz(path: Path) -> None:
    try:
        with np.load(path, allow_pickle=True) as data:
            print(f"File: {path}")
            keys = list(data.keys())
            print(f"Entries: {len(keys)}")
            for k in keys:
                try:
                    v = data[k]
                except Exception as e:
                    print(f"  - {k}: <failed to load: {e}>")
                    continue

                # numpy arrays are the common case
                if isinstance(v, np.ndarray):
                    dtype = v.dtype
                    shape = v.shape
                    if dtype == object:
                        # show a few sample element types for object arrays
                        sample_types = set()
                        cnt = 0
                        for el in v.flat:
                            if el is None:
                                continue
                            sample_types.add(type(el).__name__)
                            cnt += 1
                            if cnt >= 5:
                                break
                        sample_types_str = ", ".join(sorted(sample_types)) if sample_types else "empty"
                        print(f"  - {k}: ndarray(dtype=object, shape={shape}, sample_types={sample_types_str})")
                    else:
                        print(f"  - {k}: ndarray(dtype={dtype}, shape={shape})")
                else:
                    print(f"  - {k}: {type(v).__name__}")
            print()
    except Exception as e:
        print(f"Failed to read {path}: {e}", file=sys.stderr)


def find_npz_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            files.extend(sorted(pth.glob("*.npz")))
        elif pth.is_file():
            files.append(pth)
        else:
            print(f"Skipping missing path: {pth}", file=sys.stderr)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect .npz dataset files and print entry names and types.")
    parser.add_argument("paths", nargs="+", help="Files or directories to inspect (.npz). Directories list *.npz.")
    args = parser.parse_args()

    files = find_npz_files(args.paths)
    if not files:
        print("No .npz files found.", file=sys.stderr)
        return 2

    for f in files:
        inspect_npz(f)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
