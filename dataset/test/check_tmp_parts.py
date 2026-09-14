"""Validate a streaming parts directory produced by generate_dataset --stream-output.

Checks performed:
- meta.json exists and parseable
- each key in meta.keys has a corresponding .npy file
- array shapes and dtypes match meta (when available)
- basic integrity: NaN/Inf counts and a sampled "nonzero rows" estimate

Usage:
  python -m dataset.check_tmp_parts --parts-dir data/tmp3_parts
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Dict, Any

import numpy as np


def _fmt(x: float) -> str:
    if x is None:
        return "?"
    if math.isfinite(x):
        return f"{x:.3f}"
    return str(x)


def check_parts(parts_dir: str, sample_rows: int = 1000) -> int:
    p = pathlib.Path(parts_dir)
    if not p.exists():
        print(f"ERROR: parts directory not found: {p}")
        return 2

    meta_path = p / "meta.json"
    if not meta_path.exists():
        print(f"ERROR: meta.json not found in {p}")
        return 2

    try:
        meta = json.load(open(meta_path, "r"))
    except Exception as e:
        print(f"ERROR: failed to parse meta.json: {e}")
        return 2

    total = meta.get("total")
    keys_meta: Dict[str, Any] = meta.get("keys") if isinstance(meta.get("keys"), dict) else {}

    print(f"parts_dir: {p}")
    print(f"meta.total: {total}")
    print(f"keys in meta: {len(keys_meta)}")

    ok = True
    summaries = []

    # preload obstacle masks if present so we can treat infs at obstacle positions as OK
    obstacle_placement_arr = None
    neighbor_obstacle_arr = None
    try:
        if (p / "obstacle_placement.npy").exists():
            obstacle_placement_arr = np.load(p / "obstacle_placement.npy", mmap_mode="r")
    except Exception:
        obstacle_placement_arr = None
    try:
        if (p / "neighbor_obstacle.npy").exists():
            neighbor_obstacle_arr = np.load(p / "neighbor_obstacle.npy", mmap_mode="r")
    except Exception:
        neighbor_obstacle_arr = None

    for key, info in sorted(keys_meta.items()):
        fn = p / f"{key}.npy"
        s = {
            "key": key,
            "path": str(fn.name),
            "exists": fn.exists(),
            "shape": None,
            "dtype": None,
            "expected_shape": None,
            "expected_dtype": None,
            "shape_ok": None,
            "dtype_ok": None,
            "nan_count": None,
            "inf_count": None,
            "estimated_nonzero_rows": None,
            "notes": [],
        }

        if not fn.exists():
            s["notes"].append("missing file")
            ok = False
            summaries.append(s)
            continue

        try:
            arr = np.load(fn, mmap_mode="r")
        except Exception as e:
            s["notes"].append(f"failed to load: {e}")
            ok = False
            summaries.append(s)
            continue

        s["shape"] = tuple(int(x) for x in arr.shape)
        s["dtype"] = str(arr.dtype)

        if isinstance(info, dict) and info.get("shape"):
            s["expected_shape"] = tuple(int(x) for x in info.get("shape"))
            s["shape_ok"] = (s["shape"] == s["expected_shape"])
            if not s["shape_ok"]:
                s["notes"].append("shape mismatch")
                ok = False

        if isinstance(info, dict) and info.get("dtype"):
            try:
                expdtype = np.dtype(info.get("dtype"))
            except Exception:
                expdtype = None
            s["expected_dtype"] = str(expdtype) if expdtype is not None else info.get("dtype")
            s["dtype_ok"] = (expdtype is None) or (arr.dtype == expdtype)
            if s["dtype_ok"] is False:
                s["notes"].append("dtype mismatch")
                ok = False

        # check first-dimension consistency with meta.total when possible
        if total is not None:
            try:
                if int(s["shape"][0]) != int(total):
                    s["notes"].append(f"first-dim != meta.total ({s['shape'][0]} != {total})")
                    ok = False
            except Exception:
                pass

        # basic numeric diagnostics (only for numeric dtypes)
        kind = arr.dtype.kind
        if kind in "f":
            try:
                nan_count = int(np.isnan(arr).sum())
                inf_count = int(np.isinf(arr).sum())
            except Exception:
                nan_count = None
                inf_count = None
            s["nan_count"] = nan_count
            s["inf_count"] = inf_count
            if nan_count and nan_count > 0:
                s["notes"].append(f"nan_count={nan_count}")
                ok = False
            if inf_count and inf_count > 0:
                # allow infs if they correspond to obstacle placements
                handled = False
                if key == "cost" and obstacle_placement_arr is not None and arr.shape == obstacle_placement_arr.shape:
                    inf_mask = np.isinf(arr)
                    extra_inf = np.logical_and(inf_mask, np.logical_not(obstacle_placement_arr))
                    if np.any(extra_inf):
                        s["notes"].append(f"inf_count={inf_count} (some inf outside obstacle placements)")
                        ok = False
                    else:
                        s["notes"].append(f"inf_count={inf_count} (all at obstacle positions)")
                    handled = True

                if not handled and key == "neighbor_cost" and neighbor_obstacle_arr is not None and arr.shape == neighbor_obstacle_arr.shape:
                    inf_mask = np.isinf(arr)
                    extra_inf = np.logical_and(inf_mask, np.logical_not(neighbor_obstacle_arr))
                    if np.any(extra_inf):
                        s["notes"].append(f"inf_count={inf_count} (some inf not matching neighbor_obstacle)")
                        ok = False
                    else:
                        s["notes"].append(f"inf_count={inf_count} (all at neighbor_obstacle positions)")
                    handled = True

                if not handled:
                    s["notes"].append(f"inf_count={inf_count}")
                    ok = False

        # estimate how many rows appear to be "filled" by sampling
        nrows = s["shape"][0]
        sample_n = min(int(sample_rows), nrows)
        try:
            if nrows <= sample_n:
                # full sweep
                if arr.ndim == 1:
                    nonzero_rows = np.count_nonzero(arr != 0)
                    s["estimated_nonzero_rows"] = int(nonzero_rows)
                else:
                    any_nonzero = np.any(arr != 0, axis=tuple(range(1, arr.ndim)))
                    s["estimated_nonzero_rows"] = int(np.count_nonzero(any_nonzero))
            else:
                # sample evenly
                idx = np.linspace(0, nrows - 1, sample_n, dtype=int)
                count = 0
                for i in idx:
                    row = arr[i]
                    if row.ndim == 0:
                        nonz = bool(row != 0)
                    else:
                        nonz = bool(np.any(row != 0))
                    if nonz:
                        count += 1
                frac = count / len(idx) if len(idx) else 0.0
                s["estimated_nonzero_rows"] = int(frac * nrows)
        except Exception as e:
            s["notes"].append(f"sampling failed: {e}")

        summaries.append(s)

    # print compact report
    print("\nSummary:\n")
    for s in summaries:
        notes = ", ".join(s.get("notes", [])) or "OK"
        print(
            f"{s['key']}: shape={s['shape']} dtype={s['dtype']} expected_shape={s['expected_shape']} expected_dtype={s['expected_dtype']} nnz_rows_est={s['estimated_nonzero_rows']} nan={s['nan_count']} inf={s['inf_count']} -> {notes}"
        )

    if ok:
        print("\nAll checks passed.")
        return 0
    else:
        print("\nSome checks FAILED. See notes above.")
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", default="data/tmp3_parts")
    parser.add_argument("--sample-rows", type=int, default=1000, help="How many rows to sample for nonzero estimation (evenly spaced)")
    args = parser.parse_args(argv)

    return check_parts(args.parts_dir, sample_rows=args.sample_rows)


if __name__ == "__main__":
    raise SystemExit(main())
