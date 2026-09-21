"""Build a dataset whose target is the *extra* path cost caused by one unknown sonar.

For a sonar placement P with all sonars known, the planner cost is

    J0(P)    = shortest-path cost under the full cost map.

If sonar s is instead unknown at the start, the agent plans without s, discovers it
when within ``discover_distance`` and replans. Its realised cost is J1(P, s) and

    delta(P, s) = J1(P, s) - J0(P)  >= 0

is the regression target written to ``outputs[:, start_index]``. Every other key is
kept in the exact layout of the source parts directory (``known_sonar_mask`` = all
but s, ``unknown_sonar_mask`` = s, ``start_as_unknown_number`` = 1), so
``PartsDataset``, ``combine_local_features``, ``merge_datasets``, ``Trainer`` and
``LocalGlobalNet`` work unchanged. Three scalar keys are added for analysis:
``base_cost`` (J0), ``unknown_cost`` (J1) and ``source_row``.

Two ways to build it:

rollout  (recommended) - start from an ALL-KNOWN parts dataset and simulate the
         planner for every sonar (``--mode all``) or K random sonars per placement.
         The rollout before discovery is simulated step by step with the project's
         own IncrementalPlanner (so tie-breaking is identical). Once the sonar is
         discovered every sonar is known, so the remaining cost is taken from one
         reverse Dijkstra - this is exact and roughly halves the work.

    python -m dataset.build_unknown_delta_dataset rollout \
        --input data/best_dataset_complete_parts \
        --output data/delta_1unknown_all_parts \
        --mode all --workers 16

pair     - pure differencing of two existing, row-aligned datasets: an all-known one
         and a 1-unknown one derived from it with ``convert_unknown_parts``
         (``random``/``best`` = 1 row per source row, ``all`` = S rows per source row).
         No planning is done. Note that a ``best``-mode source only contains the
         worst sonar of each placement, which biases the delta distribution.

    python -m dataset.build_unknown_delta_dataset pair \
        --known data/best_dataset_complete_parts \
        --unknown data/best_dataset_1unknown_complete_parts \
        --output data/delta_1unknown_pair_parts
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from environment.cost import CostMap
from environment.perception import SonarKnowledge, update_sonar_visibility
from environment.planner import IncrementalPlanner
from environment.state import MapState

EXTRA_KEYS = {
    "base_cost": np.float64,     # J0: all sonars known
    "unknown_cost": np.float64,  # J1: realised cost with the one unknown sonar
    "source_row": np.int64,      # row in the source (all-known) dataset
}
# Rounding noise tolerated before a negative delta is treated as an error.
NEGATIVE_TOLERANCE = 1e-4
# Deltas below this (relative to J0) are rounding noise and stored as exactly 0.
ZERO_SNAP = 1e-5


# ---------------------------------------------------------------------------
# Parts-directory helpers
# ---------------------------------------------------------------------------

def _load_parts(parts_dir: Path) -> tuple[dict, int, dict[str, np.ndarray]]:
    meta_path = parts_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    with meta_path.open("r") as handle:
        meta = json.load(handle)
    rows = int(meta.get("processed", meta.get("total", 0)))
    keys = list(meta.get("keys", {}))
    if rows <= 0 or not keys:
        raise ValueError(f"Invalid or empty parts metadata: {meta_path}")
    arrays = {key: np.load(parts_dir / f"{key}.npy", mmap_mode="r") for key in keys}
    return meta, rows, arrays


def _open_output(
    output: Path,
    source_meta: dict,
    total: int,
    config: dict,
    overwrite: bool,
    resume: bool,
) -> tuple[dict[str, np.ndarray], dict, int]:
    """Create (or reopen for resume) the output memmaps. Returns (arrays, meta, input_rows_done)."""
    meta_path = output / "meta.json"
    if output.exists() and meta_path.exists() and resume:
        with meta_path.open("r") as handle:
            meta = json.load(handle)
        if meta.get("delta_config") != config or int(meta["total"]) != total:
            raise ValueError(
                "Cannot resume: existing output was built with a different configuration.\n"
                f"existing={meta.get('delta_config')}\nrequested={config}"
            )
        arrays = {
            key: np.lib.format.open_memmap(output / f"{key}.npy", mode="r+")
            for key in meta["keys"]
        }
        done = int(meta.get("input_rows_done", 0))
        print(f"Resuming {output}: {done} input rows / {meta['processed']} output rows already written")
        return arrays, meta, done

    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --overwrite or --resume")
        for path in list(output.glob("*.npy")) + [meta_path]:
            if path.exists():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)

    specs = {key: (tuple(v["shape"][1:]), np.dtype(v["dtype"])) for key, v in source_meta["keys"].items()}
    for key, dtype in EXTRA_KEYS.items():
        specs[key] = ((), np.dtype(dtype))

    arrays, key_meta = {}, {}
    for key, (tail, dtype) in sorted(specs.items()):
        shape = (total,) + tail
        arrays[key] = np.lib.format.open_memmap(output / f"{key}.npy", mode="w+", dtype=dtype, shape=shape)
        key_meta[key] = {"shape": list(shape), "dtype": str(dtype)}

    meta = {
        "total": total,
        "processed": 0,
        "input_rows_done": 0,
        "keys": key_meta,
        "delta_config": config,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_meta(output, meta)
    return arrays, meta, 0


def _write_meta(output: Path, meta: dict) -> None:
    tmp = output / "meta.json.tmp"
    with tmp.open("w") as handle:
        json.dump(meta, handle, indent=2)
    os.replace(tmp, output / "meta.json")


def _check_fixed_obstacles(obstacle_placement: np.ndarray, rows: int, chunk: int = 4096) -> list[int]:
    reference = np.asarray(obstacle_placement[0])
    for start in range(0, rows, chunk):
        block = np.asarray(obstacle_placement[start:min(rows, start + chunk)])
        if not np.all(block == reference):
            raise ValueError("Obstacle placement differs between rows; this builder assumes fixed obstacles.")
    return np.flatnonzero(reference).astype(int).tolist()


def _write_rows(
    dst: dict[str, np.ndarray],
    src: dict[str, np.ndarray],
    src_rows: np.ndarray,
    out_slice: slice,
    unknown_cells: np.ndarray,
    base: np.ndarray,
    unknown_cost: np.ndarray,
    start_index: int,
) -> np.ndarray:
    """Copy features for ``src_rows`` into ``out_slice`` and set masks/targets. Returns deltas."""
    n = len(src_rows)
    # Memmap fancy indexing is much faster on sorted unique indices.
    unique_rows, inverse = np.unique(src_rows, return_inverse=True)
    for key, target in dst.items():
        if key in EXTRA_KEYS:
            continue
        target[out_slice] = np.asarray(src[key][unique_rows])[inverse]

    sonar = np.asarray(dst["sonar_placement"][out_slice])
    cells = sonar.shape[1]
    unknown_mask = np.zeros((n, cells), dtype=bool)
    unknown_mask[np.arange(n), unknown_cells] = True
    if not np.all(sonar[np.arange(n), unknown_cells]):
        raise ValueError("Chosen unknown cell is not a sonar cell; source data is inconsistent.")

    delta = unknown_cost - base
    bad = np.isfinite(delta) & (delta < -NEGATIVE_TOLERANCE * np.maximum(1.0, np.abs(base)))
    if np.any(bad):
        raise ValueError(
            f"{int(bad.sum())} rows have a clearly negative delta (min {delta[bad].min():.4g}). "
            "The pair of costs is not consistent (misaligned rows or wrong --start-index?)."
        )
    # Snap rounding noise (float32 labels in pair mode) to an exact zero so the
    # "sonar did not matter" case is represented consistently.
    noise = np.isfinite(delta) & (delta < ZERO_SNAP * np.maximum(1.0, np.abs(base)))
    delta = np.where(noise, 0.0, delta)
    delta = np.where(np.isfinite(delta), delta, np.inf)

    outputs = np.zeros((n, cells), dtype=np.float32)
    outputs[:, start_index] = delta
    dst["outputs"][out_slice] = outputs
    dst["reachable_mask"][out_slice] = np.isfinite(outputs)
    dst["unknown_sonar_mask"][out_slice] = unknown_mask
    dst["known_sonar_mask"][out_slice] = sonar & ~unknown_mask
    sonar_count = sonar.sum(axis=1)
    dst["start_as_unknown_number"][out_slice] = 1
    dst["start_as_known_number"][out_slice] = sonar_count - 1
    dst["best_sonar_split"][out_slice] = False
    dst["base_cost"][out_slice] = base
    dst["unknown_cost"][out_slice] = unknown_cost
    dst["source_row"][out_slice] = src_rows
    return delta


# ---------------------------------------------------------------------------
# Planner rollouts
# ---------------------------------------------------------------------------

def distances_to_goal(cost_field: np.ndarray, neighbor_map: list[list[int]], obstacles: set[int], goal_index: int) -> np.ndarray:
    """Shortest-path cost from every cell to the goal, with the same convention as
    ``environment.path.dijkstra``: moving into cell v costs cost_field[v]."""
    cost = cost_field.tolist()
    dist = [float("inf")] * len(cost)
    dist[goal_index] = 0.0
    heap = [(0.0, goal_index)]
    while heap:
        d_v, v = heapq.heappop(heap)
        if d_v > dist[v]:
            continue
        candidate = d_v + cost[v]
        for u in neighbor_map[v]:
            if u in obstacles:
                continue
            if candidate < dist[u]:
                dist[u] = candidate
                heapq.heappush(heap, (candidate, u))
    return np.asarray(dist, dtype=np.float64)


def rollout_with_unknowns(
    static,
    cost_map: CostMap,
    dist_full: np.ndarray,
    known_ids: np.ndarray,
    unknown_ids: np.ndarray,
    start_index: int,
    goal_index: int,
    discover_distance: int = 3,
) -> float:
    """Realised cost of the incremental planner, identical to ``Features`` /
    ``convert_unknown_parts._rollout``, but stops simulating as soon as every sonar is
    known and adds the exact remaining shortest-path cost instead."""
    planner = IncrementalPlanner(cost_map)
    state = MapState(
        obstacle_map=static.obstacle_map,
        sonar_knowledge=SonarKnowledge(known_ids, unknown_ids),
        agent_index=start_index,
        goal_index=goal_index,
    )
    sonar_cells = np.asarray(cost_map.sonar_indices, dtype=np.int64)
    score = 0.0
    while state.agent_index != state.goal_index:
        update_sonar_visibility(
            raw_map=static.raw_map,
            agent_index=state.agent_index,
            sonar_knowledge=state.sonar,
            discover_distance=discover_distance,
            sonar_indices=sonar_cells,
        )
        if len(state.sonar.unknown) == 0:
            return score + float(dist_full[state.agent_index])
        path, _, _ = planner.plan(state)
        if not path:
            return float("inf")
        if len(path) < 2:
            break
        state.agent_index = path[1]
        score += cost_map.static_cost_map[state.agent_index]
    return float(score)


_WORKER: dict = {}


def _worker_init(height: int, width: int, obstacle_indices: list[int], cache_dir: str | None) -> None:
    from dataset.static_cache import build_static

    static = build_static(height=height, width=width, obstacle_indices=obstacle_indices, cache_dir=cache_dir)
    _WORKER["static"] = static
    _WORKER["obstacles"] = set(int(i) for i in static.obstacles)


def _evaluate_batch(tasks: list[tuple]) -> list[tuple]:
    """tasks: (row, sonar_cells, goal, start, sonar_ids_to_hide, verify)."""
    static = _WORKER["static"]
    results = []
    for row, sonar_cells, goal, start, hide_ids, verify in tasks:
        sonar_count = len(sonar_cells)
        cost_map = CostMap(
            static.obstacle_map,
            static.detection_probability,
            sonar_number=sonar_count,
            sonar_indices=list(sonar_cells),
            base_contributions=static.base_contributions,
        )
        dist_full = distances_to_goal(cost_map.static_cost_map, static.raw_map.neighbor_map, _WORKER["obstacles"], goal)
        base = float(dist_full[start])
        all_ids = np.arange(sonar_count, dtype=np.int64)
        for sonar_id in hide_ids:
            known = np.delete(all_ids, sonar_id)
            unknown = np.array([sonar_id], dtype=np.int64)
            j1 = rollout_with_unknowns(static, cost_map, dist_full, known, unknown, start, goal)
            if verify:
                from dataset.convert_unknown_parts import _rollout

                ref_j0 = _rollout(static, cost_map, all_ids, np.array([], dtype=np.int64), start, goal)
                ref_j1 = _rollout(static, cost_map, known, unknown, start, goal)
                if abs(ref_j0 - base) > 1e-6 or abs(ref_j1 - j1) > 1e-6:
                    raise AssertionError(
                        f"Verification failed at row {row}, sonar {sonar_id}: "
                        f"J0 {base} vs {ref_j0}, J1 {j1} vs {ref_j1}"
                    )
            results.append((row, int(sonar_id), base, j1))
    return results


def build_by_rollout(args: argparse.Namespace) -> None:
    source_dir, output = Path(args.input), Path(args.output)
    source_meta, rows, src = _load_parts(source_dir)
    if args.max_rows is not None:
        rows = min(rows, args.max_rows)

    cells = int(src["sonar_placement"].shape[1])
    if args.height * args.width != cells:
        raise ValueError(f"--height * --width must equal {cells}")
    obstacle_indices = _check_fixed_obstacles(src["obstacle_placement"], rows)

    sonar_counts = np.asarray(src["sonar_number"][:rows])
    if not np.all(sonar_counts == sonar_counts[0]):
        raise ValueError("Sonar count varies between rows; not supported.")
    sonar_count = int(sonar_counts[0])
    per_row = sonar_count if args.mode == "all" else args.per_row
    if not 1 <= per_row <= sonar_count:
        raise ValueError(f"--per-row must be in [1, {sonar_count}]")
    total = rows * per_row

    config = {
        "method": "rollout", "source": str(source_dir), "mode": args.mode, "per_row": per_row,
        "rows": rows, "seed": args.seed, "start_index": args.start_index,
    }
    dst, meta, done = _open_output(output, source_meta, total, config, args.overwrite, args.resume)

    # Build the static cache once in the main process so workers only mmap-load it.
    from dataset.static_cache import build_static

    build_static(args.height, args.width, obstacle_indices, cache_dir=args.cache_dir)
    init_args = (args.height, args.width, obstacle_indices, args.cache_dir)
    workers = max(1, args.workers)
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=init_args) if workers > 1 else None
    if pool is None:
        _worker_init(*init_args)

    stats = {"max_label_err": 0.0}
    t_start = time.time()
    rows_at_start = done
    try:
        for chunk_start in tqdm(range(done, rows, args.chunk_rows)):
            chunk_end = min(rows, chunk_start + args.chunk_rows)
            sonar_block = np.asarray(src["sonar_placement"][chunk_start:chunk_end])
            goal_block = np.asarray(src["goal_onehot"][chunk_start:chunk_end])
            unknown_block = np.asarray(src["unknown_sonar_mask"][chunk_start:chunk_end])
            if unknown_block.any():
                raise ValueError("The rollout input must be an ALL-KNOWN dataset (found unknown sonars).")

            tasks = []
            for i, row in enumerate(range(chunk_start, chunk_end)):
                if args.mode == "all":
                    hide = list(range(sonar_count))
                else:
                    rng = np.random.default_rng([args.seed, row])
                    hide = sorted(rng.choice(sonar_count, size=per_row, replace=False).tolist())
                verify = row < args.verify_rows
                tasks.append((row, np.flatnonzero(sonar_block[i]).tolist(), int(np.argmax(goal_block[i])), args.start_index, hide, verify))

            batch = max(1, len(tasks) // (workers * 4))
            batches = [tasks[i:i + batch] for i in range(0, len(tasks), batch)]
            mapper = pool.map(_evaluate_batch, batches) if pool else map(_evaluate_batch, batches)
            results = [item for part in mapper for item in part]

            results.sort(key=lambda r: (r[0], r[1]))
            src_rows = np.array([r[0] for r in results], dtype=np.int64)
            sonar_ids = np.array([r[1] for r in results], dtype=np.int64)
            base = np.array([r[2] for r in results], dtype=np.float64)
            j1 = np.array([r[3] for r in results], dtype=np.float64)
            unknown_cells = np.array(
                [np.flatnonzero(sonar_block[r - chunk_start])[s] for r, s in zip(src_rows, sonar_ids)],
                dtype=np.int64,
            )

            stored = np.asarray(src["outputs"][chunk_start:chunk_end, args.start_index], dtype=np.float64)
            label_err = np.abs(stored[src_rows - chunk_start] - base)
            stats["max_label_err"] = max(stats["max_label_err"], float(np.max(label_err)))
            if np.max(label_err) > 1e-3 * max(1.0, float(np.max(np.abs(base)))):
                raise ValueError(
                    f"Recomputed all-known cost differs from stored outputs[:, {args.start_index}] "
                    f"(max err {np.max(label_err):.4g}). Is --start-index correct?"
                )

            out_slice = slice(chunk_start * per_row, chunk_end * per_row)
            _write_rows(dst, src, src_rows, out_slice, unknown_cells, base, j1, args.start_index)

            for array in dst.values():
                array.flush()
            meta["processed"] = chunk_end * per_row
            meta["input_rows_done"] = chunk_end
            _write_meta(output, meta)

            elapsed = time.time() - t_start
            rate = (chunk_end - rows_at_start) / max(elapsed, 1e-9)
            eta = (rows - chunk_end) / max(rate, 1e-9)
            print(f"{chunk_end}/{rows} source rows | {meta['processed']}/{total} outputs | "
                  f"{rate:.1f} rows/s | ETA {eta / 60:.1f} min", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    print(f"Max |recomputed J0 - stored label| this run: {stats['max_label_err']:.3g}")
    _print_summary(output)


# ---------------------------------------------------------------------------
# Differencing two aligned datasets
# ---------------------------------------------------------------------------

def build_by_pairing(args: argparse.Namespace) -> None:
    known_dir, unknown_dir, output = Path(args.known), Path(args.unknown), Path(args.output)
    _, n_known, known = _load_parts(known_dir)
    unknown_meta, n_unknown, unknown = _load_parts(unknown_dir)
    if n_unknown % n_known != 0:
        raise ValueError(f"Unknown rows ({n_unknown}) must be a multiple of known rows ({n_known}).")
    ratio = n_unknown // n_known
    total = n_unknown if args.max_rows is None else min(n_unknown, args.max_rows * ratio)

    config = {"method": "pair", "known": str(known_dir), "unknown": str(unknown_dir), "rows": total, "start_index": args.start_index}
    dst, meta, done_rows = _open_output(output, unknown_meta, total, config, args.overwrite, args.resume)
    start = args.start_index
    chunk = args.chunk_rows * ratio

    for out_start in range(done_rows * ratio, total, chunk):
        out_end = min(total, out_start + chunk)
        u_rows = np.arange(out_start, out_end)
        k_rows = u_rows // ratio
        k_unique, k_inverse = np.unique(k_rows, return_inverse=True)

        for key in ("sonar_placement", "obstacle_placement", "goal_onehot"):
            if not np.array_equal(np.asarray(unknown[key][out_start:out_end]), np.asarray(known[key][k_unique])[k_inverse]):
                raise ValueError(f"'{key}' does not match between the datasets around row {out_start}; they are not aligned.")
        if np.asarray(known["unknown_sonar_mask"][k_unique]).any():
            raise ValueError("--known dataset contains unknown sonars.")
        unknown_mask = np.asarray(unknown["unknown_sonar_mask"][out_start:out_end])
        if not np.all(unknown_mask.sum(axis=1) == 1):
            raise ValueError("--unknown dataset must have exactly one unknown sonar per row.")

        base = np.asarray(known["outputs"][k_unique, start], dtype=np.float64)[k_inverse]
        j1 = np.asarray(unknown["outputs"][out_start:out_end, start], dtype=np.float64)
        unknown_cells = np.argmax(unknown_mask, axis=1)
        _write_rows(dst, unknown, u_rows, slice(out_start, out_end), unknown_cells, base, j1, start)
        dst["source_row"][out_start:out_end] = k_rows

        for array in dst.values():
            array.flush()
        meta["processed"] = out_end
        meta["input_rows_done"] = out_end // ratio
        _write_meta(output, meta)
        print(f"{out_end}/{total} rows", flush=True)

    _print_summary(output)


# ---------------------------------------------------------------------------

def _print_summary(output: Path) -> None:
    with (output / "meta.json").open("r") as handle:
        meta = json.load(handle)
    n = int(meta["processed"])
    if n == 0:
        return
    base = np.load(output / "base_cost.npy", mmap_mode="r")[:n]
    j1 = np.load(output / "unknown_cost.npy", mmap_mode="r")[:n]
    delta = np.asarray(np.load(output / "outputs.npy", mmap_mode="r")[:n, meta["delta_config"]["start_index"]], dtype=np.float64)
    finite = np.isfinite(delta)
    d = delta[finite]
    q = np.percentile(d, [50, 90, 99]) if d.size else [np.nan] * 3
    print("\nDelta dataset summary")
    print(f"  rows            : {n}  (non-finite: {int((~finite).sum())})")
    print(f"  J0 mean         : {np.mean(base[finite]):.4f}")
    print(f"  delta mean/max  : {d.mean():.4f} / {d.max():.4f}")
    print(f"  delta p50/p90/p99: {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}")
    print(f"  exactly zero    : {np.mean(d <= 1e-9) * 100:.1f}%")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output", required=True, help="Output _parts directory")
        p.add_argument("--start-index", type=int, default=0, help="Start cell of the single-value datasets")
        p.add_argument("--max-rows", type=int, default=None, help="Only use the first N source (all-known) rows")
        p.add_argument("--chunk-rows", type=int, default=2000, help="Source rows per write/checkpoint")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--resume", action="store_true", help="Continue an interrupted run with identical settings")

    roll = sub.add_parser("rollout", help="Simulate the planner on an all-known dataset")
    roll.add_argument("--input", required=True, help="All-known _parts directory")
    roll.add_argument("--mode", choices=("all", "random"), default="all",
                      help="all: one row per sonar; random: --per-row random sonars per placement")
    roll.add_argument("--per-row", type=int, default=1)
    roll.add_argument("--seed", type=int, default=0)
    roll.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    roll.add_argument("--verify-rows", type=int, default=10,
                      help="Cross-check the first N rows against convert_unknown_parts._rollout")
    roll.add_argument("--height", type=int, default=20)
    roll.add_argument("--width", type=int, default=20)
    roll.add_argument("--cache-dir", default="data/precomp")
    common(roll)

    pair = sub.add_parser("pair", help="Difference an all-known and an aligned 1-unknown dataset")
    pair.add_argument("--known", required=True)
    pair.add_argument("--unknown", required=True)
    common(pair)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "rollout":
        build_by_rollout(args)
    else:
        build_by_pairing(args)


if __name__ == "__main__":
    main()
