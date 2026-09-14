"""Convert a streaming dataset to one with more initially unknown sonars.

The input must be a ``generate_dataset(..., stream_output=True)`` parts
 directory.  Sonar locations are recovered from ``sonar_placement.npy`` and
existing known/unknown status from the corresponding masks.

Examples:
    python -m dataset.convert_unknown_parts \
        --input data/training_dataset_ALLKNOWN_parts \
        --output data/training_dataset_UNKNOWN1_parts \
        --unknown-count 1 --mode random

    python -m dataset.convert_unknown_parts \
        --input data/training_dataset_UNKNOWN1_parts \
        --output data/training_dataset_UNKNOWN2_parts \
        --unknown-count 2 --mode best --start-index 0

    python -m dataset.convert_unknown_parts \
        --input data/training_dataset_ALLKNOWN_parts \
        --output data/training_dataset_UNKNOWN2_ALL_parts \
        --unknown-count 2 --mode all

``all`` writes one output sample for every possible unknown-sonar split.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from environment.features import StaticFeatures
from environment.map import ObstacleMap, RawMap
from environment.perception import SonarKnowledge, update_sonar_visibility
from environment.planner import IncrementalPlanner
from environment.cost import CostMap
from environment.utils import detection_probability


def _load_meta(parts_dir: Path) -> tuple[dict, int, set[str]]:
    meta_path = parts_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    with meta_path.open("r") as handle:
        meta = json.load(handle)
    total = int(meta.get("processed", meta.get("total", 0)))
    keys = set(meta.get("keys", {}))
    if total <= 0 or not keys:
        raise ValueError(f"Invalid or empty parts metadata: {meta_path}")
    missing = [key for key in keys if not (parts_dir / f"{key}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"Missing arrays in {parts_dir}: {missing}")
    return meta, total, keys


def _open_arrays(parts_dir: Path, keys: set[str], mode: str) -> dict[str, np.ndarray]:
    return {key: np.load(parts_dir / f"{key}.npy", mmap_mode=mode) for key in keys}


def _infer_start(outputs: np.ndarray, explicit_start: int | None) -> int:
    if explicit_start is not None:
        return int(explicit_start)
    nonzero = np.flatnonzero(outputs != 0)
    if len(nonzero) != 1:
        raise ValueError(
            "Could not infer a single start index from outputs. "
            "Pass --start-index (this converter targets single-value parts datasets)."
        )
    return int(nonzero[0])


def _rollout(
    static: StaticFeatures,
    cost_map: CostMap,
    known_ids: np.ndarray,
    unknown_ids: np.ndarray,
    start_index: int,
    goal_index: int,
) -> float:
    """Return the same path cost used by Features for one start and split."""
    planner = IncrementalPlanner(cost_map)
    state = __import__("environment.state", fromlist=["MapState"]).MapState(
        obstacle_map=static.obstacle_map,
        sonar_knowledge=SonarKnowledge(known_ids, unknown_ids),
        agent_index=start_index,
        goal_index=goal_index,
    )
    score = 0.0
    sonar_cells = np.asarray(cost_map.sonar_indices, dtype=np.int64)

    while state.agent_index != state.goal_index:
        update_sonar_visibility(
            raw_map=static.raw_map,
            agent_index=state.agent_index,
            sonar_knowledge=state.sonar,
            discover_distance=3,
            sonar_indices=sonar_cells,
        )
        path, _, _ = planner.plan(state)
        if not path:
            return float("inf")
        if len(path) < 2:
            break
        state.agent_index = path[1]
        score += cost_map.static_cost_map[state.agent_index]
    return float(score)


def _choose_unknowns(
    static: StaticFeatures,
    sonar_cells: np.ndarray,
    existing_unknown_cells: np.ndarray,
    target_count: int,
    start_index: int,
    goal_index: int,
    mode: str,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, float]]:
    sonar_count = len(sonar_cells)
    existing_unknown_ids = np.flatnonzero(np.isin(sonar_cells, existing_unknown_cells)).astype(np.int64)
    known_ids = np.setdiff1d(np.arange(sonar_count, dtype=np.int64), existing_unknown_ids)
    additions = target_count - len(existing_unknown_ids)
    if additions < 0:
        raise ValueError(
            f"Target unknown count {target_count} is smaller than the row's "
            f"existing count {len(existing_unknown_ids)}"
        )
    if additions > len(known_ids):
        raise ValueError(f"Cannot mark {target_count} of {sonar_count} sonars unknown")

    cost_map = CostMap(
        static.obstacle_map,
        static.detection_probability,
        sonar_number=sonar_count,
        sonar_indices=sonar_cells.tolist(),
        base_contributions=static.base_contributions,
    )

    if mode == "random":
        added = rng.choice(known_ids, size=additions, replace=False)
        unknown_ids = np.sort(np.concatenate((existing_unknown_ids, added))).astype(np.int64)
        score = _rollout(static, cost_map, np.setdiff1d(np.arange(sonar_count), unknown_ids), unknown_ids, start_index, goal_index)
        return [(unknown_ids, score)]

    best_unknown: np.ndarray | None = None
    best_score = -float("inf")
    all_splits: list[tuple[np.ndarray, float]] = []
    for added_tuple in itertools.combinations(known_ids.tolist(), additions):
        candidate = np.sort(np.concatenate((existing_unknown_ids, np.asarray(added_tuple, dtype=np.int64))))
        candidate_known = np.setdiff1d(np.arange(sonar_count, dtype=np.int64), candidate)
        score = _rollout(static, cost_map, candidate_known, candidate, start_index, goal_index)
        if mode == "all":
            all_splits.append((candidate, score))
        if score > best_score:
            best_score = score
            best_unknown = candidate

    if mode == "all":
        return all_splits
    assert best_unknown is not None
    return [(best_unknown, best_score)]


def convert_parts(
    input_dir: str | Path,
    output_dir: str | Path,
    target_unknown_count: int,
    mode: str = "random",
    height: int = 20,
    width: int = 20,
    start_index: int | None = None,
    seed: int | None = None,
    overwrite: bool = False,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    meta, total, keys = _load_meta(input_path)
    if target_unknown_count < 0:
        raise ValueError("--unknown-count must be non-negative")
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}; use --overwrite")
        for path in output_path.glob("*.npy"):
            path.unlink()
        meta_output = output_path / "meta.json"
        if meta_output.exists():
            meta_output.unlink()
    output_path.mkdir(parents=True, exist_ok=True)

    source = _open_arrays(input_path, keys, "r")
    n_cells = int(source["sonar_placement"].shape[1])
    if height * width != n_cells:
        raise ValueError(f"--height * --width must equal {n_cells}, got {height} * {width}")
    obstacle_indices = np.flatnonzero(source["obstacle_placement"][0]).astype(int).tolist()
    raw_map = RawMap(width=width, height=height)
    obstacle_map = ObstacleMap(raw_map, obstacle_indices=obstacle_indices)
    static = StaticFeatures(raw_map, obstacle_map, detection_probability=detection_probability)
    rng = np.random.default_rng(seed)

    output_total = 0
    if mode == "all":
        for row_index in range(total):
            sonar_cells = np.flatnonzero(source["sonar_placement"][row_index]).astype(np.int64)
            existing_unknown_cells = np.flatnonzero(source["unknown_sonar_mask"][row_index]).astype(np.int64)
            existing_unknown_count = int(np.isin(sonar_cells, existing_unknown_cells).sum())
            additions = target_unknown_count - existing_unknown_count
            if additions < 0 or additions > len(sonar_cells) - existing_unknown_count:
                raise ValueError(
                    f"Cannot mark {target_unknown_count} of {len(sonar_cells)} sonars unknown "
                    f"for row {row_index}"
                )
            output_total += math.comb(len(sonar_cells) - existing_unknown_count, additions)
    else:
        output_total = total

    destination: dict[str, np.ndarray] = {}
    output_meta = {"total": output_total, "processed": output_total, "keys": {}, "created_from": str(input_path)}
    for key in sorted(keys):
        source_array = source[key]
        shape = (output_total,) + source_array.shape[1:]
        destination[key] = np.lib.format.open_memmap(
            output_path / f"{key}.npy", mode="w+", dtype=source_array.dtype, shape=shape
        )
        output_meta["keys"][key] = {"shape": list(shape), "dtype": str(source_array.dtype)}

    output_index = 0
    for row_index in range(total):
        sonar_cells = np.flatnonzero(source["sonar_placement"][row_index]).astype(np.int64)
        existing_unknown_cells = np.flatnonzero(source["unknown_sonar_mask"][row_index]).astype(np.int64)
        goal_index = int(np.flatnonzero(source["goal_onehot"][row_index])[0])
        row_start = _infer_start(source["outputs"][row_index], start_index)
        splits = _choose_unknowns(
            static, sonar_cells, existing_unknown_cells, target_unknown_count,
            row_start, goal_index, mode, rng,
        )
        for unknown_ids, score in splits:
            for key, target in destination.items():
                target[output_index] = source[key][row_index]

            unknown_cells = sonar_cells[unknown_ids]
            known_cells = np.delete(sonar_cells, unknown_ids)
            known_mask = np.zeros(n_cells, dtype=bool)
            unknown_mask = np.zeros(n_cells, dtype=bool)
            known_mask[known_cells] = True
            unknown_mask[unknown_cells] = True
            outputs = np.zeros(n_cells, dtype=np.float32)
            outputs[row_start] = score
            destination["known_sonar_mask"][output_index] = known_mask
            destination["unknown_sonar_mask"][output_index] = unknown_mask
            destination["outputs"][output_index] = outputs
            destination["reachable_mask"][output_index] = np.isfinite(outputs)
            destination["start_as_unknown_number"][output_index] = target_unknown_count
            destination["start_as_known_number"][output_index] = len(known_cells)
            destination["best_sonar_split"][output_index] = mode == "best"
            output_index += 1

        if (row_index + 1) % 100 == 0 or row_index + 1 == total:
            print(f"Converted {row_index + 1}/{total} input rows ({output_index}/{output_total} outputs)", flush=True)

    for array in destination.values():
        array.flush()
    with (output_path / "meta.json").open("w") as handle:
        json.dump(output_meta, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input _parts directory")
    parser.add_argument("--output", required=True, help="Output _parts directory")
    parser.add_argument("--unknown-count", required=True, type=int, help="Target total number of unknown sonars")
    parser.add_argument("--mode", choices=("random", "best", "all"), default="random")
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=None, help="Use this start for every row; otherwise infer it from outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert_parts(
        args.input, args.output, args.unknown_count, args.mode,
        args.height, args.width, args.start_index, args.seed, args.overwrite,
    )


if __name__ == "__main__":
    main()
