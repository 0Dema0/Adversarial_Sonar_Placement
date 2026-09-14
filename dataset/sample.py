import time
import datetime
import os
import numpy as np

from environment.cost import CostMap
from environment.features import StaticFeatures, Features

def place_sonars(
        static: StaticFeatures,
        sonar_number: int,
        start_index: int,
        goal_index: int,
        forbidden_radius: int,
        sonar_placement_policy: str = "random",
        starting_sonar_placement: list[int] | None = None,
        change_count: int = 3,
        first_time: bool = False,
    ) -> list[int]:
    """
    Place sonars on the hexagonal grid, avoiding a forbidden radius around the start and goal indices.
    """
    non_obstacle_indices = np.setdiff1d(
        static.obstacle_map.raw_map.index_coordinates,
        static.obstacle_map.obstacle_indices,
        assume_unique=True
    )
    available_indices = np.setdiff1d(
        non_obstacle_indices,
        np.concatenate([
            static.obstacle_map.raw_map.get_indices_within_radius(start_index, forbidden_radius),
            static.obstacle_map.raw_map.get_indices_within_radius(goal_index, forbidden_radius)
        ]),
        assume_unique=True
    )
    if starting_sonar_placement is None:
        if sonar_placement_policy == "random":
            chosen = np.random.choice(available_indices, size=sonar_number, replace=False)
        elif sonar_placement_policy == "endpoint":
            ds = static.obstacle_map.raw_map.hex_distance[start_index, available_indices].astype(np.float32)
            dg = static.obstacle_map.raw_map.hex_distance[goal_index, available_indices].astype(np.float32)
            d_end = np.minimum(ds, dg)
            max_d_end = float(d_end.max())
            endpoint_weights = np.ones_like(d_end, dtype=np.float32)
            if max_d_end > 0:
                endpoint_weights = 1.0 + 0.20 * (max_d_end - d_end) / max_d_end
            probs = endpoint_weights / endpoint_weights.sum()
            chosen = np.random.choice(available_indices, size=sonar_number, replace=False, p=probs)
        elif sonar_placement_policy == "mixture":
            ds = static.obstacle_map.raw_map.hex_distance[start_index, available_indices].astype(np.float32)
            dg = static.obstacle_map.raw_map.hex_distance[goal_index, available_indices].astype(np.float32)
            d_end = np.minimum(ds, dg)
            max_d_end = float(d_end.max())
            endpoint_weights = np.ones_like(d_end, dtype=np.float32)
            if max_d_end > 0:
                endpoint_weights = 1.0 + 0.20 * (max_d_end - d_end) / max_d_end
            random_weights = np.ones_like(d_end, dtype=np.float32)
            combined_weights = 0.5 * random_weights + 0.5 * endpoint_weights
            probs = combined_weights / combined_weights.sum()
            chosen = np.random.choice(available_indices, size=sonar_number, replace=False, p=probs)
        else:
            raise ValueError(f"Unknown sonar placement policy: {sonar_placement_policy}")
        return chosen.tolist()
    else:
        if len(starting_sonar_placement) != sonar_number:
            raise ValueError(f"Starting sonar placement must have exactly {sonar_number} indices.")
        if sonar_placement_policy != "random":
            raise ValueError("Only 'random' sonar_placement_policy is supported when starting_sonar_placement is provided.")
        if first_time:
            # First sample uses the provided starting placement exactly.
            return list(starting_sonar_placement)
        starting_sonar_random_changes = change_count

        starting_sonar_indices = np.array(starting_sonar_placement, dtype=np.int64, copy=True)

        keep_count = sonar_number - starting_sonar_random_changes
        fixed_positions = np.random.choice(sonar_number, size=keep_count, replace=False) if keep_count > 0 else np.array([], dtype=np.int64)
        fixed_sonars = starting_sonar_indices[fixed_positions] if keep_count > 0 else np.array([], dtype=np.int64)

        candidate_indices = np.setdiff1d(available_indices, fixed_sonars, assume_unique=False)
        if len(candidate_indices) < starting_sonar_random_changes:
            raise ValueError("Not enough free available indices to randomize the requested number of sonars.")

        replace_positions = np.setdiff1d(np.arange(sonar_number), fixed_positions, assume_unique=True)
        starting_sonar_indices[replace_positions] = np.random.choice(candidate_indices, size=starting_sonar_random_changes, replace=False)
        return starting_sonar_indices.tolist()

def evaluate_configuration(
    static: StaticFeatures,
    sonar_indices: list[int] | np.ndarray,
    sonar_number: int,
    unknown_sonar_number: int | None = None,
    goal_index: int = 399,
    single_value: bool = False,
    start_index: int | None = None,
    best_sonar_split: bool = False,
):
    slow_thresh = globals().get("_SLOW_SAMPLE_THRESHOLD", None)

    t0 = time.time()

    cost_map = CostMap(
        static.obstacle_map,
        static.detection_probability,
        sonar_number=sonar_number,
        sonar_indices=sonar_indices,
        base_contributions=static.base_contributions,
    )
    t1 = time.time()

    t_feat_start = time.time()

    features = Features(
        static=static,
        cost_map=cost_map,
        goal_index=goal_index,
        start_as_unknown_number=unknown_sonar_number,
        single_value=single_value,
        start_index=start_index,
        best_sonar_split=best_sonar_split,
    )

    t2 = time.time()

    result = features.to_dict()

    t3 = time.time()

    total = t3 - t0

    if slow_thresh is not None and total >= float(slow_thresh):
        print(
            f"{datetime.datetime.now().isoformat()} worker(pid={os.getpid()}): "
            f"SLOW_EVAL total={total:.3f}s "
            f"cost_map={t1-t0:.3f}s "
            f"features_init={t2-t_feat_start:.3f}s "
            f"to_dict={t3-t2:.3f}s",
            flush=True,
        )

    return result

def sample_one(
    static: StaticFeatures,
    sonar_number: int  | None = None,
    sonar_indices: list[int] | None = None,
    unknown_sonar_number: int | None = None,
    goal_index: int = 399,
    single_value: bool = False,
    start_index: int | None = None,
    forbidden_radius: int = 0,
    best_sonar_split: bool = False,
    sonar_placement_policy: str = "random",
    starting_sonar_placement: list[int] | None = None,
    first_time: bool = False,
):

    if start_index is not None and sonar_indices is None:
        sonar_indices = place_sonars(
            static=static,
            sonar_number=sonar_number,
            start_index=start_index,
            goal_index=goal_index,
            forbidden_radius=forbidden_radius,
            sonar_placement_policy=sonar_placement_policy,
            starting_sonar_placement=starting_sonar_placement,
            first_time=first_time,
        )

    result = evaluate_configuration(
        static=static,
        sonar_indices=sonar_indices,
        sonar_number=sonar_number,
        unknown_sonar_number=unknown_sonar_number,
        goal_index=goal_index,
        single_value=single_value,
        start_index=start_index,
        best_sonar_split=best_sonar_split,
    )

    return result