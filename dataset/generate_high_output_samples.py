# generate_high_value_dataset.py - Part 1

import heapq
import time

import numpy as np
from pathlib import Path

from dataset.sample import evaluate_configuration
from dataset.utils import save_dataset, stack_samples


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def canonical_key(sonar_indices):
    """
    Canonical representation of a sonar placement.

    Sonar ordering is irrelevant, so two placements containing the
    same cells are considered identical.
    """
    return tuple(sorted(int(i) for i in sonar_indices))


def score_sample(sample, start_index):
    """
    Extract the scalar objective value.

    Since single_value=True, only start_index contains a non-zero
    value in outputs.
    """
    return float(sample["outputs"][start_index])


def available_positions(
    static,
    start_index,
    goal_index,
    forbidden_radius,
):
    """
    Compute legal sonar positions.
    """

    non_obstacle_indices = np.setdiff1d(
        static.obstacle_map.raw_map.index_coordinates,
        static.obstacle_map.obstacle_indices,
        assume_unique=True,
    )

    forbidden_indices = np.concatenate(
        [
            static.obstacle_map.raw_map.get_indices_within_radius(
                start_index,
                forbidden_radius,
            ),
            static.obstacle_map.raw_map.get_indices_within_radius(
                goal_index,
                forbidden_radius,
            ),
        ]
    )

    return np.setdiff1d(
        non_obstacle_indices,
        forbidden_indices,
        assume_unique=True,
    )


# ------------------------------------------------------------
# Candidate generation
# ------------------------------------------------------------

def exhaustive_one_change(
    current,
    available_indices,
):
    """
    Generate all configurations differing by exactly one sonar.
    """

    current = np.asarray(
        current,
        dtype=np.int32,
    )

    occupied = set(current.tolist())

    for sonar_position in range(len(current)):

        old_cell = current[sonar_position]

        for new_cell in available_indices:

            if new_cell == old_cell:
                continue

            if new_cell in occupied:
                continue

            candidate = current.copy()
            candidate[sonar_position] = new_cell

            yield candidate.tolist()


def random_k_change(
    current,
    available_indices,
    k,
):
    """
    Generate a random configuration differing by k sonar positions.
    """

    current = np.asarray(
        current,
        dtype=np.int32,
    )

    if k >= len(current):
        return None

    candidate = current.copy()

    changed_positions = np.random.choice(
        len(candidate),
        size=k,
        replace=False,
    )

    unchanged = np.delete(
        candidate,
        changed_positions,
    )

    free_locations = np.setdiff1d(
        available_indices,
        unchanged,
        assume_unique=False,
    )

    if len(free_locations) < k:
        return None

    candidate[changed_positions] = np.random.choice(
        free_locations,
        size=k,
        replace=False,
    )

    return candidate.tolist()


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_candidate(
    static,
    sonar_indices,
    sonar_number,
    unknown_sonar_number,
    goal_index,
    start_index,
    single_value,
    best_sonar_split,
):
    """
    Evaluate one explicit sonar placement.
    """

    sample = evaluate_configuration(
        static=static,
        sonar_indices=sonar_indices,
        sonar_number=sonar_number,
        unknown_sonar_number=unknown_sonar_number,
        goal_index=goal_index,
        single_value=single_value,
        start_index=start_index,
        best_sonar_split=best_sonar_split,
    )

    score = score_sample(
        sample,
        start_index,
    )

    return sample, score


# ------------------------------------------------------------
# Neighbour exploration
# ------------------------------------------------------------

def search_one_change(
    current,
    static,
    available_indices,
    visited,
    threshold,
    sonar_number,
    unknown_sonar_number,
    goal_index,
    start_index,
    single_value,
    best_sonar_split,
):
    """
    Exhaustive search of the 1-sonar neighbourhood.
    """

    for candidate in exhaustive_one_change(
        current,
        available_indices,
    ):

        key = canonical_key(candidate)

        if key in visited:
            continue

        sample, score = evaluate_candidate(
            static=static,
            sonar_indices=candidate,
            sonar_number=sonar_number,
            unknown_sonar_number=unknown_sonar_number,
            goal_index=goal_index,
            start_index=start_index,
            single_value=single_value,
            best_sonar_split=best_sonar_split,
        )

        visited.add(key)

        if score > threshold:
            return candidate, sample, score

    return None


def search_random_k_change(
    current,
    static,
    available_indices,
    visited,
    threshold,
    sonar_number,
    unknown_sonar_number,
    goal_index,
    start_index,
    single_value,
    best_sonar_split,
    k,
    timeout,
):
    """
    Random exploration for k-sonar changes.

    Used for k>=2 because exhaustive enumeration grows too quickly.
    """

    start_time = time.time()

    while time.time() - start_time < timeout:

        candidate = random_k_change(
            current=current,
            available_indices=available_indices,
            k=k,
        )

        if candidate is None:
            return None

        key = canonical_key(candidate)

        if key in visited:
            continue

        sample, score = evaluate_candidate(
            static=static,
            sonar_indices=candidate,
            sonar_number=sonar_number,
            unknown_sonar_number=unknown_sonar_number,
            goal_index=goal_index,
            start_index=start_index,
            single_value=single_value,
            best_sonar_split=best_sonar_split,
        )

        visited.add(key)

        if score > threshold:
            return candidate, sample, score

    return None


# ------------------------------------------------------------
# Priority queue helpers
# ------------------------------------------------------------

def push_solution(
    heap,
    counter,
    sonar_indices,
    sample,
    score,
):
    """
    Add a successful configuration to the max-priority queue.
    """

    heapq.heappush(
        heap,
        (
            -float(score),  # max heap behaviour
            counter,
            sonar_indices,
            sample,
        ),
    )


def pop_solution(heap):
    """
    Retrieve the highest scoring configuration.
    """

    _, _, sonar_indices, sample = heapq.heappop(heap)

    return sonar_indices, sample


def initialize_memmaps(
    first_sample,
    total,
    stream_dir,
):
    """
    Create .npy memmaps for streaming output.
    """

    stream_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    memmaps = {}
    meta = {}

    for key, value in first_sample.items():

        arr = np.asarray(value)

        shape = (
            total,
        ) + arr.shape

        mmap = np.lib.format.open_memmap(
            stream_dir / f"{key}.npy",
            mode="w+",
            dtype=arr.dtype,
            shape=shape,
        )

        memmaps[key] = mmap

        meta[key] = {
            "shape": list(shape),
            "dtype": str(arr.dtype),
        }


    with open(stream_dir / "meta.json", "w") as f:

        import json

        json.dump(
            {
                "total": int(total),
                "processed": 0,
                "keys": meta,
                "created": time.time(),
            },
            f,
        )


    return memmaps

# ------------------------------------------------------------
# Main high-value search
# ------------------------------------------------------------

def generate_high_value_dataset(
    static,
    initial_sonar_placement,
    target_samples,
    out,
    thresholds,
    sonar_number,
    unknown_sonar_number,
    goal_index,
    start_index,
    forbidden_radius,
    single_value=True,
    best_sonar_split=False,
    max_changes=None,
    random_trials_timeout=5.0,
    checkpoint_interval=50,
    global_timeout=None,
    stream_output=False,
):
    """
    Generate a dataset containing configurations whose output exceeds
    the requested threshold.

    Search strategy:
        - maintain a max-priority queue of successful configurations
        - expand best configurations first
        - try 1-sonar changes exhaustively
        - try k-sonar changes randomly for k>=2

    Parameters
    ----------
    initial_sonar_placement:
        A complete sonar configuration used as the seed.

    target_samples:
        Number of high-value samples to collect.

    threshold:
        Required output value.
    """

    start_time = time.time()

    if max_changes is None:
        max_changes = sonar_number

    if len(thresholds) != len(target_samples):
        raise ValueError(
            "thresholds and samples_per_threshold must have "
            "the same length"
        )

    # Highest threshold first
    threshold_targets = sorted(
        zip(thresholds, target_samples),
        reverse=True,
    )

    total_target = sum(
        target
        for _, target in threshold_targets
    )

    threshold_counts = {
        threshold: 0
        for threshold, _ in threshold_targets
    }

    available_indices = available_positions(
        static=static,
        start_index=start_index,
        goal_index=goal_index,
        forbidden_radius=forbidden_radius,
    )


    # --------------------------------------------------------
    # Evaluate seed
    # --------------------------------------------------------

    print(
        "Evaluating initial sonar configuration..."
    )

    seed_sample, seed_score = evaluate_candidate(
        static=static,
        sonar_indices=initial_sonar_placement,
        sonar_number=sonar_number,
        unknown_sonar_number=unknown_sonar_number,
        goal_index=goal_index,
        start_index=start_index,
        single_value=single_value,
        best_sonar_split=best_sonar_split,
    )

    seed_key = canonical_key(
        initial_sonar_placement
    )

    visited = {
        seed_key
    }


    if seed_score <= max(thresholds):
        raise RuntimeError(
            f"Initial sonar placement has score "
            f"{seed_score:.3f}, which is not above threshold {max(thresholds)}."
        )


    results = []

    processed = 0

    memmaps = None


    # --------------------------------------------------------
    # Streaming setup
    # --------------------------------------------------------

    if stream_output:

        stream_dir = (
            Path(out).parent /
            (Path(out).stem + "_parts")
        )

    else:

        stream_dir = None


    # --------------------------------------------------------
    # Store seed
    # --------------------------------------------------------

    if stream_output:

        memmaps = initialize_memmaps(
            first_sample=seed_sample,
            total=total_target,
            stream_dir=stream_dir,
        )

        for key, mmap in memmaps.items():
            mmap[0] = seed_sample[key]

    else:

        results.append(seed_sample)


    processed = 1


    # --------------------------------------------------------
    # Priority queue
    # --------------------------------------------------------

    queue = []

    counter = 0

    push_solution(
        heap=queue,
        counter=counter,
        sonar_indices=initial_sonar_placement,
        sample=seed_sample,
        score=seed_score,
    )

    counter += 1


    print(
        f"Seed accepted. score={seed_score:.3f}"
    )


    # --------------------------------------------------------
    # Search loop
    # --------------------------------------------------------

    for threshold, target_samples in threshold_targets:

        threshold_processed = 0 if threshold != max(thresholds) else 1

        print()
        print("=" * 60)
        print(
            f"Starting threshold {threshold}"
        )
        print(
            f"Target: {target_samples}"
        )
        print("=" * 60)

        while (
            threshold_processed < target_samples
            and queue
        ):

            if (
                global_timeout is not None
                and time.time() - start_time > global_timeout
            ):
                print(
                    "Global timeout reached."
                )
                break


            current, _ = pop_solution(
                queue
            )


            found = None


            # --------------------------------------------
            # Try increasing neighbourhood size
            # --------------------------------------------

            for changes in range(
                1,
                max_changes + 1,
            ):

                print(
                    f"Searching {changes}-sonar neighbours "
                    f"(threshold={threshold}, "
                    f"dataset={processed}/{total_target}, "
                    f"threshold_samples="
                    f"{threshold_processed}/{target_samples})"
                )


                if changes == 1:

                    found = search_one_change(
                        current=current,
                        static=static,
                        available_indices=available_indices,
                        visited=visited,
                        threshold=threshold,
                        sonar_number=sonar_number,
                        unknown_sonar_number=unknown_sonar_number,
                        goal_index=goal_index,
                        start_index=start_index,
                        single_value=single_value,
                        best_sonar_split=best_sonar_split,
                    )

                else:

                    found = search_random_k_change(
                        current=current,
                        static=static,
                        available_indices=available_indices,
                        visited=visited,
                        threshold=threshold,
                        sonar_number=sonar_number,
                        unknown_sonar_number=unknown_sonar_number,
                        goal_index=goal_index,
                        start_index=start_index,
                        single_value=single_value,
                        best_sonar_split=best_sonar_split,
                        k=changes,
                        timeout=random_trials_timeout,
                    )


                if found is not None:
                    break


            # --------------------------------------------
            # No neighbour found
            # --------------------------------------------

            if found is None:

                print(
                    f"No new sample found for threshold {threshold}."
                )

                continue


            # --------------------------------------------
            # Store new solution
            # --------------------------------------------

            sonar_indices, sample, score = found


            if stream_output:

                if memmaps is None:

                    memmaps = initialize_memmaps(
                        first_sample=sample,
                        total=total_target,
                        stream_dir=stream_dir,
                    )


                for key, mmap in memmaps.items():

                    mmap[processed] = sample[key]

            else:

                results.append(sample)

            processed += 1
            threshold_processed += 1

            if stream_output and processed % 100 == 0:

                import json

                meta_path = stream_dir / "meta.json"

                with open(meta_path, "r") as f:
                    meta = json.load(f)


                meta["processed"] = processed


                with open(meta_path, "w") as f:
                    json.dump(meta, f)


            push_solution(
                heap=queue,
                counter=counter,
                sonar_indices=sonar_indices,
                sample=sample,
                score=score,
            )

            counter += 1


            print(
                f"Found sample "
                f"{processed}/{target_samples} "
                f"score={score:.3f}"
            )

    if stream_output:

        if memmaps is not None:

            for mmap in memmaps.values():
                mmap.flush()


        meta_path = stream_dir / "meta.json"

        with open(meta_path,"r") as f:
            meta=json.load(f)

        meta["processed"]=processed

        with open(meta_path,"w") as f:
            json.dump(meta,f)


        print(
            f"Streaming dataset written to {stream_dir}"
        )

        return None


    else:

        dataset = stack_samples(results)

        save_dataset(
            out,
            dataset,
        )

        return dataset

# ------------------------------------------------------------
# Seed search
# ------------------------------------------------------------

from dataset.sample import place_sonars
from dataset.static_cache import build_static
from dataset.utils import stack_samples, save_dataset

import argparse


def find_seed_configuration(
    static,
    sonar_number,
    unknown_sonar_number,
    goal_index,
    start_index,
    forbidden_radius,
    threshold,
    sonar_placement_policy,
    best_sonar_split,
    single_value,
    max_attempts=10000,
):
    """
    Randomly sample sonar configurations until a configuration
    exceeding the target threshold is found.
    """

    print(
        f"Searching seed configuration with score > {threshold}"
    )

    best_score = -np.inf
    best_configuration = None


    for attempt in range(1, max_attempts + 1):

        sonar_indices = place_sonars(
            static=static,
            sonar_number=sonar_number,
            start_index=start_index,
            goal_index=goal_index,
            forbidden_radius=forbidden_radius,
            sonar_placement_policy=sonar_placement_policy,
        )


        sample, score = evaluate_candidate(
            static=static,
            sonar_indices=sonar_indices,
            sonar_number=sonar_number,
            unknown_sonar_number=unknown_sonar_number,
            goal_index=goal_index,
            start_index=start_index,
            single_value=single_value,
            best_sonar_split=best_sonar_split,
        )


        if score > best_score:
            best_score = score
            best_configuration = sonar_indices

            print(
                f"New best seed: "
                f"{best_score:.3f} "
                f"(attempt {attempt})"
            )


        if score > threshold:
            print(
                f"Found seed after {attempt} attempts "
                f"(score={score:.3f})"
            )

            return sonar_indices


    raise RuntimeError(
        f"Could not find a seed above threshold after "
        f"{max_attempts} attempts. "
        f"Best score={best_score:.3f}"
    )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def build_parser():

    parser = argparse.ArgumentParser(
        description="Generate high-value sonar placement dataset"
    )


    parser.add_argument(
        "--height",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--obstacle-indices",
        nargs="+",
        type=int,
        default=[
            43, 44, 24, 183, 184, 203, 204, 205,
            224, 225, 244, 245, 262, 263, 264, 265,
            285, 286, 287, 267, 132, 133, 152, 190,
            191, 211, 212, 213, 355, 376, 113, 112,
            94, 92, 91, 63, 64, 19, 18, 17, 16, 39,
            38, 37, 59, 231, 232, 233, 234, 251, 252,
            253, 271, 290, 291, 292, 309, 310, 329, 351
        ],
    )


    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[
            13.0,
            12.0,
            11.0,
            10.0,
            9.0,
            8.0,
            7.0,
            6.0,
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
            0.0,
        ],
    )


    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=[
            5,
            50,
            500,
            500,
            500,
            500,
            500,
            500,
            500,
            500,
            500,
            500,
            500,
            500,
        ],
    )


    parser.add_argument(
        "--sonar-number",
        type=int,
        default=8,
    )


    parser.add_argument(
        "--unknown-sonar-number",
        type=int,
        default=0,
    )


    parser.add_argument(
        "--goal-index",
        type=int,
        default=399,
    )


    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )


    parser.add_argument(
        "--forbidden-radius",
        type=int,
        default=6,
    )


    parser.add_argument(
        "--single-value",
        action="store_true",
        default=True,
    )


    parser.add_argument(
        "--best-sonar-split",
        action="store_true",
        default=False,
    )


    parser.add_argument(
        "--sonar-placement-policy",
        choices=[
            "random",
            "endpoint",
            "mixture",
        ],
        default="random",
    )


    parser.add_argument(
        "--starting-sonar-placement",
        nargs="+",
        type=int,
        default=[235, 236, 238, 259, 314, 349, 373, 391],
    )


    parser.add_argument(
        "--out",
        default="data/evaluating_best_dataset.npz",
    )


    parser.add_argument(
        "--max-changes",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--seed-attempts",
        type=int,
        default=10000,
    )


    parser.add_argument(
        "--global-timeout",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--stream-output",
        action="store_true",
        default=True,
        help="Write samples incrementally to disk to avoid memory growth"
    )


    return parser


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    args = build_parser().parse_args()


    static = build_static(
        height=args.height,
        width=args.width,
        obstacle_indices=args.obstacle_indices,
    )


    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    if args.starting_sonar_placement is not None:

        seed = place_sonars(
            static=static,
            sonar_number=args.sonar_number,
            start_index=args.start_index,
            goal_index=args.goal_index,
            forbidden_radius=args.forbidden_radius,
            sonar_placement_policy="random",
            starting_sonar_placement=args.starting_sonar_placement,
            first_time=True,
        )

    else:

        seed = find_seed_configuration(
            static=static,
            sonar_number=args.sonar_number,
            unknown_sonar_number=args.unknown_sonar_number,
            goal_index=args.goal_index,
            start_index=args.start_index,
            forbidden_radius=args.forbidden_radius,
            threshold=max(args.thresholds),
            sonar_placement_policy=args.sonar_placement_policy,
            best_sonar_split=args.best_sonar_split,
            single_value=args.single_value,
            max_attempts=args.seed_attempts,
        )


    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    generate_high_value_dataset(
        static=static,
        initial_sonar_placement=seed,
        target_samples=args.samples,
        out=args.out,
        thresholds=args.thresholds,
        sonar_number=args.sonar_number,
        unknown_sonar_number=args.unknown_sonar_number,
        goal_index=args.goal_index,
        start_index=args.start_index,
        forbidden_radius=args.forbidden_radius,
        single_value=args.single_value,
        best_sonar_split=args.best_sonar_split,
        max_changes=args.max_changes,
        global_timeout=args.global_timeout,
        stream_output=args.stream_output,
    )


if __name__ == "__main__":
    main()