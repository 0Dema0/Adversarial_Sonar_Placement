import argparse

from dataset.generator import generate_dataset, sampler
from dataset.static_cache import build_static


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--height",
        type=int,
        default=20,
        help="Height of the hexagonal grid",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=20,
        help="Width of the hexagonal grid",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100000,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--obstacle-number",
        type=int,
        default=60,
        help="Number of obstacle cells in the hexagonal grid",
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
        help="Indices of the obstacle cells in the hexagonal grid",
    )
    parser.add_argument(
        "--sonar-number",
        type=int,
        default=8,
        help="Number of sonars that are known to the agent",
    )
    parser.add_argument(
        "--unknown-sonar-number",
        type=int,
        default=2,
        help="Number of sonars that are unknown to the agent",
    )
    parser.add_argument(
        "--goal-index",
        type=int,
        default=399,
        help="Index of the goal cell in the hexagonal grid",
    )

    parser.add_argument(
        "--single-value",
        action="store_true",
        default=True,
        help="Generate features for a single start index instead of all valid start indices",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index of the start cell in the hexagonal grid",
    )
    parser.add_argument(
        "--forbidden-radius",
        type=int,
        default=0,
        help="Radius around start and goal indices where sonar positioning is forbidden"
    )
    parser.add_argument(
        "--sonar-placement-policy",
        type=str,
        default="random",
        choices=["random", "endpoint", "mixture"],
        help='Policy for proposing sonar locations. "random" favors random placements; "endpoint" favors placements near start or goal; "mixture" blends both.',
    )
    parser.add_argument(
        "--starting_sonar_placement",
        type=int,
        nargs="+",
        default=None,
        help="This is a HACK: it allows to specify a starting sonar placement for the dataset generation. The placement has to be complete (all sonars must be specified). The sonars will be placed by modifying the given placement (for example, if the number of sonars is 8, 2 of them could be moved).",
    )
    parser.add_argument(
        "--best-sonar-split",
        action="store_true",
        default=False,
        help="Use the best sonar split instead of a random split",
    )
    parser.add_argument(
        "--out",
        default="data/training_dataset__fixed_obstacles__fixed_goal__random_unknown.npz",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of worker processes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of samples per worker task"
    )
    parser.add_argument(
        "--slow-threshold",
        type=float,
        default=None,
        help="Log samples slower than this (seconds)"
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        default=False,
        help="Write samples incrementally to disk to avoid memory growth"
    )

    return parser

def main():
    args = build_parser().parse_args()

    static = build_static(
        height=args.height,
        width=args.width,
        obstacle_indices=args.obstacle_indices,
    )

    generate_dataset(
        static=static,
        sampler=lambda: sampler(args),
        total=args.samples,
        out=args.out,
        workers=args.workers,
        batch_size=args.batch_size,
        slow_threshold=args.slow_threshold,
        stream_output=args.stream_output,
    )


if __name__ == "__main__":
    main()