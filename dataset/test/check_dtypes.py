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

root = Path("data/best_all_dataset_1unknown_complete_parts")

for key in LOCAL_FEATURES:
    arr = np.load(root / f"{key}.npy", mmap_mode="r")
    print(f"{key:25s} shape={arr.shape} dtype={arr.dtype}")