"""
This module defines the CostMap class, which represents a cost map for a hexagonal grid. The cost map is constructed based on the visibility of sonar sensors and their detection probabilities. It provides methods to place sonars, build a visibility matrix, and compute the sonar cost contributions to the overall cost map.
The CostMap class is initialized with an ObstacleMap, a detection probability function, and optional parameters for the number of sonars and their indices. It calculates the visibility of each sonar to every cell in the grid and computes the cost contributions based on the detection probabilities.
"""

import numpy as np
import numpy.typing as npt
from collections.abc import Callable

from environment.map import ObstacleMap

DetectionFn = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

class CostMap:
    """
    Represents a cost map for a hexagonal grid.
    """
    def __init__(
            self,
            obstacle_map: ObstacleMap,
            detection_probability: DetectionFn,
            sonar_number: int = 0,
            sonar_indices: list[int] | None = None,
            base_contributions: npt.NDArray[np.float64] | None = None,
        ):
        self.obstacle_map = obstacle_map

        self.sonar_number = sonar_number
        self.sonar_indices = sonar_indices or []
        self.sonar_cartesian_coordinates: npt.NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)
        self.place_sonar()

        self.detection_probability = detection_probability

        if base_contributions is not None:
            self.sonar_cost_contribution = base_contributions[self.sonar_indices]
        else:
            self.sonar_cost_contribution: npt.NDArray[np.float64] = np.zeros((self.sonar_number, self.obstacle_map.raw_map.total_cells), dtype=np.float64)
            self.compute_sonar_cost_contribution()

        self.static_cost_map: npt.NDArray[np.float64] = self.get_cost_map(np.arange(self.sonar_number))

    def place_sonar(self) -> None:
        """
        Place sonars on the hexagonal grid.

        If `sonar_indices` is provided, use those indices; otherwise, place sonars randomly.
        """
        if not self.sonar_indices:
            available_indices = np.setdiff1d(
                self.obstacle_map.raw_map.index_coordinates,
                self.obstacle_map.obstacle_indices,
                assume_unique=True
            )
            self.sonar_indices = np.random.choice(
                available_indices,
                size=self.sonar_number,
                replace=False
            ).tolist()
        self.sonar_cartesian_coordinates = self.obstacle_map.raw_map.cartesian_coordinates[self.sonar_indices]

    def build_visibility_matrix(self) -> npt.NDArray[np.bool_]:
        visibility_matrix: npt.NDArray[np.bool_] = np.zeros((self.sonar_number, self.obstacle_map.raw_map.total_cells), dtype=bool)
        for si, source in enumerate(self.sonar_indices):
            for target in range(self.obstacle_map.raw_map.total_cells):
                visibility_matrix[si, target] = self.obstacle_map.visible(source, target)
        return visibility_matrix

    def compute_sonar_cost_contribution(self) -> None:
        distances = self.obstacle_map.raw_map.euclidean_distance[self.sonar_indices]

        prob = self.detection_probability(distances)

        eps = 1e-12
        log_probability = -np.log1p(-np.clip(prob, 0.0, 1.0 - eps))

        visibility_matrix = self.build_visibility_matrix()
        self.sonar_cost_contribution = log_probability * visibility_matrix

    def get_cost_map(self, known_sonars: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        cost = self.sonar_cost_contribution[known_sonars].sum(axis=0)
        cost[self.obstacle_map.obstacle_indices] = np.inf
        return cost