"""
This module defines the MapState class, which represents the state of a hexagonal map in a dynamic environment. The MapState class encapsulates the positions of cells, obstacles, sonar knowledge, and indices for the agent and goal. It provides a structured representation of the environment's state, allowing for efficient planning and decision-making based on the current configuration of the map and sonar information.
The MapState class is initialized with the positions of cells, a set of obstacle indices, sonar knowledge (known and unknown sonar readings), and the indices for the agent and goal. This structured representation enables effective management of the environment's state, facilitating path planning and navigation in the presence of obstacles and sonar readings.
"""

import numpy as np
import numpy.typing as npt

from environment.map import ObstacleMap
from environment.perception import SonarKnowledge, random_sonar_split

class MapState:
    def __init__(
        self,
        obstacle_map: ObstacleMap,
        sonar_knowledge: SonarKnowledge,
        agent_index: int,
        goal_index: int,
    ):
        self.cart_coords: npt.NDArray[np.float64] = obstacle_map.raw_map.cartesian_coordinates
        self.neighbor_map: list[list[int]] = obstacle_map.raw_map.neighbor_map
        self.obstacles: set[int] = set(obstacle_map.obstacle_indices)

        self.sonar = sonar_knowledge

        self.agent_index = agent_index
        self.goal_index = goal_index

    def update_sonar_knowledge(self, newly_discovered_sonars: npt.NDArray[np.int64]) -> None:
        self.sonar.known = np.union1d(
            self.sonar.known,
            newly_discovered_sonars
        )

        self.sonar.unknown = np.setdiff1d(
            self.sonar.unknown,
            newly_discovered_sonars
        )