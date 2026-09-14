"""
This module defines the IncrementalPlanner class, which is responsible for planning paths in a dynamic environment using an incremental approach. The planner uses a value map to compute costs and employs the A* algorithm to find optimal paths based on the current world state and newly discovered sonar information.
The IncrementalPlanner class is initialized with a value map and provides methods to plan paths and update the world state with newly discovered sonar readings, allowing for efficient replanning in response to changes in the environment.
"""

import numpy as np
import numpy.typing as npt

from environment.cost import CostMap
from environment.path import dijkstra

class IncrementalPlanner:
    def __init__(self, cost_map: CostMap):
        self.cost_map = cost_map
        self.cached_path = None
        self.last_known_sonars = None

    def plan(
        self,
        map_state,
    ):
        known_sonars = map_state.sonar.known
        if self.last_known_sonars is None or not np.array_equal(known_sonars, self.last_known_sonars):
            if self.last_known_sonars is not None:
                self.cached_path = None
        self.last_known_sonars = known_sonars.copy()

        cost_field = self.cost_map.get_cost_map(known_sonars)

        path, perceived_cost, nodes = dijkstra(
            cost_field=cost_field,
            neighbor_map=map_state.neighbor_map,
            positions=map_state.cart_coords,
            obstacles=map_state.obstacles,
            goal_index=map_state.goal_index,
            start_index=map_state.agent_index,
            return_path=True,
        )

        self.cached_path = path
        return path, perceived_cost, nodes