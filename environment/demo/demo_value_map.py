import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from environment.cost import CostMap
from environment.map import RawMap, ObstacleMap
from environment.perception import SonarKnowledge, random_sonar_split, update_sonar_visibility
from environment.planner import IncrementalPlanner
from environment.plotting import draw_hex_map, draw_value_map
from environment.state import MapState
from environment.utils import detection_probability

raw_map = RawMap(width=20, height=20)

obstacle_map = ObstacleMap(
    raw_map,
    obstacle_number=40,
)

cost_map = CostMap(
    obstacle_map,
    detection_probability,
    sonar_number=8,
)

start_as_known, start_as_unknown = random_sonar_split(8, 2) # (total_sonars, unknown_sonars)

planner = IncrementalPlanner(cost_map)
real_trajectory_cost = np.zeros((raw_map.total_cells,), dtype=np.float64)

valid_nodes = [
    i for i in raw_map.index_coordinates
    if i not in obstacle_map.obstacle_indices
]

for i in valid_nodes:
    sonar = SonarKnowledge(
            np.copy(start_as_known),
            np.copy(start_as_unknown),
        )

    state = MapState(
        obstacle_map=obstacle_map,
        sonar_knowledge=sonar,
        agent_index=i,
        goal_index=raw_map.total_cells - 1,
    )

    step = 0
    trajectory = [state.agent_index]
    real_trajectory_cost[i] = 0.0

    while state.agent_index != state.goal_index:
        newly_discovered = update_sonar_visibility(
            raw_map=raw_map,
            agent_index=state.agent_index,
            sonar_knowledge=state.sonar,
            discover_distance=3.0,
            sonar_indices=raw_map.index_coordinates[cost_map.sonar_indices],
        )

        path, _ , nodes = planner.plan(state)

        if not path or len(path) < 2:
            break
        
        next_node = path[1]
        real_trajectory_cost[i] += cost_map.static_cost_map[next_node]

        state.agent_index = next_node
        trajectory.append(next_node)

        step += 1

fig, ax = plt.subplots(figsize=(8, 8))

draw_hex_map(ax, raw_map)
value_map = draw_value_map(ax, obstacle_map, real_trajectory_cost)

plt.colorbar(value_map, ax=ax, label="Cell cost")

plt.show()