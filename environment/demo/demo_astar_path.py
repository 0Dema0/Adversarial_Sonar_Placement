import matplotlib.pyplot as plt
import numpy as np

from environment.cost import CostMap
from environment.map import RawMap, ObstacleMap
from environment.perception import SonarKnowledge, random_sonar_split, update_sonar_visibility
from environment.planner import IncrementalPlanner
from environment.plotting import draw_hex_map, draw_cost_map
from environment.state import MapState
from environment.utils import detection_probability

raw_map = RawMap(width=20, height=20)

obstacles_indices = None
obstacles_indices = [43, 44, 24, 183, 184, 203, 204, 205, 224, 225, 244, 245, 262, 263, 264, 265, 285, 286, 287, 267, 132, 133, 152, 190, 191, 211, 212, 213, 355, 376, 113, 112, 94, 92, 91, 63, 64, 19, 18, 17, 16, 39, 38, 37, 59, 231, 232, 233, 234, 251, 252, 253, 271, 290, 291, 292, 309, 310, 329, 351]

obstacle_map = ObstacleMap(
    raw_map,
    obstacle_number=40,
    obstacle_indices=obstacles_indices
)

sonar_indices = None
#sonar_indices = [235, 236, 238, 259, 314, 349, 373, 391]

forbidden_radius = 6
no_obstacles_indices = np.setdiff1d(
    raw_map.index_coordinates,
    obstacle_map.obstacle_indices,
    assume_unique=True
)
available_indices = np.setdiff1d(
    no_obstacles_indices,
    np.concatenate([
        raw_map.get_indices_within_radius(0, forbidden_radius),
        raw_map.get_indices_within_radius(raw_map.total_cells - 1, forbidden_radius)
    ]),
    assume_unique=True
)
sonar_indices = np.random.choice(
    available_indices,
    size=8,
    replace=False
).tolist()

cost_map = CostMap(
    obstacle_map,
    detection_probability,
    sonar_number=8,
    sonar_indices=sonar_indices,
)

start_as_known, start_as_unknown = random_sonar_split(8, 2)
sonar = SonarKnowledge(start_as_known, start_as_unknown)

state = MapState(
    obstacle_map=obstacle_map,
    sonar_knowledge=sonar,
    agent_index=0,
    goal_index=raw_map.total_cells - 1,
)

planner = IncrementalPlanner(cost_map)

step = 0
trajectory = [state.agent_index]
perceived_trajectory_cost = 0.0
real_trajectory_cost = 0.0

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
    step_cost = cost_map.get_cost_map(state.sonar.known)[next_node]
    perceived_trajectory_cost += step_cost
    real_trajectory_cost += cost_map.static_cost_map[next_node]

    print(f"Step {step} | perceived_trajectory_cost ={perceived_trajectory_cost} | real_trajectory_cost = {real_trajectory_cost} | discovered = {len(newly_discovered)}")
    print(f"Total known sonars: {len(state.sonar.known)} | Total unknown sonars: {len(state.sonar.unknown)}")

    state.agent_index = next_node
    trajectory.append(next_node)

    step += 1

print("Full trajectory 0 → goal:")
print(trajectory)

fig, ax = plt.subplots(figsize=(8, 8))

draw_hex_map(ax, raw_map)
draw_cost_map(ax, cost_map)

if trajectory:
    pts = raw_map.cartesian_coordinates[trajectory]
    ax.plot(pts[:, 0], pts[:, 1], color="cyan", linewidth=2)

ax.scatter(*raw_map.cartesian_coordinates[trajectory[0]], color="green", s=80, label="start")
ax.scatter(*raw_map.cartesian_coordinates[trajectory[-1]], color="blue", s=80, label="goal")

ax.legend(loc="best")
plt.show()