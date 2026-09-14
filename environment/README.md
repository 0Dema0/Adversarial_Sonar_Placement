# Environment package

This package defines the simulated hex-grid world used by the dataset generator and planning pipeline. It models a 2D hexagonal map with obstacles, sonar coverage, visibility constraints, and path-planning utilities.

The environment is built around the idea that each cell in a hex grid can be indexed, mapped to axial/cartesian coordinates, and evaluated under a cost model that depends on visible sonar contributions and obstacle geometry.

## Package structure

- `map.py`: geometry of the hex grid and obstacle-aware visibility
- `cost.py`: sonar-based cost map and static cost computation
- `perception.py`: sonar knowledge state and visibility updates
- `planner.py`: incremental planner using the current visible cost field
- `path.py`: A* and Dijkstra pathfinding utilities
- `state.py`: dynamic map state representation for the agent
- `utils/`: geometry helpers and detection-probability utilities
- `demo/`: small scripts that visualize the environment and its components

## Core concepts

### RawMap

`RawMap` represents a hexagonal map without obstacles. It stores:

- cell indices
- axial coordinates
- cubic coordinates
- cartesian coordinates
- neighbor relations
- hex-distance matrices
- Euclidean distances
- ring structure for neighborhood/feature extraction

This class is the underlying geometric model for all planning and dataset generation.

### ObstacleMap

`ObstacleMap` wraps a `RawMap` and adds obstacle placement and line-of-sight visibility checks.

The key idea is that movement and sonar visibility are blocked by obstacle polygons. Visibility is computed by testing whether the segment from one cell center to another intersects any obstacle polygon.

### CostMap

`CostMap` builds a static cost field from the visible sonar sensors.

For each sonar location:

1. compute the distance to all cells
2. evaluate the detection probability
3. convert it into a log-cost contribution
4. mask out cells blocked by obstacles
5. sum contributions over the currently known sonar set

This produces a scalar cost field that the planner uses to reason about navigation under partial information.

### SonarKnowledge and perception

`SonarKnowledge` stores which sonar indices are currently:

- known to the agent
- unknown to the agent

The package provides functions for:

- random sonar split
- best sonar split selection
- updating known/unknown sonar status after discovery from the current agent position

This model is used to simulate partial observability during planning and dataset generation.

### Planner and pathfinding

The environment includes:

- `dijkstra(...)` in `path.py`
- `a_star(...)` in `path.py`
- `IncrementalPlanner` in `planner.py`

The planner uses the current visible cost field and the agent's known sonar information to compute a path to the goal while updating costs as new sonars are discovered.

## Typical usage

```python
import numpy as np

from environment.map import RawMap, ObstacleMap
from environment.cost import CostMap
from environment.utils import detection_probability

height = 20
width = 20
obstacle_indices = [10, 20, 30, 40]

raw_map = RawMap(width=width, height=height)
obstacle_map = ObstacleMap(raw_map, obstacle_indices=obstacle_indices)

cost_map = CostMap(
    obstacle_map=obstacle_map,
    detection_probability=detection_probability,
    sonar_number=8,
    sonar_indices=[5, 12, 35, 44, 70, 90, 120, 160],
)

print(cost_map.static_cost_map.shape)
print(cost_map.sonar_indices)
```

## Visibility and geometry

The geometry layer computes:

- grid layout from width and height
- neighbor map for each cell
- index-to-coordinate conversion
- euclidean and hex distances
- ring neighborhoods used in feature extraction

This is essential both for simulation correctness and for the feature generation used in the dataset package.

## Utility modules

The `environment/utils` folder contains helper logic such as:

- axial/cartesian conversion
- hex-neighbor generation
- corner generation for hex cells
- segment/ polygon intersection tests
- detection probability functions

These utilities are used throughout the project and are the low-level building blocks for map visibility and feature construction.

## Demonstrations

The demo scripts in `environment/demo/` visualize different environment concepts, including:

- raw map geometry
- obstacle placement
- sonar map behavior
- cost map structure
- A*/Dijkstra-based navigation

## Design notes

This package is intentionally structured so that the geometry, perception, and planning layers are separated:

- geometry: where cells and obstacles live
- perception: which areas are visible and known
- planning: how the agent chooses a path under uncertainty

That separation makes it easier to swap or test different sonar models, different obstacle layouts, or different path-planning strategies without rewriting the whole simulation.

## Minimal planning example

```python
from environment.map import RawMap, ObstacleMap
from environment.cost import CostMap
from environment.perception import SonarKnowledge
from environment.state import MapState
from environment.planner import IncrementalPlanner
from environment.utils import detection_probability

raw_map = RawMap(width=20, height=20)
obstacle_map = ObstacleMap(raw_map, obstacle_indices=[10, 11, 12, 20])

cost_map = CostMap(
    obstacle_map=obstacle_map,
    detection_probability=detection_probability,
    sonar_number=4,
    sonar_indices=[1, 2, 3, 4],
)

planner = IncrementalPlanner(cost_map)

state = MapState(
    obstacle_map=obstacle_map,
    sonar_knowledge=SonarKnowledge(
        start_as_known=np.array([0, 1], dtype=int),
        start_as_unknown=np.array([2, 3], dtype=int),
    ),
    agent_index=0,
    goal_index=100,
)

path, cost, visited = planner.plan(state)
print(path)
print(cost)
```

## Summary

The environment package acts as the simulation engine for the dataset: it defines the hex grid, obstacle layout, sensing model, and path-planning logic used to generate and evaluate the synthetic training samples.

It is the core geometry and decision layer behind the dataset generation pipeline.
