import matplotlib.pyplot as plt

from environment.cost import CostMap
from environment.map import RawMap, ObstacleMap
from environment.plotting import draw_hex_map, draw_cost_map
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

fig, ax = plt.subplots(figsize=(8, 8))

draw_hex_map(ax, raw_map)
cost_field = draw_cost_map(ax, cost_map)

plt.colorbar(cost_field, ax=ax, label="Cell cost")

plt.show()