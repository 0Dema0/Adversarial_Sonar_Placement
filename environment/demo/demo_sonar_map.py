import matplotlib.pyplot as plt

from environment.cost import CostMap
from environment.map import RawMap, ObstacleMap
from environment.plotting import draw_hex_map, draw_obstacles, draw_sonars
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
draw_obstacles(ax, obstacle_map)
draw_sonars(ax, cost_map)

plt.show()