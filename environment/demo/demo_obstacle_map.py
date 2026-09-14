import matplotlib.pyplot as plt

from environment.map import RawMap, ObstacleMap
from environment.plotting import draw_hex_map, draw_obstacles

raw_map = RawMap(
    width=20,
    height=20,
)

obstacle_map = ObstacleMap(
    raw_map,
    obstacle_number=40,
)

fig, ax = plt.subplots(figsize=(8, 8))

draw_hex_map(ax, raw_map)
draw_obstacles(ax, obstacle_map)

plt.show()