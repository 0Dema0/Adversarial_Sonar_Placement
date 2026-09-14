import matplotlib.pyplot as plt

from environment.map import RawMap
from environment.plotting import draw_hex_map

raw_map = RawMap(
    width=20,
    height=20,
)

fig, ax = plt.subplots(figsize=(8, 8))

draw_hex_map(
    ax,
    raw_map,
)

plt.show()