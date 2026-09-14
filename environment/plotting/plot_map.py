"""
Plotting utilities for hexagonal maps.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import numpy.ma as ma

from matplotlib.collections import PolyCollection

from environment.map import RawMap, ObstacleMap
from environment.cost import CostMap


def draw_hex_map(
    ax: plt.Axes,
    raw_map: RawMap,
    facecolor: str = "white",
    edgecolor: str = "black",
    linewidth: float = 0.3,
):
    """
    Draw a hexagonal map.

    Parameters
    ----------
    ax
        Matplotlib axes.

    raw_map
        Map to draw.

    facecolor
        Color of the hexagon faces.

    edgecolor
        Color of the hexagon edges.

    linewidth
        Width of the hexagon edges.

    Returns
    -------
    PolyCollection
    """

    polygons = raw_map.cell_polygons

    collection = PolyCollection(
        polygons,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
    )

    ax.add_collection(collection)

    vertices = raw_map.cell_polygons.reshape(-1, 2)

    xmin, ymin = vertices.min(axis=0)
    xmax, ymax = vertices.max(axis=0)

    margin = 0.05 * raw_map.cell_radius

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return collection

def draw_obstacles(
    ax: plt.Axes,
    obstacle_map: ObstacleMap,
    facecolor: str = "black",
) -> PolyCollection:
    """
    Draw the obstacle cells.
    """

    collection = PolyCollection(
        obstacle_map.obstacle_polygons,
        facecolors=facecolor,
    )

    ax.add_collection(collection)

    return collection

def draw_sonars(
    ax: plt.Axes,
    cost_map: CostMap,
    color: str = "red",
    size: float = 50,
):
    """
    Draw sonar locations as points.
    """
    pts = cost_map.sonar_cartesian_coordinates

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=color,
        s=size,
        zorder=10,
        marker="o",
    )

def draw_cost_map(
    ax: plt.Axes,
    cost_map: CostMap,
    cmap: str = "hot",
) -> PolyCollection:
    """
    Draw the static cost map.
    """
    cost = np.array(cost_map.static_cost_map, dtype=float)

    masked_cost = ma.masked_where(np.isinf(cost), cost)

    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")

    collection = PolyCollection(
        cost_map.obstacle_map.raw_map.cell_polygons,
        cmap=cmap_obj,
    )

    collection.set_array(masked_cost)

    ax.add_collection(collection)

    return collection

def draw_value_map(
    ax: plt.Axes,
    obstacle_map: ObstacleMap,
    values: npt.NDArray[np.float64],
    cmap: str = "viridis",
) -> PolyCollection:
    """
    Draw the value map.
    """
    values = np.array(values, dtype=float)

    obstacle_mask = np.zeros_like(values, dtype=bool)
    obstacle_mask[list(obstacle_map.obstacle_indices)] = True

    full_mask = obstacle_mask | np.isinf(values)

    masked_values = ma.masked_where(full_mask, values)

    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")

    collection = PolyCollection(
        obstacle_map.raw_map.cell_polygons,
        cmap=cmap_obj,
    )

    collection.set_array(masked_values)

    ax.add_collection(collection)

    return collection