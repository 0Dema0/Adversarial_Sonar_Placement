"""
Hexagonal grid relation helpers: this module contains functions that relate to the structure
and relationships of cells in hexagonal grids.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

def neighbors(cube_coordinates: npt.NDArray[np.int64]) -> list[list[int]]:
	"""
	Compute neighbor adjacency list for cube coordinates.

	Returns a list of neighbors for each index in the same order as `cube_coordinates`.
	"""
	directions = [
		(+1, -1, 0),
		(+1, 0, -1),
		(0, +1, -1),
		(-1, +1, 0),
		(-1, 0, +1),
		(0, -1, +1),
	]
	cmap = {tuple(c): i for i, c in enumerate(cube_coordinates)}
	N = len(cube_coordinates)
	neighbor_map: list[list[int]] = [[] for _ in range(N)]
	for i, (x, y, z) in enumerate(cube_coordinates):
		for dx, dy, dz in directions:
			j = cmap.get((x + dx, y + dy, z + dz))
			if j is not None:
				neighbor_map[i].append(j)
	return neighbor_map