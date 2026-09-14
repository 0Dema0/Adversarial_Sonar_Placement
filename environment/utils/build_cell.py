"""
Hexagonal grid geometry helpers: this module contains functions that compute geometric properties
of hexagonal grids.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

def hex_radius(cartesian_coordinates: npt.NDArray[np.float64]) -> np.float64:
	"""
	Estimate hex radius from cartesian coordinates as min nonzero neighbour distance / sqrt(3).
	The hex radius is the distance from the center of a hex to its corners,
	which is related to the distance between adjacent hex centers.
	"""
	from scipy.spatial import cKDTree

	tree = cKDTree(cartesian_coordinates)
	dists, _ = tree.query(cartesian_coordinates, k=2)
	min_dist = dists[:, 1].min()
	return min_dist / np.sqrt(3.0)


def hex_corners(center: npt.NDArray[np.float64], hex_radius: float) -> npt.NDArray[np.float64]:
	"""
	Return 6 (x,y) corner coordinates for a pointy-top hex at `center` as an ndarray (6,2).
	"""
	cx, cy = center[0], center[1]
	angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6.0
	xs = cx + hex_radius * np.cos(angles)
	ys = cy + hex_radius * np.sin(angles)
	return np.stack((xs, ys), axis=1)