"""
Hexagonal grid conversion helpers: this module contains functions that convert between
different coordinate systems used in hexagonal grids.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

def axial_to_cart(axial_coordinates: npt.NDArray[np.int64], scale: float = 1.0) -> npt.NDArray[np.float64]:
	"""
	Convert axial (q,r) coordinates to 2D cartesian (pointy-top) positions, with formula:
	    px = s * sqrt(3) * (q + r/2)
		py = s * 3/2 * r

	Parameters
	----------
	axial_coordinates : npt.NDArray[np.int64]
		Axial coordinates (q,r) of hexagonal grid cells.
	scale : float, optional
		Scale factor for the hexagon size. Default is 1.0.

	Returns
	-------
	npt.NDArray[np.float64]
		2D cartesian coordinates of the hexagonal grid cells.
	"""
	q = axial_coordinates[:, 0]
	r = axial_coordinates[:, 1]
	px = scale * np.sqrt(3.0) * (q + 0.5 * r)
	py = scale * 1.5 * r
	return np.stack((px, py), axis=1)