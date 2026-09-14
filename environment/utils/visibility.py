"""
Visibility utilities: this module provides functions to determine visibility and
intersections between points, segments, and polygons.
"""

import numpy as np
import numpy.typing as npt

def point_in_convex_polygon(point: npt.NDArray[np.float64], polygon: npt.NDArray[np.float64]) -> bool:
	"""
	Return whether `point` is inside convex `polygon` or not.
	"""
	x, y = point[0], point[1]
	pos = None
	n = len(polygon)
	if n == 0:
		return False
	for i in range(n):
		x1, y1 = polygon[i]
		x2, y2 = polygon[(i + 1) % n]
		cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
		if abs(cross) < 1e-12:
			continue
		if pos is None:
			pos = cross > 0
		else:
			if (cross > 0) != pos:
				return False
	return True if pos is not None else False


def segment_intersects_segment(a1: npt.NDArray[np.float64], a2: npt.NDArray[np.float64], b1: npt.NDArray[np.float64], b2: npt.NDArray[np.float64]) -> bool:
	"""
	Return whether segments a1-a2 and b1-b2 intersect or not (including collinear overlap).
	"""
	def orient(p, q, r):
		return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

	def on_seg(p, q, r):
		return (
			min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
			and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
		)

	o1 = orient(a1, a2, b1)
	o2 = orient(a1, a2, b2)
	o3 = orient(b1, b2, a1)
	o4 = orient(b1, b2, a2)

	if o1 * o2 < 0 and o3 * o4 < 0:
		return True
	if abs(o1) < 1e-12 and on_seg(a1, b1, a2):
		return True
	if abs(o2) < 1e-12 and on_seg(a1, b2, a2):
		return True
	if abs(o3) < 1e-12 and on_seg(b1, a1, b2):
		return True
	if abs(o4) < 1e-12 and on_seg(b1, a2, b2):
		return True
	return False


def segment_intersects_polygon(p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64], polygon: npt.NDArray[np.float64]) -> bool:
	"""
	Return whether segment p1-p2 intersects convex `polygon` or not.
	"""
	if point_in_convex_polygon(p1, polygon) or point_in_convex_polygon(p2, polygon):
		return True
	n = len(polygon)
	for i in range(n):
		q1 = polygon[i]
		q2 = polygon[(i + 1) % n]
		if segment_intersects_segment(tuple(p1), tuple(p2), tuple(q1), tuple(q2)):
			return True
	return False