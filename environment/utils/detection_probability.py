"""
Detection probability utilities: this module provides functions to calculate
the probability of detecting an object based on distance.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

def detection_probability(distance: npt.NDArray[np.float64], d0: float = 3.0, beta: float = 1.0) -> npt.NDArray[np.float64]:
	"""
	Logistic detection probability P(d) = 1 / (1 + exp(beta*(d - d0))).

	Returns the probability of detection as a function of distance `d`, with parameters `d0`
	(the distance at which the probability is 0.5) and `beta` (the steepness of the logistic curve).
	"""
	return 1.0 / (1.0 + np.exp(beta * (distance - d0)))