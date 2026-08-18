"""Classification helpers for turning continuous indices into discrete classes."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr


def classify_by_thresholds(index, thresholds: Sequence[float], class_values: Sequence[int] | None = None):
	"""Classify an index raster using ascending thresholds.

	Each threshold defines the upper bound for the matching class. Values greater
	than or equal to the last threshold receive the final class value.
	"""
	thresholds = list(thresholds)
	if class_values is None:
		class_values = list(range(len(thresholds) + 1))
	else:
		class_values = list(class_values)

	if len(class_values) != len(thresholds) + 1:
		raise ValueError("class_values must contain exactly one more value than thresholds")

	dtype = np.asarray(class_values).dtype
	classified = xr.full_like(index, class_values[-1], dtype=dtype)

	for threshold, class_value in reversed(list(zip(thresholds, class_values[:-1]))):
		classified = xr.where(index < threshold, class_value, classified)

	return classified
