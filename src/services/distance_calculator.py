from typing import List
import numpy as np


class DistanceCalculator:
    """
    Calculates and caches the distance matrix for a set of points.
    """

    def __init__(self):
        self._cached_distance_matrix = None
        self._cached_points_hash = None

    def calculate(self, points: List[tuple], dtype: np.dtype) -> np.ndarray:
        """
        Computes Euclidean distance matrix for `points`, caching results if input hasn't changed.
        """
        points_hash = hash(tuple(points))
        if (
            self._cached_points_hash == points_hash
            and self._cached_distance_matrix is not None
        ):
            return self._cached_distance_matrix

        points_array = np.array(points, dtype=dtype)
        diff = points_array[:, None, :] - points_array[None, :, :]
        distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        self._cached_points_hash = points_hash
        self._cached_distance_matrix = distance_matrix
        return distance_matrix
