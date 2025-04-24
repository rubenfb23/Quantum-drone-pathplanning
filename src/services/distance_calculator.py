from typing import List
import numpy as np
import numba


class DistanceCalculator:
    """
    Calculates and caches the distance matrix for a set of points.
    """

    def __init__(self):
        self._cached_distance_matrix = None
        self._cached_points_hash = None

    def calculate(self, points: List[tuple], dtype: np.dtype) -> np.ndarray:
        """
        Computes the Euclidean distance matrix for `points` using Numba
        parallel loops.
        Caches results if input hasn't changed.
        """

        @numba.njit(parallel=True)
        def calculate_distances(points_array):
            n = points_array.shape[0]
            result = np.zeros((n, n), dtype=points_array.dtype)
            for i in numba.prange(n):
                for j in range(n):
                    diff = points_array[i] - points_array[j]
                    result[i, j] = np.sqrt(np.sum(diff * diff))
            return result

        points_hash = hash(tuple(points))
        if (
            self._cached_points_hash == points_hash
            and self._cached_distance_matrix is not None
        ):
            return self._cached_distance_matrix

        points_array = np.array(points, dtype=dtype)
        distance_matrix = calculate_distances(points_array)

        self._cached_points_hash = points_hash
        self._cached_distance_matrix = distance_matrix
        return distance_matrix
