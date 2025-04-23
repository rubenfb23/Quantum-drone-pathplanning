# Add project src to path for imports
import sys, os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import pytest
import numpy as np
import tqdm as tqdm_module
from qibo import get_backend
from services.path_planning_service import (
    PathPlanningService,
    PathPlanningError,
    SingleTqdm,
)


def test_tune_penalty_weight_default():
    # 2x2 distance matrix with max_dist=3, n=2 -> weight=3*(2**2)=12
    dm = np.array([[0, 3], [3, 0]], dtype=float)
    service = PathPlanningService(seed=42)
    weight = service._tune_penalty_weight(dm)
    assert weight == pytest.approx(12)


def test_create_hamiltonian_nqubits():
    dm = np.array([[0, 1], [1, 0]], dtype=float)
    service = PathPlanningService(penalty_weight=5.0)
    H = service._create_tsp_hamiltonian(dm)
    assert hasattr(H, "nqubits")
    assert H.nqubits == 4


def test_calculate_distance_matrix_caching():
    points = [(0.0, 0.0), (3.0, 4.0)]
    service = PathPlanningService(seed=1)
    mat1 = service._calculate_distance_matrix(points)
    mat2 = service._calculate_distance_matrix(points)
    assert mat1 is mat2  # should return cached object


def test_patched_tqdm_context_manager():
    service = PathPlanningService()
    original = tqdm_module.tqdm
    with service._patched_tqdm():
        assert tqdm_module.tqdm is SingleTqdm
    assert tqdm_module.tqdm is original


def test_find_optimal_path_empty_raises():
    service = PathPlanningService()
    with pytest.raises(PathPlanningError):
        service.find_optimal_path([])
