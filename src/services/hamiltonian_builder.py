from typing import Optional, Callable
import numpy as np
from qibo import hamiltonians
from qibo.symbols import Z
import logging

logger = logging.getLogger(__name__)


class HamiltonianBuilder:
    """
    Builds the TSP Hamiltonian for QAOA given a distance matrix.
    """

    def __init__(self, penalty_tuner: Optional[Callable] = None):
        # Tune penalty weight using provided tuner or default heuristic
        self.penalty_tuner = penalty_tuner or (
            lambda dm: np.max(dm) * (dm.shape[0] ** 2)
        )

    def build(self, distance_matrix: np.ndarray) -> hamiltonians.SymbolicHamiltonian:
        n = distance_matrix.shape[0]
        num_qubits = n * n

        # Tuning penalty weight
        try:
            weight = np.float64(self.penalty_tuner(distance_matrix))
            logger.info(f"Penalty weight set to {weight}.")
        except Exception as e:
            logger.error(f"Failed to tune penalty weight: {e}")
            raise

        # Cost term: vectorized for all valid i,k,j pairs
        i_idx, k_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mask = (i_idx != k_idx) & (distance_matrix != 0)
        flat_i = np.repeat(i_idx[mask], n)
        flat_k = np.repeat(k_idx[mask], n)
        j_pos = np.tile(np.arange(n), mask.sum())
        pos_next = (j_pos + 1) % n
        dist_vals = np.repeat(distance_matrix[i_idx[mask], k_idx[mask]], n)
        qi = flat_i * n + j_pos
        qk = flat_k * n + pos_next
        terms = (dist_vals / 4.0) * (1 - Z(qi) - Z(qk) + Z(qi) * Z(qk))
        H_cost = sum(terms)

        # Constraint terms: row and column sums
        row_terms = []
        for i in range(n):
            ops = Z(i * n + np.arange(n))
            row_sum = np.sum((1 - ops) / 2, axis=0)
            row_terms.append((1 - row_sum) ** 2)
        col_terms = []
        for j in range(n):
            ops = Z(np.arange(n) * n + j)
            col_sum = np.sum((1 - ops) / 2, axis=0)
            col_terms.append((1 - col_sum) ** 2)
        H_cons = sum(row_terms) + sum(col_terms)

        total = H_cost + weight * H_cons
        if not isinstance(total, hamiltonians.SymbolicHamiltonian):
            total = hamiltonians.SymbolicHamiltonian(total, nqubits=num_qubits)
        return total
