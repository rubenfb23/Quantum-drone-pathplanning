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

        # Cost term: loop to avoid numpy.ndarray in qubit indices
        H_cost = None
        for i in range(n):
            for k in range(n):
                if i != k and distance_matrix[i, k] != 0:
                    v = float(distance_matrix[i, k])
                    for j in range(n):
                        qi = i * n + j
                        qk = k * n + ((j + 1) % n)
                        term = (v / 4.0) * (1 - Z(qi) - Z(qk) + Z(qi) * Z(qk))
                        if H_cost is None:
                            H_cost = term
                        else:
                            H_cost = H_cost + term
        # Ensure H_cost is initialized
        if H_cost is None:
            H_cost = hamiltonians.SymbolicHamiltonian(0, nqubits=num_qubits)

        # Constraint terms: enforce one visit per row and column via explicit loops
        row_terms = []
        for i in range(n):
            row_sum_op = None
            for j in range(n):
                op = (1 - Z(i * n + j)) / 2
                row_sum_op = op if row_sum_op is None else row_sum_op + op
            row_terms.append((1 - row_sum_op) ** 2)
        col_terms = []
        for j in range(n):
            col_sum_op = None
            for i in range(n):
                op = (1 - Z(i * n + j)) / 2
                col_sum_op = op if col_sum_op is None else col_sum_op + op
            col_terms.append((1 - col_sum_op) ** 2)
        H_cons = None
        for term in row_terms + col_terms:
            H_cons = term if H_cons is None else H_cons + term

        total = H_cost + weight * H_cons
        if not isinstance(total, hamiltonians.SymbolicHamiltonian):
            total = hamiltonians.SymbolicHamiltonian(total, nqubits=num_qubits)
        return total
