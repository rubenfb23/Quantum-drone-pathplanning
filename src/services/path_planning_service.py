# services/path_planning_service.py

"""
Service module for quantum path-planning logic.
This module encapsulates the core logic for the quantum algorithm.
"""
from qibo import hamiltonians, models, set_backend
from qibo.symbols import Z  # removed X import
import numpy as np

# Remove sympy import if no longer needed elsewhere, or keep if used
# import sympy as sp


class PathPlanningService:
    """Service class for quantum path-planning using Qibo."""

    def __init__(
        self,
        depth: int = 2,
        optimizer: str = "BFGS",
        shots: int = 100,
        penalty_weight: float = None,  # added
    ):
        """Initialize the Qibo backend and QAOA parameters.

        Args:
            depth (int): The depth of the QAOA circuit.
                Defaults to 2.
            optimizer (str):
                The classical optimizer to use.
                Defaults to "BFGS".
            shots (int): The number of shots for the final circuit execution.
                Defaults to 100.
            penalty_weight (float): The penalty weight for the constraints.
                Defaults to None.
        """
        set_backend("qibojit", platform="cuquantum")  # Optimized for GPU
        self.depth = depth
        self.optimizer = optimizer
        self.shots = shots
        self.penalty_weight = penalty_weight  # new

    def find_optimal_path(self, points):
        """
        Find the optimal path for the given points using QAOA.

        Args:
            points (list): A list of (x, y) coordinates.

        Returns:
            list: The sequence of point indices representing the optimal path.
        """
        if not points:
            raise ValueError("Input 'points' list cannot be empty.")
        if len(points) < 2:
            return list(range(len(points)))

        num_points = len(points)
        num_qubits = (
            num_points * num_points
        )  # Se usan n^2 qubits para la codificación TSP

        distance_matrix = self._calculate_distance_matrix(points)
        hamiltonian = self._create_tsp_hamiltonian(distance_matrix)

        # Inicializa el modelo QAOA.
        # La profundidad viene de len(initial_parameters)
        qaoa = models.QAOA(hamiltonian)
        initial_parameters = [0.01] * (2 * self.depth)

        # Optimiza los parámetros variacionales
        best_energy, best_params, _ = qaoa.minimize(
            initial_parameters, method=self.optimizer
        )

        # Ejecuta circuito y obtiene statevector
        qaoa.set_parameters(best_params)
        state = qaoa.execute()
        # Muestra localmente usando el statevector
        counts = self._sample_counts(state, self.shots, num_qubits)
        # Decodifica recuentos de medición
        return self._decode_result(counts, points, num_qubits)

    def _calculate_distance_matrix(self, points):
        """
        Calculate the distance matrix for the given points.
        """
        num_points = len(points)
        distance_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                distance_matrix[i, j] = np.linalg.norm(
                    np.array(points[i]) - np.array(points[j])
                )
        return distance_matrix

    def _create_tsp_hamiltonian(self, distance_matrix):
        """
        True QUBO encoding with squared constraints:
          - b_ij = (1 − Z_ij)/2
          - cost    = Σ_{i≠j} d[i,j] * b_ij
          - constraints = Σ_i (1−Σ_j b_ij)^2 + Σ_j (1−Σ_i b_ij)^2
        """
        num_points = len(distance_matrix)
        n = num_points
        penalty = (
            self.penalty_weight
            if self.penalty_weight is not None
            else np.max(distance_matrix) * n * 10
        )

        # cost term
        H_cost = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    q = i * n + j
                    H_cost += distance_matrix[i, j] * (1 - Z(q)) / 2

        # squared‐constraint terms
        H_cons = 0
        # each city appears exactly once in the tour (row sums)
        for i in range(n):
            row_qubits = [(1 - Z(i * n + j)) / 2 for j in range(n)]
            H_cons += (1 - sum(row_qubits)) ** 2
        # Cada posición en la ruta
        # está ocupada por una ciudad única
        for j in range(n):
            col_qubits = [(1 - Z(i * n + j)) / 2 for i in range(n)]
            H_cons += (1 - sum(col_qubits)) ** 2

        return hamiltonians.SymbolicHamiltonian(H_cost + penalty * H_cons)

    def _decode_result(self, counts, points, num_qubits):
        """
        Decode the result of the quantum computation into a valid path.
        """
        num_points = len(points)
        # If passed a result object with frequencies(), extract counts
        if not isinstance(counts, dict):
            counts = counts.frequencies()
        # Normalize keys by removing spaces
        processed_counts = {s.replace(" ", ""): c for s, c in counts.items()}
        # Select the most probable bitstring satisfying row/column constraints
        valid_state = None
        for s in sorted(
            processed_counts,
            key=processed_counts.get,
            reverse=True,
        ):
            if self._is_valid_bitstring(s, num_points):
                valid_state = s
                break
        if valid_state is None:
            raise ValueError("No valid solution found in measurement results.")
        print("Selected valid state:", valid_state)
        state = valid_state
        # Pad state if necessary
        if len(state) < num_qubits:
            state = state.zfill(num_qubits)
        path = [0] * num_points
        for idx, bit in enumerate(state):
            if bit == "1":
                i, j = divmod(idx, num_points)
                path[j] = i
        return path

    def _is_valid_bitstring(self, bitstring, num_points):
        """
        Check if bitstring is valid TSP encoding.
        Requires one '1' per row and column.
        """
        n = num_points
        if len(bitstring) != n * n:
            return False
        # Check exactly one '1' per row
        for i in range(n):
            if bitstring[i * n : (i + 1) * n].count("1") != 1:
                return False
        # Check exactly one '1' per column
        for j in range(n):
            if sum(bitstring[i * n + j] == "1" for i in range(n)) != 1:
                return False
        return True

    def _sample_counts(self, statevector, shots, num_qubits):
        """
        Generate measurement counts from statevector by sampling.
        """
        # Convert Qibo tensor to numpy array if needed
        if hasattr(statevector, "get"):
            statevector = statevector.get()
        probs = np.abs(statevector) ** 2
        # Sample indices according to probabilities
        indices = np.random.choice(len(probs), size=shots, p=probs)
        from collections import Counter

        counter = Counter(indices)
        # Build bitstring count dict
        freq = {}
        for idx, cnt in counter.items():
            bitstr = format(idx, f"0{num_qubits}b")
            freq[bitstr] = cnt
        return freq
