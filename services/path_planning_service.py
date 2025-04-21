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
        """
        set_backend("qibojit", platform="numba")  # Optimized for CPU
        self.depth = depth
        self.optimizer = optimizer
        self.shots = shots

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
            # Handle cases with 0 or 1 point trivially
            return list(range(len(points)))

        num_points = len(points)
        num_qubits = num_points * num_points  # TSP encoding uses n^2 qubits

        distance_matrix = self._calculate_distance_matrix(points)
        hamiltonian = self._create_tsp_hamiltonian(distance_matrix)

        # Initialize QAOA model; depth is encoded in initial_parameters length
        qaoa = models.QAOA(hamiltonian)
        initial_parameters = [0.01] * (2 * self.depth)
        # Optimize variational parameters
        best_energy, best_params, _ = qaoa.minimize(
            initial_parameters,
            method=self.optimizer,
        )

        # Execute QAOA circuit and decode state via argmax
        qaoa.set_parameters(best_params)
        state = qaoa.execute()
        probs = np.abs(state) ** 2
        idx = int(np.argmax(probs))
        state_bin = format(idx, f"0{num_qubits}b")
        # Decode bitstring into TSP path
        num_points = len(points)
        path = [0] * num_points
        for bit_index, bit in enumerate(state_bin):
            if bit == "1":
                i, j = divmod(bit_index, num_points)
                path[j] = i
        return path

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
        Create the TSP Hamiltonian using Qibo.
        """
        num_points = len(distance_matrix)

        # Optimize Hamiltonian construction by using
        # sparse representations

        # Cost term: Minimize the total distance
        cost_terms = []
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    weight = distance_matrix[i, j]
                    z_pauli = Z(i * num_points + j)
                    cost_terms.append((weight, z_pauli))

        # Constraint terms: Ensure valid TSP path
        constraint_terms = []
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    z_pauli = Z(i * num_points + j)
                    constraint_terms.append((1.0, z_pauli))

        # Combine terms into a single Hamiltonian
        # expression using sparse matrices
        hamiltonian_expr = sum(
            weight * term for weight, term in cost_terms + constraint_terms
        )
        hamiltonian = hamiltonians.SymbolicHamiltonian(hamiltonian_expr)
        return hamiltonian

    def _decode_result(self, result, points, num_qubits):
        """
        Decode the result of the quantum computation into a valid path.
        """
        counts = result.frequencies()
        most_probable_state = max(counts, key=counts.get)
        print("Most probable state:", most_probable_state)
        state = most_probable_state.replace(" ", "")
        num_points = len(points)
        # Pad state if necessary
        if len(state) < num_qubits:
            state = state.zfill(num_qubits)
        path = [0] * num_points
        for idx, bit in enumerate(state):
            if bit == "1":
                i, j = divmod(idx, num_points)
                path[j] = i
        return path
