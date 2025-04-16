# services/path_planning_service.py

"""
Service module for quantum path-planning logic.
This module encapsulates the core logic for the quantum algorithm.
"""
from qibo import hamiltonians, models, set_backend
from qibo.optimizers import optimize
from qibo.symbols import Z, X
import numpy as np


class PathPlanningService:
    """Service class for quantum path-planning using Qibo."""

    def __init__(self):
        """Initialize the Qibo backend and QAOA parameters."""
        set_backend("qibojit", platform="numba")  # Optimized for CPU
        self.depth = 2  # QAOA depth
        self.optimizer = "BFGS"  # Classical optimizer

    def execute(self, points):
        """
        Execute the quantum path-planning algorithm with the given points.
        """
        distance_matrix = self._calculate_distance_matrix(points)
        num_points = len(distance_matrix)
        num_qubits = (
            num_points**2
        )  # Assuming TSP encoding requires n^2 qubits for n points
        hamiltonian = self._create_tsp_hamiltonian(distance_matrix)

        # Define a mixer Hamiltonian using qibo.symbols.X
        # Sum Pauli X operators over all qubits
        mixer_expr = sum(X(i) for i in range(num_qubits))
        mixer = hamiltonians.SymbolicHamiltonian(mixer_expr, nqubits=num_qubits)

        # Initialize QAOA model with the cost and mixer Hamiltonians
        qaoa = models.QAOA(hamiltonian, mixer=mixer)
        # qaoa.compile() removed because compile() is not available

        # Create initial parameters - for depth self.depth, we assume 2*self.depth parameters
        initial_parameters = np.random.uniform(0, 2 * np.pi, 2 * self.depth)

        # Replace direct optimization with an objective function that updates parameters.
        def objective_function(params):
            qaoa.set_parameters(np.array(params))
            cost = qaoa()  # Evaluate cost
            return float(np.squeeze(cost)[0])  # Squeeze cost and take first element

        result_opt = optimize(
            objective_function, initial_parameters, method=self.optimizer
        )

        # Handle optimize result - could be tuple or direct array
        if isinstance(result_opt, tuple):
            best_params = result_opt[0]
        else:
            best_params = result_opt

        best_params = np.array(best_params)  # Ensure best_params has a 'shape'
        qaoa.set_parameters(best_params)

        # Check if qaoa.params has an empty shape, and if so, initialize it with a default
        if qaoa.params.shape == ():
            qaoa.params = np.zeros(2)

        # Execute the circuit and get state vector
        state = qaoa.execute()

        # Replace qaoa.sample(nshots=1000) with custom sampling
        samples = self._sample_state(state, nshots=1000, num_qubits=num_qubits)

        return self._decode_result(samples, points)

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
        Create the TSP Hamiltonian using Qibo symbolic expressions.
        NOTE: This is a simplified placeholder. A correct TSP Hamiltonian
              encoding (e.g., QUBO or Ising) is complex.
              The number of qubits is n_cities * n_positions_in_path (often n^2).
        """
        num_points = len(distance_matrix)
        n_qubits = num_points**2  # Qubits q_{i,p} = 1 if city i is at path position p

        hamiltonian_expr = 0
        # Example Cost Part (sum distances for adjacent paths):
        for i in range(num_points):
            for j in range(i + 1, num_points):  # Avoid duplicates and i==j
                weight = distance_matrix[i, j]
                if i * num_points + j < n_qubits:  # Basic check
                    hamiltonian_expr += weight * Z(i * num_points + j)  # Placeholder

        # Constraint terms (e.g., each city visited once, each position filled once)
        for k in range(n_qubits):
            hamiltonian_expr += 1.0 * (1 - Z(k)) / 2  # Example penalty if qubit k is 0

        if hamiltonian_expr == 0:  # Handle case where no terms were added
            print("Warning: Hamiltonian expression is empty. Using a default Z(0).")
            hamiltonian_expr = Z(0)

        hamiltonian = hamiltonians.SymbolicHamiltonian(
            hamiltonian_expr, nqubits=n_qubits
        )
        return hamiltonian

    def _decode_result(self, samples, points):
        """
        Decode the result of the quantum computation into a valid path.
        """
        print("Sampling results:", samples)

        # Find the most frequent result
        if isinstance(samples, dict):
            # If samples is a dictionary of bitstrings and counts
            most_probable_state = (
                max(samples, key=samples.get) if samples else "0" * (len(points) ** 2)
            )
        else:
            # If samples is a different format, try to handle it
            try:
                most_probable_state = samples[0]  # Take the first sample as fallback
            except (IndexError, TypeError):
                most_probable_state = "0" * (len(points) ** 2)

        print("Most probable state:", most_probable_state)

        # For now, return a simple path through all points
        # In a real implementation, you would decode the quantum result into an optimal path
        return list(range(len(points)))

    def _sample_state(self, state, nshots, num_qubits):
        """
        Sample measurement outcomes from the state vector.
        """
        probabilities = np.abs(state) ** 2
        indices = np.arange(len(probabilities))
        sampled_indices = np.random.choice(indices, size=nshots, p=probabilities)
        samples = [bin(idx)[2:].zfill(num_qubits) for idx in sampled_indices]
        return samples
