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

        # Ejecuta el circuito QAOA y decodifica el estado mediante argmax
        qaoa.set_parameters(best_params)
        state = qaoa.execute()  # Se utiliza el vector de estado obtenido
        probs = np.abs(state) ** 2
        idx = int(np.argmax(probs))
        state_bin = format(idx, f"0{num_qubits}b")

        # Convertir bitstring a matriz (num_points x num_points)
        # Cada fila es una ciudad; cada columna es posición en ruta.
        values = list(state_bin)
        matrix = np.array(values, dtype=int)
        matrix = matrix.reshape((num_points, num_points))
        # Decode bitstring: require exactly one '1' per column
        path = [-1] * num_points
        for j in range(num_points):
            col = matrix[:, j]
            ones = np.where(col == 1)[0]
            if len(ones) != 1:
                raise ValueError(
                    f"Invalid quantum output: column {j} has {len(ones)} ones."
                )
            path[j] = ones[0]
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
