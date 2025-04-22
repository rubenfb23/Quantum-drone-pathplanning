# services/path_planning_service.py

"""
Service module for quantum path-planning logic using QAOA for TSP.
Encapsulates the core logic, backend setup, Hamiltonian creation,
optimization, and result decoding directly from the state vector.
"""

# Import set_precision along with other qibo functions
from qibo import hamiltonians, models, set_backend, set_precision, gates
from qibo.symbols import Z, I  # I might be useful for Hamiltoniano nulo
import numpy as np
import time  # Para medir tiempos


class PathPlanningService:
    """Service class for quantum path-planning using Qibo."""

    def __init__(
        self,
        depth: int = 4,
        optimizer: str = "BFGS",
        penalty_weight: float = None,
        precision: str = "float32",
        gate_fusion: bool = True,  # Note: gate_fusion is often controlled by the backend choice itself
        devices: list = None,
    ):
        """Initialize the Qibo backend and QAOA parameters.

        Args:
            depth (int): The depth p of the QAOA circuit.
            optimizer (str): Classical optimizer for QAOA parameters.
            penalty_weight (float): Weight for the constraint Hamiltonian. If None, calculated based on max distance * n.
            precision (str): Numerical precision ("float32" or "float64"). "float32" recommended for GPU.
            gate_fusion (bool): Hint to enable gate fusion optimization (effect depends on backend).
            devices (list): List of GPU device IDs for multi-GPU execution (e.g., [0, 1]).
        """
        self.depth = depth
        self.optimizer = optimizer
        self.penalty_weight = penalty_weight
        self.gate_fusion = gate_fusion
        self.precision = precision
        self.numpy_precision = np.float32 if precision == "float32" else np.float64
        self.cached_distance_matrix = None
        self.cached_points_hash = None

        try:
            # --- Corrected Backend Setup ---
            # 1. Set backend and platform first, without precision.
            print(
                f"Attempting to set Qibo backend to 'qibojit' with platform='cuquantum'..."
            )
            backend_config = {"platform": "cuquantum"}  # Config without precision
            if devices:
                backend_config["devices"] = devices
                print(f"Utilizing GPU devices: {devices}")

            set_backend("qibojit", **backend_config)
            print("Backend 'qibojit' with 'cuquantum' set successfully.")

            # 2. Set precision *after* the backend is set.
            print(f"Setting precision to '{self.precision}'...")
            set_precision(self.precision)
            print(f"Precision set to '{self.precision}'.")
            # --- End Correction ---

        except Exception as e:
            print(
                f"\nWarning: Failed to set Qibo backend 'qibojit' with 'cuquantum'. Error: {e}"
            )
            print("Falling back to default 'numpy' backend (CPU).")
            # --- Corrected Fallback Setup ---
            # 1. Set numpy backend
            set_backend("numpy")
            # 2. Set precision for the numpy backend
            print(f"Setting precision for numpy backend to '{self.precision}'...")
            set_precision(self.precision)
            print(f"Precision set to '{self.precision}' for numpy backend.")
            # --- End Correction ---

    def find_optimal_path(self, points: list) -> list:
        """
        Find the optimal path for the given points using QAOA and state vector simulation.

        Args:
            points (list): A list of (x, y) coordinate tuples.

        Returns:
            list: The sequence of point indices representing the most probable valid path.
        """
        if not points:
            raise ValueError("Input 'points' list cannot be empty.")

        num_points = len(points)
        if num_points < 2:
            print("Warning: Path for < 2 points is trivial.")
            return list(range(num_points))

        print(f"\nStarting TSP optimization for {num_points} points.")
        start_time = time.time()

        # 1. Calcular matriz de distancias (con precisión optimizada)
        print("Calculating distance matrix...")
        distance_matrix = self._calculate_distance_matrix(points)
        print(
            f"Distance matrix calculated (shape: {distance_matrix.shape}, dtype: {distance_matrix.dtype})."
        )

        # 2. Crear el Hamiltoniano TSP
        print("Creating TSP Hamiltonian...")
        hamiltonian = self._create_tsp_hamiltonian(distance_matrix)
        num_qubits = hamiltonian.nqubits
        if num_qubits != num_points * num_points:
            print(
                f"Warning: Hamiltonian qubit count ({num_qubits}) differs from expected ({num_points*num_points})."
            )
        print(f"Hamiltonian created for {num_qubits} qubits.")

        # 3. Inicializar el modelo QAOA
        print(f"Initializing QAOA model (depth p={self.depth})...")
        qaoa = models.QAOA(hamiltonian)

        # 4. Optimizar los parámetros variacionales (ángulos gamma y beta)
        initial_parameters = np.random.uniform(0, 0.1, 2 * self.depth).astype(
            self.numpy_precision
        )
        print(
            f"Optimizing {len(initial_parameters)} QAOA parameters using '{self.optimizer}'..."
        )

        try:
            best_energy, best_params, _ = qaoa.minimize(
                initial_parameters,
                method=self.optimizer,
                options={"disp": False},  # Set True for optimizer details
            )
            print(f"Optimization finished. Best energy found: {best_energy:.4f}")
        except Exception as e:
            print(f"Error during QAOA optimization: {e}")
            raise RuntimeError("QAOA parameter optimization failed.") from e

        # 5. Ejecutar el circuito QAOA con parámetros óptimos -> Obtener vector de estado final
        print("Executing final QAOA circuit with optimal parameters...")
        qaoa.set_parameters(best_params)
        final_state_vector = qaoa.execute()
        print(
            f"Final state vector obtained (shape: {final_state_vector.shape}, dtype: {final_state_vector.dtype})."
        )

        # 6. Decodificar el resultado directamente del vector de estado
        print("Decoding result from final state vector...")
        try:
            if hasattr(final_state_vector, "get"):  # Handle GPU tensors (CuPy)
                state_vector_np = final_state_vector.get()
            else:  # Already numpy
                state_vector_np = final_state_vector

            probabilities = np.abs(state_vector_np) ** 2
            most_probable_index = int(np.argmax(probabilities))
            max_prob = probabilities[most_probable_index]
            print(
                f"Most probable state index: {most_probable_index} (Probability: {max_prob:.4f})"
            )

            state_binary = format(most_probable_index, f"0{num_qubits}b")
            path = self._decode_binary_state_to_path(state_binary, num_points)
            print(f"Decoded path from most probable state: {path}")

            if not self._is_valid_permutation_matrix(state_binary, num_points):
                print(
                    "Warning: The most probable state found does not represent a valid TSP permutation."
                )

        except Exception as e:
            print(f"Error during state vector decoding: {e}")
            raise RuntimeError("Failed to decode the final state vector.") from e

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds.")

        return path

    def _calculate_distance_matrix(self, points: list) -> np.ndarray:
        """Calculate the Euclidean distance matrix using vectorized operations."""
        num_points = len(points)
        points_array = np.array(points, dtype=self.numpy_precision)
        diff = points_array[:, np.newaxis, :] - points_array[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        return distance_matrix

    def _create_tsp_hamiltonian(
        self, distance_matrix: np.ndarray
    ) -> hamiltonians.SymbolicHamiltonian:
        """
        Create the TSP Hamiltonian using QUBO encoding with squared constraints.
        H = H_cost + penalty * H_constraints
        """
        n = len(distance_matrix)
        num_qubits = n * n

        if self.penalty_weight is None:
            max_dist = np.max(distance_matrix)
            self.penalty_weight = float(max_dist * n * 1.5)
            print(f"Auto-calculated penalty weight: {self.penalty_weight:.2f}")
        else:
            print(f"Using provided penalty weight: {self.penalty_weight:.2f}")

        # H_cost: Sum d[i,k] * b_ij * b_k,next
        H_cost = 0
        print("Building H_cost...")
        for i in range(n):
            for k in range(n):
                if i == k:
                    continue
                dist = distance_matrix[i, k]
                if dist == 0:
                    continue
                for j in range(n):
                    pos_j = j
                    pos_next = (j + 1) % n
                    q_i_j = i * n + pos_j
                    q_k_next = k * n + pos_next
                    term = (dist / 4.0) * (
                        1 - Z(q_i_j) - Z(q_k_next) + Z(q_i_j) * Z(q_k_next)
                    )
                    H_cost += term

        # H_cons: Squared constraints
        H_cons = 0
        print("Building H_cons...")
        for i in range(n):
            row_sum_term = sum([(1 - Z(i * n + j)) / 2 for j in range(n)])
            H_cons += (1 - row_sum_term) ** 2
        for j in range(n):
            col_sum_term = sum([(1 - Z(i * n + j)) / 2 for i in range(n)])
            H_cons += (1 - col_sum_term) ** 2

        print("Hamiltonian terms built. Combining H_cost and H_cons...")
        total_hamiltonian_symbolic = H_cost + self.penalty_weight * H_cons

        # Ensure correct type and qubit count
        if not isinstance(total_hamiltonian_symbolic, hamiltonians.SymbolicHamiltonian):
            final_H = hamiltonians.SymbolicHamiltonian(
                total_hamiltonian_symbolic, nqubits=num_qubits
            )
        else:
            final_H = total_hamiltonian_symbolic

        if hasattr(final_H, "nqubits") and final_H.nqubits != num_qubits:
            print(
                f"Warning: SymbolicHamiltonian inferred {final_H.nqubits} qubits, expected {num_qubits}. Forcing qubit count."
            )
            final_H = hamiltonians.SymbolicHamiltonian(
                final_H.formula, nqubits=num_qubits
            )
        elif not hasattr(final_H, "nqubits"):
            # If it doesn't even have nqubits, recreate forcing it (edge case)
            final_H = hamiltonians.SymbolicHamiltonian(
                final_H.formula, nqubits=num_qubits
            )

        return final_H

    def _decode_binary_state_to_path(self, state_binary: str, num_points: int) -> list:
        """Decodes the most probable binary string state into a TSP path."""
        num_qubits = num_points * num_points
        state_binary = state_binary.zfill(num_qubits)  # Ensure correct length

        try:
            values = list(map(int, list(state_binary)))
            matrix = np.array(values).reshape((num_points, num_points))
        except Exception as e:
            raise RuntimeError(
                f"Failed to reshape binary state '{state_binary}' to matrix: {e}"
            )

        path = [-1] * num_points
        try:
            # Use argmax on columns: path[j] = city i that is in position j
            row_indices = np.argmax(matrix, axis=0)
            path = list(row_indices)

            if len(set(path)) != num_points:
                print(
                    f"Warning: Decoded path {path} does not contain unique cities. State matrix likely invalid."
                )

        except Exception as e:
            print(f"Error during path decoding from matrix: {e}")
            raise RuntimeError("Could not decode path from the state matrix.")

        return [int(p) for p in path]

    def _is_valid_permutation_matrix(self, state_binary: str, num_points: int) -> bool:
        """Checks if the binary string represents a valid permutation matrix."""
        num_qubits = num_points * num_points
        state_binary = state_binary.zfill(num_qubits)

        try:
            values = list(map(int, list(state_binary)))
            matrix = np.array(values).reshape((num_points, num_points))
            row_sums = np.sum(matrix, axis=1)
            col_sums = np.sum(matrix, axis=0)
            return np.all(row_sums == 1) and np.all(col_sums == 1)
        except Exception:
            return False
