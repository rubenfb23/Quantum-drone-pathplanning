# services/path_planning_service.py

"""
Service module for quantum path-planning logic using QAOA for TSP.
Encapsulates the core logic, backend setup, Hamiltonian creation,
optimization, and result decoding directly from the state vector.
"""

from qibo import hamiltonians, models, set_backend, set_precision, gates, get_backend
from qibo.symbols import Z, I
import numpy as np
import time


class PathPlanningService:
    """Service class for quantum path-planning using Qibo."""

    def __init__(
        self,
        depth: int = 4,
        optimizer: str = "BFGS",
        penalty_weight: float = None,
        precision: str = "float32",  # User input: "float32" or "float64"
        gate_fusion: bool = True,
        devices: list = None,
    ):
        self.depth = depth
        self.optimizer = optimizer
        self.penalty_weight = penalty_weight
        self.gate_fusion = gate_fusion
        self.user_precision_str = precision
        self.qibo_precision_str = "single" if precision == "float32" else "double"
        self.numpy_real_dtype = np.float32 if precision == "float32" else np.float64
        self.cached_distance_matrix = None
        self.cached_points_hash = None

        try:
            print(
                f"Attempting to set Qibo backend to 'qibojit' with platform='cuquantum'..."
            )
            backend_config = {"platform": "cuquantum"}
            if devices:
                backend_config["devices"] = devices
                print(f"Utilizing GPU devices: {devices}")

            set_backend("qibojit", **backend_config)
            print(
                f"Backend 'qibojit' with 'cuquantum' set successfully. Using {get_backend().device}"
            )

            print(
                f"Setting precision to '{self.qibo_precision_str}' (corresponds to user '{self.user_precision_str}')..."
            )
            set_precision(self.qibo_precision_str)
            print(
                f"Precision set to '{self.qibo_precision_str}' ({get_backend().dtype})."
            )

        except Exception as e:
            print(
                f"\nWarning: Failed to set Qibo backend 'qibojit' with 'cuquantum'. Error: {e}"
            )
            print("Falling back to default 'numpy' backend (CPU).")
            set_backend("numpy")
            print(
                f"Setting precision for numpy backend to '{self.qibo_precision_str}' (corresponds to user '{self.user_precision_str}')..."
            )
            try:
                set_precision(self.qibo_precision_str)
                print(
                    f"Precision set to '{self.qibo_precision_str}' ({get_backend().dtype}) for numpy backend."
                )
            except Exception as pe:
                print(
                    f"\nError setting precision '{self.qibo_precision_str}' on numpy fallback: {pe}"
                )
                print("Continuing with default numpy precision.")

    def find_optimal_path(self, points: list) -> list:
        if not points:
            raise ValueError("Input 'points' list cannot be empty.")
        num_points = len(points)
        if num_points < 2:
            print("Warning: Path for < 2 points is trivial.")
            return list(range(num_points))

        print(f"\nStarting TSP optimization for {num_points} points.")
        start_time = time.time()

        print("Calculating distance matrix...")
        distance_matrix = self._calculate_distance_matrix(points)
        print(
            f"Distance matrix calculated (shape: {distance_matrix.shape}, dtype: {distance_matrix.dtype})."
        )

        print("Creating TSP Hamiltonian...")
        hamiltonian = self._create_tsp_hamiltonian(distance_matrix)
        num_qubits = hamiltonian.nqubits
        if num_qubits != num_points * num_points:
            print(
                f"Warning: Hamiltonian qubit count ({num_qubits}) differs from expected ({num_points*num_points})."
            )
        print(f"Hamiltonian created for {num_qubits} qubits.")

        print(f"Initializing QAOA model (depth p={self.depth})...")
        qaoa = models.QAOA(hamiltonian)

        initial_parameters = np.random.uniform(0, 0.1, 2 * self.depth).astype(
            self.numpy_real_dtype
        )
        print(
            f"Optimizing {len(initial_parameters)} QAOA parameters (dtype: {initial_parameters.dtype}) using '{self.optimizer}'..."
        )

        try:
            # Note: Scipy optimizers might show warnings (like the division warning)
            # which don't necessarily mean failure, but indicate numerical difficulties.
            best_energy, best_params, _ = qaoa.minimize(
                initial_parameters,
                method=self.optimizer,
                options={"disp": False},  # Change to True for more optimizer output
            )
            print(f"Optimization finished. Best energy found: {best_energy:.4f}")
        except Exception as e:
            print(f"Error during QAOA optimization: {e}")
            raise RuntimeError("QAOA parameter optimization failed.") from e

        print("Executing final QAOA circuit with optimal parameters...")
        qaoa.set_parameters(best_params)
        final_state_vector = qaoa.execute()
        print(
            f"Final state vector obtained (shape: {final_state_vector.shape}, dtype: {final_state_vector.dtype})."
        )

        print("Decoding result from final state vector...")
        try:
            if hasattr(final_state_vector, "get"):
                state_vector_np = final_state_vector.get()
            else:
                state_vector_np = final_state_vector

            probabilities = np.abs(state_vector_np) ** 2
            # --- Find *Valid* Most Probable State ---
            # Instead of just argmax, sort by probability and find the first *valid* one
            sorted_indices = np.argsort(probabilities)[
                ::-1
            ]  # Indices from most to least probable

            best_valid_path = None
            best_valid_prob = 0
            found_valid = False

            print("Searching for the most probable valid TSP state...")
            for idx in sorted_indices:
                prob = probabilities[idx]
                # Stop searching if probability gets too low (e.g., < 1e-5) or after checking top N states?
                if (
                    prob < 1e-6 and found_valid
                ):  # Optimization: stop if probability is tiny AND we already found one
                    break

                state_binary = format(idx, f"0{num_qubits}b")
                if self._is_valid_permutation_matrix(state_binary, num_points):
                    print(
                        f"Found valid state: index={idx}, probability={prob:.4f}, bitstring='{state_binary[:20]}...'"
                    )
                    best_valid_path = self._decode_binary_state_to_path(
                        state_binary, num_points
                    )
                    best_valid_prob = prob
                    found_valid = True
                    break  # Found the most probable valid one

            if best_valid_path is None:
                # If no valid state found (even with low probability), report the most probable overall (as before)
                print("Warning: No valid TSP state found among highly probable states.")
                most_probable_index = int(np.argmax(probabilities))
                max_prob = probabilities[most_probable_index]
                print(
                    f"Reporting path from overall most probable state (index {most_probable_index}, prob {max_prob:.4f}), which is likely invalid."
                )
                state_binary = format(most_probable_index, f"0{num_qubits}b")
                path = self._decode_binary_state_to_path(state_binary, num_points)
            else:
                path = best_valid_path
                print(f"Using decoded path from most probable valid state: {path}")

        except Exception as e:
            print(f"Error during state vector decoding: {e}")
            raise RuntimeError("Failed to decode the final state vector.") from e

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds.")
        return path

    def _calculate_distance_matrix(self, points: list) -> np.ndarray:
        num_points = len(points)
        points_array = np.array(points, dtype=self.numpy_real_dtype)
        diff = points_array[:, np.newaxis, :] - points_array[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        return distance_matrix

    def _create_tsp_hamiltonian(
        self, distance_matrix: np.ndarray
    ) -> hamiltonians.SymbolicHamiltonian:
        n = len(distance_matrix)
        num_qubits = n * n

        if self.penalty_weight is None:
            max_dist = np.max(distance_matrix)
            if max_dist == 0:  # Avoid penalty=0 if all points are the same
                max_dist = 1.0
            # --- Increased Penalty Weight ---
            # Scale penalty with n^2 and max distance to ensure constraints are met
            self.penalty_weight = self.numpy_real_dtype(max_dist * (n**2))
            print(
                f"Auto-calculated penalty weight (scaled by n^2 * max_dist): {self.penalty_weight:.2f}"
            )
            # --- End Change ---
        else:
            self.penalty_weight = self.numpy_real_dtype(self.penalty_weight)
            print(f"Using provided penalty weight: {self.penalty_weight:.2f}")

        # H_cost
        H_cost = 0
        # print("Building H_cost...") # Reduced verbosity
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

        # H_cons
        H_cons = 0
        # print("Building H_cons...") # Reduced verbosity
        for i in range(n):
            row_sum_term = sum([(1 - Z(i * n + j)) / 2 for j in range(n)])
            H_cons += (1 - row_sum_term) ** 2
        for j in range(n):
            col_sum_term = sum([(1 - Z(i * n + j)) / 2 for i in range(n)])
            H_cons += (1 - col_sum_term) ** 2

        # print("Hamiltonian terms built. Combining H_cost and H_cons...")
        total_hamiltonian_symbolic = H_cost + self.penalty_weight * H_cons

        if not isinstance(total_hamiltonian_symbolic, hamiltonians.SymbolicHamiltonian):
            final_H = hamiltonians.SymbolicHamiltonian(
                total_hamiltonian_symbolic, nqubits=num_qubits
            )
        else:
            final_H = total_hamiltonian_symbolic

        inferred_qubits = getattr(final_H, "nqubits", None)
        if inferred_qubits is not None and inferred_qubits != num_qubits:
            print(
                f"Warning: SymbolicHamiltonian inferred {inferred_qubits} qubits, expected {num_qubits}. Forcing qubit count."
            )
            final_H = hamiltonians.SymbolicHamiltonian(
                final_H.formula, nqubits=num_qubits
            )
        elif inferred_qubits is None and num_qubits > 0:
            print(
                f"Warning: Hamiltonian seems constant. Forcing qubit count to {num_qubits}."
            )
            final_H = hamiltonians.SymbolicHamiltonian(
                final_H.formula, nqubits=num_qubits
            )

        return final_H

    def _decode_binary_state_to_path(self, state_binary: str, num_points: int) -> list:
        num_qubits = num_points * num_points
        state_binary = state_binary.zfill(num_qubits)
        try:
            values = list(map(int, list(state_binary)))
            matrix = np.array(values).reshape((num_points, num_points))
        except Exception as e:
            raise RuntimeError(
                f"Failed to reshape binary state '{state_binary}' to matrix: {e}"
            )

        path = [-1] * num_points
        try:
            row_indices = np.argmax(matrix, axis=0)  # city index for each position j
            path = list(row_indices)
            # Basic check if path seems valid (unique cities)
            if len(set(path)) != num_points:
                # This might happen if the input state_binary wasn't a valid permutation
                print(
                    f"Internal Warning: Decoded path {path} from supposedly valid state does not contain unique cities."
                )
        except Exception as e:
            print(f"Error during path decoding from matrix: {e}")
            raise RuntimeError("Could not decode path from the state matrix.")
        return [int(p) for p in path]

    def _is_valid_permutation_matrix(self, state_binary: str, num_points: int) -> bool:
        num_qubits = num_points * num_points
        state_binary = state_binary.zfill(num_qubits)
        try:
            values = list(map(int, list(state_binary)))
            matrix = np.array(values).reshape((num_points, num_points))
            row_sums = np.sum(matrix, axis=1)
            col_sums = np.sum(matrix, axis=0)
            return np.all(row_sums == 1) and np.all(col_sums == 1)
        except Exception:
            # If reshape or conversion fails, it's not valid
            return False
