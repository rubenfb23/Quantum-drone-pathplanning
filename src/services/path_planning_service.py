# services/path_planning_service.py

"""
Service module for quantum path-planning logic using QAOA for TSP.
Encapsulates the core logic, backend setup, Hamiltonian creation,
optimization, and result decoding directly from the state vector.
Includes progress bar via tqdm.
"""

from qibo import hamiltonians, models, set_backend, set_precision, gates, get_backend
from qibo.symbols import Z, I
import numpy as np
import time
from tqdm import tqdm  # Importar tqdm


class PathPlanningService:
    """Service class for quantum path-planning using Qibo."""

    def __init__(
        self,
        depth: int = 4,
        optimizer: str = "BFGS",
        penalty_weight: float = None,
        precision: str = "float32",
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

        # --- Backend Setup (sin cambios) ---
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
        # --- Fin Backend Setup ---

    def find_optimal_path(self, points: list) -> list:
        if not points:
            raise ValueError("Input 'points' list cannot be empty.")
        num_points = len(points)
        if num_points < 2:
            print("Warning: Path for < 2 points is trivial.")
            return list(range(num_points))

        print(f"\nStarting TSP optimization for {num_points} points.")
        start_time = time.time()

        # --- Barra de Progreso tqdm ---
        # Define el número total de etapas principales
        total_stages = 5
        # Crea el objeto tqdm, se cerrará automáticamente al salir del bloque 'with'
        with tqdm(total=total_stages, desc="TSP Optimization Progress") as pbar:

            # Etapa 1: Calcular Matriz de Distancias
            pbar.set_description("Stage 1/5: Calculating distances")
            # print("Calculating distance matrix...") # Ya no es necesario con tqdm
            distance_matrix = self._calculate_distance_matrix(points)
            # print(f"Distance matrix calculated (shape: {distance_matrix.shape}, dtype: {distance_matrix.dtype}).")
            pbar.update(1)  # Actualizar barra después de la etapa 1

            # Etapa 2: Crear Hamiltoniano
            pbar.set_description("Stage 2/5: Building Hamiltonian")
            # print("Creating TSP Hamiltonian...")
            hamiltonian = self._create_tsp_hamiltonian(distance_matrix)
            num_qubits = hamiltonian.nqubits
            # print(f"Hamiltonian created for {num_qubits} qubits.")
            pbar.update(1)  # Actualizar barra después de la etapa 2

            # Etapa 3: Inicializar Modelo QAOA
            pbar.set_description("Stage 3/5: Initializing QAOA")
            # print(f"Initializing QAOA model (depth p={self.depth})...")
            qaoa = models.QAOA(hamiltonian)
            initial_parameters = np.random.uniform(0, 0.1, 2 * self.depth).astype(
                self.numpy_real_dtype
            )
            pbar.update(1)  # Actualizar barra después de la etapa 3 (rápida)

            # Etapa 4: Optimizar Parámetros (La más larga)
            pbar.set_description(f"Stage 4/5: Optimizing ({self.optimizer})")
            # print(f"Optimizing {len(initial_parameters)} QAOA parameters (dtype: {initial_parameters.dtype}) using '{self.optimizer}'...")
            try:
                # NOTA: La barra no avanzará *durante* esta llamada, solo después.
                best_energy, best_params, _ = qaoa.minimize(
                    initial_parameters,
                    method=self.optimizer,
                    options={
                        "disp": False
                    },  # Mantener en False para no interferir con tqdm
                )
                print(
                    f"\nOptimization finished. Best energy found: {best_energy:.4f}"
                )  # Salto de línea para separar de tqdm
            except Exception as e:
                pbar.close()  # Asegurarse de cerrar la barra en caso de error aquí
                print(f"\nError during QAOA optimization: {e}")
                raise RuntimeError("QAOA parameter optimization failed.") from e
            pbar.update(1)  # Actualizar barra después de la etapa 4

            # Etapa 5: Ejecutar Circuito Final y Decodificar
            pbar.set_description("Stage 5/5: Final execution & decoding")
            # print("Executing final QAOA circuit with optimal parameters...")
            qaoa.set_parameters(best_params)
            final_state_vector = qaoa.execute()
            # print(f"Final state vector obtained (shape: {final_state_vector.shape}, dtype: {final_state_vector.dtype}).")

            # print("Decoding result from final state vector...")
            try:
                if hasattr(final_state_vector, "get"):
                    state_vector_np = final_state_vector.get()
                else:
                    state_vector_np = final_state_vector

                probabilities = np.abs(state_vector_np) ** 2
                sorted_indices = np.argsort(probabilities)[::-1]
                best_valid_path = None
                found_valid = False
                # print("Searching for the most probable valid TSP state...") # Menos verboso
                for idx in sorted_indices:
                    prob = probabilities[idx]
                    if prob < 1e-6 and found_valid:
                        break
                    state_binary = format(idx, f"0{num_qubits}b")
                    if self._is_valid_permutation_matrix(state_binary, num_points):
                        # print(f"Found valid state: index={idx}, probability={prob:.4f}, bitstring='{state_binary[:20]}...'")
                        best_valid_path = self._decode_binary_state_to_path(
                            state_binary, num_points
                        )
                        found_valid = True
                        break

                if best_valid_path is None:
                    print(
                        "\nWarning: No valid TSP state found among highly probable states."
                    )
                    most_probable_index = int(np.argmax(probabilities))
                    max_prob = probabilities[most_probable_index]
                    print(
                        f"Reporting path from overall most probable state (index {most_probable_index}, prob {max_prob:.4f}), which is likely invalid."
                    )
                    state_binary = format(most_probable_index, f"0{num_qubits}b")
                    path = self._decode_binary_state_to_path(state_binary, num_points)
                else:
                    path = best_valid_path
                    # print(f"Using decoded path from most probable valid state: {path}") # Menos verboso

            except Exception as e:
                pbar.close()  # Asegurarse de cerrar la barra en caso de error aquí
                print(f"\nError during state vector decoding: {e}")
                raise RuntimeError("Failed to decode the final state vector.") from e
            pbar.update(1)  # Actualizar barra después de la etapa 5

            # --- Fin Barra de Progreso ---

        end_time = time.time()
        # Salto de línea antes del resultado final para separarlo de la barra completada
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")
        return path

    # --- MÉTODOS AUXILIARES (_calculate_distance_matrix, _create_tsp_hamiltonian, etc.) ---
    # (Sin cambios respecto a la versión anterior que funcionaba bien)

    def _calculate_distance_matrix(self, points: list) -> np.ndarray:
        # ... (código sin cambios) ...
        num_points = len(points)
        points_array = np.array(points, dtype=self.numpy_real_dtype)
        diff = points_array[:, np.newaxis, :] - points_array[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        return distance_matrix

    def _create_tsp_hamiltonian(
        self, distance_matrix: np.ndarray
    ) -> hamiltonians.SymbolicHamiltonian:
        # ... (código sin cambios, usando penalty_weight = max_dist * (n**2)) ...
        n = len(distance_matrix)
        num_qubits = n * n
        if self.penalty_weight is None:
            max_dist = np.max(distance_matrix)
            if max_dist == 0:
                max_dist = 1.0
            self.penalty_weight = self.numpy_real_dtype(max_dist * (n**2))
            # print(f"Auto-calculated penalty weight (scaled by n^2 * max_dist): {self.penalty_weight:.2f}") # Menos verboso
        else:
            self.penalty_weight = self.numpy_real_dtype(self.penalty_weight)
            # print(f"Using provided penalty weight: {self.penalty_weight:.2f}")
        H_cost = 0
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
        H_cons = 0
        for i in range(n):
            row_sum_term = sum([(1 - Z(i * n + j)) / 2 for j in range(n)])
            H_cons += (1 - row_sum_term) ** 2
        for j in range(n):
            col_sum_term = sum([(1 - Z(i * n + j)) / 2 for i in range(n)])
            H_cons += (1 - col_sum_term) ** 2
        total_hamiltonian_symbolic = H_cost + self.penalty_weight * H_cons
        if not isinstance(total_hamiltonian_symbolic, hamiltonians.SymbolicHamiltonian):
            final_H = hamiltonians.SymbolicHamiltonian(
                total_hamiltonian_symbolic, nqubits=num_qubits
            )
        else:
            final_H = total_hamiltonian_symbolic
        inferred_qubits = getattr(final_H, "nqubits", None)
        if inferred_qubits is not None and inferred_qubits != num_qubits:
            # print(f"Warning: SymbolicHamiltonian inferred {inferred_qubits} qubits, expected {num_qubits}. Forcing qubit count.")
            final_H = hamiltonians.SymbolicHamiltonian(
                final_H.formula, nqubits=num_qubits
            )
        elif inferred_qubits is None and num_qubits > 0:
            # print(f"Warning: Hamiltonian seems constant. Forcing qubit count to {num_qubits}.")
            final_H = hamiltonians.SymbolicHamiltonian(
                final_H.formula, nqubits=num_qubits
            )
        return final_H

    def _decode_binary_state_to_path(self, state_binary: str, num_points: int) -> list:
        # ... (código sin cambios) ...
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
            row_indices = np.argmax(matrix, axis=0)
            path = list(row_indices)
            if len(set(path)) != num_points:
                pass  # Warning handled in caller if needed
        except Exception as e:
            print(f"Error during path decoding from matrix: {e}")
            raise RuntimeError("Could not decode path from the state matrix.")
        return [int(p) for p in path]

    def _is_valid_permutation_matrix(self, state_binary: str, num_points: int) -> bool:
        # ... (código sin cambios) ...
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
