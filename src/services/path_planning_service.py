# services/path_planning_service.py

"""
Service module for quantum path-planning logic using QAOA for TSP.
Encapsulates the core logic, backend setup, Hamiltonian creation,
optimization, and result decoding directly from the state vector.
"""

from qibo import (
    hamiltonians,
    models,
    set_backend,
    gates,
)  # gates puede ser necesario para QAOA
from qibo.symbols import Z, I  # I puede ser útil para Hamiltoniano nulo
import numpy as np
import time  # Para medir tiempos


class PathPlanningService:
    """Service class for quantum path-planning using Qibo."""

    def __init__(
        self,
        depth: int = 4,  # Profundidad aumentada por defecto
        optimizer: str = "BFGS",
        # shots: int = 100, # 'shots' no se usa si se decodifica del state vector
        penalty_weight: float = None,
        precision: str = "float32",  # Precisión simple por defecto
        gate_fusion: bool = True,  # Fusión de puertas activada por defecto
        devices: list = None,  # Para multi-GPU opcional
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
        self.gate_fusion = (
            gate_fusion  # Guardamos el flag, aunque qibo lo maneja globalmente
        )
        self.precision = precision
        self.numpy_precision = np.float32 if precision == "float32" else np.float64
        self.cached_distance_matrix = None
        self.cached_points_hash = None

        try:
            # Configurar backend QiboJit con cuQuantum, precisión y dispositivos
            print(
                f"Attempting to set Qibo backend to 'qibojit' with platform='cuquantum', precision='{self.precision}'..."
            )
            backend_config = {
                "platform": "cuquantum",
                "precision": self.precision,
                # Gate fusion es generalmente una propiedad del backend/compilador,
                # no un argumento directo aquí, pero 'cuquantum' lo habilita.
            }
            if devices:
                backend_config["devices"] = devices
                print(f"Utilizing GPU devices: {devices}")

            set_backend("qibojit", **backend_config)
            print("Backend 'qibojit' with 'cuquantum' configured successfully.")

        except Exception as e:
            print(
                f"\nWarning: Failed to set Qibo backend 'qibojit' with 'cuquantum'. Error: {e}"
            )
            print("Falling back to default 'numpy' backend (CPU).")
            set_backend(
                "numpy", precision=self.precision
            )  # Usar precisión también en numpy

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
        # Gate fusion se controla a nivel de backend/compilación en Qibo/cuQuantum
        qaoa = models.QAOA(hamiltonian)

        # 4. Optimizar los parámetros variacionales (ángulos gamma y beta)
        # Usar parámetros iniciales pequeños aleatorios con la precisión correcta.
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
                options={"disp": False},  # Poner True para más detalles del optimizador
            )
            print(f"Optimization finished. Best energy found: {best_energy:.4f}")
        except Exception as e:
            print(f"Error during QAOA optimization: {e}")
            raise RuntimeError("QAOA parameter optimization failed.") from e

        # 5. Ejecutar el circuito QAOA con parámetros óptimos -> Obtener vector de estado final
        print("Executing final QAOA circuit with optimal parameters...")
        qaoa.set_parameters(best_params)
        final_state_vector = qaoa.execute()  # Devuelve el statevector
        print(
            f"Final state vector obtained (shape: {final_state_vector.shape}, dtype: {final_state_vector.dtype})."
        )

        # 6. Decodificar el resultado directamente del vector de estado
        print("Decoding result from final state vector...")
        try:
            # Obtener probabilidades |psi|^2
            # Asegurarse de que el vector está en CPU y es numpy para argmax/abs
            if hasattr(final_state_vector, "get"):  # Si es un objeto CuPy/GPU tensor
                state_vector_np = final_state_vector.get()
            else:  # Si ya es numpy (backend numpy)
                state_vector_np = final_state_vector

            probabilities = np.abs(state_vector_np) ** 2
            # print(f"Sum of probabilities: {np.sum(probabilities):.5f}") # Debería ser ~1.0

            # Encontrar el índice del estado con la máxima probabilidad
            most_probable_index = int(np.argmax(probabilities))
            max_prob = probabilities[most_probable_index]
            print(
                f"Most probable state index: {most_probable_index} (Probability: {max_prob:.4f})"
            )

            # Convertir índice a bitstring (asegurando el padding correcto)
            state_binary = format(most_probable_index, f"0{num_qubits}b")

            # Decodificar el bitstring en una ruta TSP
            path = self._decode_binary_state_to_path(state_binary, num_points)
            print(f"Decoded path from most probable state: {path}")

            # Opcional: Verificar si el estado más probable era realmente válido
            if not self._is_valid_permutation_matrix(state_binary, num_points):
                print(
                    "Warning: The most probable state found does not represent a valid TSP permutation (check row/column sums). The decoded path might be incorrect."
                )
                # Aquí se podría implementar una búsqueda del *siguiente* estado más probable que *sea* válido,
                # pero eso añade complejidad. Por ahora, devolvemos el path del más probable.

        except Exception as e:
            print(f"Error during state vector decoding: {e}")
            raise RuntimeError("Failed to decode the final state vector.") from e

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds.")

        return path

    def _calculate_distance_matrix(self, points: list) -> np.ndarray:
        """Calculate the Euclidean distance matrix using vectorized operations."""
        num_points = len(points)
        # Convertir a array numpy con precisión correcta
        points_array = np.array(points, dtype=self.numpy_precision)
        # Usar broadcasting para cálculo vectorizado (más rápido)
        diff = points_array[:, np.newaxis, :] - points_array[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        # Caching (opcional)
        # current_points_hash = hash(tuple(map(tuple, points)))
        # if self.cached_distance_matrix is not None and self.cached_points_hash == current_points_hash:
        #    print("Using cached distance matrix.")
        #    return self.cached_distance_matrix
        # self.cached_distance_matrix = distance_matrix
        # self.cached_points_hash = current_points_hash

        return distance_matrix

    def _create_tsp_hamiltonian(
        self, distance_matrix: np.ndarray
    ) -> hamiltonians.SymbolicHamiltonian:
        """
        Create the TSP Hamiltonian using QUBO encoding with squared constraints.
        H = H_cost + penalty * H_constraints
        H_cost = Σ_{i≠k} d[i,k] * Σ_{j=0..n-1} b_{i,j} * b_{k, (j+1)%n}
        H_constraints = Σ_i (1−Σ_j b_ij)^2 + Σ_j (1−Σ_i b_ij)^2
        where b_ij = (1 - Z(i*n + j)) / 2 represents city i being at position j.
        """
        n = len(distance_matrix)  # Number of points (cities)
        num_qubits = n * n

        # Determinar el peso de la penalización si no se proporciona
        if self.penalty_weight is None:
            # Heurística: escala con la máxima distancia y el número de puntos
            max_dist = np.max(distance_matrix)
            # Usar n * max_dist como escala base parece razonable
            self.penalty_weight = float(max_dist * n * 1.5)  # Añadir factor > 1
            print(f"Auto-calculated penalty weight: {self.penalty_weight:.2f}")
        else:
            print(f"Using provided penalty weight: {self.penalty_weight:.2f}")

        # Hamiltoniano de Costo (H_cost) - Correcta formulación TSP
        H_cost = 0
        print("Building H_cost (sum d[i,k] * b_ij * b_k,j+1)...")
        # i, k son índices de ciudades
        # j es índice de posición en la ruta
        for i in range(n):
            for k in range(n):  # Sumar sobre todos los pares (i, k)
                if i == k:
                    continue  # No hay distancia de una ciudad a sí misma en este contexto

                dist = distance_matrix[i, k]
                if dist == 0:
                    continue  # Evitar añadir términos nulos

                for j in range(n):
                    pos_j = j
                    pos_next = (j + 1) % n  # Siguiente posición (cíclica)

                    qubit_index_i_j = i * n + pos_j  # Qubit para ciudad i en posición j
                    qubit_index_k_next = (
                        k * n + pos_next
                    )  # Qubit para ciudad k en posición j+1

                    # Término: dist * b_ij * b_k,next
                    # b_x = (1 - Z(x)) / 2
                    # dist * [(1 - Z(q_i_j))/2] * [(1 - Z(q_k_next))/2]
                    # = (dist / 4.0) * (1 - Z(q_i_j) - Z(q_k_next) + Z(q_i_j) * Z(q_k_next))
                    term = (dist / 4.0) * (
                        1
                        - Z(qubit_index_i_j)
                        - Z(qubit_index_k_next)
                        + Z(qubit_index_i_j) * Z(qubit_index_k_next)
                    )
                    H_cost += term

        # Hamiltoniano de Restricciones (H_cons) - Penaliza estados inválidos
        H_cons = 0
        print("Building H_cons (squared constraints)...")
        # Restricción 1: Cada ciudad aparece exactamente una vez (suma por filas = 1)
        for i in range(n):  # Para cada ciudad i
            # row_sum = Σ_j b_ij = Σ_j (1 - Z(i*n + j))/2
            row_sum_term = sum([(1 - Z(i * n + j)) / 2 for j in range(n)])
            # Penalizar si la suma NO es 1: (1 - row_sum)^2
            H_cons += (1 - row_sum_term) ** 2

        # Restricción 2: Cada posición es ocupada por exactamente una ciudad (suma por columnas = 1)
        for j in range(n):  # Para cada posición j
            # col_sum = Σ_i b_ij = Σ_i (1 - Z(i*n + j))/2
            col_sum_term = sum([(1 - Z(i * n + j)) / 2 for i in range(n)])
            # Penalizar si la suma NO es 1: (1 - col_sum)^2
            H_cons += (1 - col_sum_term) ** 2

        print("Hamiltonian terms built. Combining H_cost and H_cons...")
        # Hamiltoniano total H = H_cost + penalty * H_cons
        total_hamiltonian_symbolic = H_cost + self.penalty_weight * H_cons

        # Asegurar que es SymbolicHamiltonian
        if not isinstance(total_hamiltonian_symbolic, hamiltonians.SymbolicHamiltonian):
            # Esto podría pasar si n=0 o algo extraño. Crear H simbólico.
            final_H = hamiltonians.SymbolicHamiltonian(
                total_hamiltonian_symbolic, nqubits=num_qubits
            )
        else:
            final_H = total_hamiltonian_symbolic

        # Verificar número de qubits (a veces SymbolicHamiltonian puede inferir menos si faltan términos)
        if final_H.nqubits != num_qubits:
            print(
                f"Warning: SymbolicHamiltonian inferred {final_H.nqubits} qubits, expected {num_qubits}. Forcing qubit count."
            )
            # Forzar el número correcto de qubits si es necesario
            final_H = hamiltonians.SymbolicHamiltonian(
                final_H.formula, nqubits=num_qubits
            )

        return final_H

    def _decode_binary_state_to_path(self, state_binary: str, num_points: int) -> list:
        """Decodes the most probable binary string state into a TSP path."""
        num_qubits = num_points * num_points
        if len(state_binary) != num_qubits:
            # Pad con ceros a la izquierda si es necesario (aunque format debería hacerlo)
            state_binary = state_binary.zfill(num_qubits)
            print(
                f"Warning: Binary state length adjusted from {len(state_binary)} to {num_qubits}."
            )
            # raise ValueError(f"Binary state length {len(state_binary)} does not match required {num_qubits} qubits.")

        # Convertir bitstring a matriz (num_points x num_points)
        # matrix[i, j] = 1 si ciudad 'i' está en posición 'j'
        try:
            values = list(map(int, list(state_binary)))
            matrix = np.array(values).reshape((num_points, num_points))
        except Exception as e:
            raise RuntimeError(
                f"Failed to reshape binary state '{state_binary}' to matrix: {e}"
            )

        # Decodificar la ruta desde la matriz (asumiendo que es una matriz de permutación válida)
        # path[j] = i significa que la ciudad 'i' está en la posición 'j'
        path = [-1] * num_points
        # Usar argmax por columna para encontrar la fila (ciudad) para cada posición
        # Esto funciona incluso si la matriz no es perfectamente válida (elige una ciudad por posición)
        try:
            col_indices = np.arange(num_points)
            row_indices = np.argmax(
                matrix, axis=0
            )  # Encuentra el índice de fila del '1' (o máximo) en cada columna
            path = list(row_indices)  # path[j] = ciudad en posición j

            # Validación simple: ¿Se asignaron todas las posiciones y todas las ciudades?
            if len(set(path)) != num_points:
                print(
                    f"Warning: Decoded path {path} does not contain unique cities. The state matrix might be invalid."
                )

        except Exception as e:
            print(f"Error during path decoding from matrix: {e}")
            # Fallback o re-raise
            raise RuntimeError("Could not decode path from the state matrix.")

        # Devolver como lista de enteros estándar
        return [int(p) for p in path]

    def _is_valid_permutation_matrix(self, state_binary: str, num_points: int) -> bool:
        """Checks if the binary string represents a valid permutation matrix (one '1' per row/col)."""
        num_qubits = num_points * num_points
        if len(state_binary) != num_qubits:
            state_binary = state_binary.zfill(num_qubits)

        try:
            values = list(map(int, list(state_binary)))
            matrix = np.array(values).reshape((num_points, num_points))
            row_sums = np.sum(matrix, axis=1)
            col_sums = np.sum(matrix, axis=0)
            return np.all(row_sums == 1) and np.all(col_sums == 1)
        except Exception:
            return False  # Error al convertir o reshape
