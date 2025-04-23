"""
Service module for quantum path-planning logic using QAOA for TSP.
Encapsulates the core logic, backend setup, Hamiltonian creation,
optimization, and result decoding directly from the state vector.
Includes a progress bar via tqdm.
"""

import time
import threading
import numpy as np

# Use monkey patch for tqdm internals
import tqdm as tqdm_module

# tqdm class for progress bars
from tqdm import tqdm
import warnings

from qibo import hamiltonians, models, set_backend, set_precision, get_backend
from qibo.symbols import Z
from typing import List, Tuple, Optional

from .single_tqdm import SingleTqdm
from .distance_calculator import DistanceCalculator
from .hamiltonian_builder import HamiltonianBuilder
from .qaoa_orchestrator import QaoaOrchestrator

import logging
from contextlib import contextmanager

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="scipy.optimize._numdiff"
)


class PathPlanningError(Exception):
    """Custom exception for path planning errors."""

    pass


# Configure module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PathPlanningService:
    """Service class for quantum path-planning using Qibo."""

    TOTAL_STAGES: int = 5

    def __init__(
        self,
        depth: int = 4,
        optimizer: str = "BFGS",
        penalty_weight: float = 1.0,
        precision: str = "float32",
        gate_fusion: bool = True,
        devices: Optional[list] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize service with QAOA parameters and reproducibility settings."""
        self.depth = depth
        self.optimizer = optimizer
        self.penalty_weight = penalty_weight
        self.user_precision_str = precision
        self.gate_fusion = gate_fusion
        self.seed = seed
        try:
            if self.seed is not None:
                np.random.seed(self.seed)
                logger.info(f"NumPy random seed set to {self.seed}.")
                # seed Qibo backend for reproducibility
                get_backend().set_seed(self.seed)
                logger.info(f"Qibo backend seed set to {self.seed}.")
        except Exception as e:
            logger.warning(f"Failed to set seed: {e}")
        # Set precision strings and corresponding numpy dtype
        if precision == "float32":
            self.qibo_precision_str = "single"
            self.numpy_real_dtype = np.float32
        else:
            self.qibo_precision_str = "double"
            self.numpy_real_dtype = np.float64

        # Initialize distance calculator
        self.distance_calculator = DistanceCalculator()

        # Initialize Hamiltonian builder with penalty tuner
        self.hamiltonian_builder = HamiltonianBuilder(self._tune_penalty_weight)

        # Initialize QAOA orchestrator
        self.qaoa_orchestrator = QaoaOrchestrator(
            self.depth, self.optimizer, self.numpy_real_dtype
        )

        self._configure_backend(devices)

    def _configure_backend(self, devices: Optional[list]):
        """Configures Qibo backend and precision."""
        try:
            print(
                "Attempting to set Qibo backend to 'qibojit' "
                "with platform='cuquantum'..."
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

    @contextmanager
    def _patched_tqdm(self):
        """Context manager to temporarily monkey-patch tqdm for consistent progress updates."""
        original = tqdm_module.tqdm
        tqdm_module.tqdm = SingleTqdm
        try:
            yield
        finally:
            tqdm_module.tqdm = original

    def _update_pbar(self, pbar: tqdm, stop_event: threading.Event) -> None:
        """Continuously refresh the progress bar until stop_event is set."""
        while not stop_event.is_set():
            pbar.refresh()
            time.sleep(0.1)

    def find_optimal_path(self, points: List[Tuple[float, float]]) -> List[int]:
        """Public API: find the optimal path for given points."""
        if not points:
            logger.error("Input 'points' list cannot be empty.")
            raise PathPlanningError("Input 'points' list cannot be empty.")
        num_points = len(points)
        if num_points < 2:
            print("Warning: Path for < 2 points is trivial.")
            return list(range(num_points))

        print(f"\nStarting TSP optimization for {num_points} points.")
        start_time = time.time()
        total_stages = self.TOTAL_STAGES

        try:
            with tqdm(total=total_stages, desc="TSP Optimization Progress") as pbar:
                stop_event = threading.Event()
                updater_thread = threading.Thread(
                    target=self._update_pbar, args=(pbar, stop_event)
                )
                updater_thread.start()

                try:
                    distance_matrix = self._stage_calculate_distances(points, pbar)
                    hamiltonian = self._stage_build_hamiltonian(distance_matrix, pbar)
                    num_qubits = hamiltonian.nqubits
                    qaoa, initial_parameters = self._stage_initialize_qaoa(
                        hamiltonian, pbar
                    )
                    with self._patched_tqdm():
                        best_params = self._stage_optimize_qaoa(
                            qaoa, initial_parameters, pbar
                        )
                    with self._patched_tqdm():
                        path = self._stage_execute_and_decode(
                            qaoa, best_params, num_qubits, num_points, pbar
                        )
                finally:
                    stop_event.set()
                    updater_thread.join()
        except Exception as e:
            logger.error(f"Path planning failed: {e}")
            raise PathPlanningError("TSP optimization error.") from e

        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")
        return path

    def _stage_calculate_distances(
        self, points: List[Tuple[float, float]], pbar: tqdm
    ) -> np.ndarray:
        pbar.set_description(f"Stage 1/{self.TOTAL_STAGES}: Calculating distances")
        matrix = self._calculate_distance_matrix(points)
        pbar.update(1)
        return matrix

    def _stage_build_hamiltonian(
        self, distance_matrix: np.ndarray, pbar: tqdm
    ) -> hamiltonians.SymbolicHamiltonian:
        pbar.set_description(f"Stage 2/{self.TOTAL_STAGES}: Building Hamiltonian")
        hamiltonian = self._create_tsp_hamiltonian(distance_matrix)
        pbar.update(1)
        return hamiltonian

    def _stage_initialize_qaoa(
        self, hamiltonian: hamiltonians.SymbolicHamiltonian, pbar: tqdm
    ) -> Tuple[models.QAOA, np.ndarray]:
        pbar.set_description(f"Stage 3/{self.TOTAL_STAGES}: Initializing QAOA")
        qaoa, params = self.qaoa_orchestrator.initialize(hamiltonian)
        pbar.update(1)
        return qaoa, params

    def _stage_optimize_qaoa(
        self, qaoa: models.QAOA, params: np.ndarray, pbar: tqdm
    ) -> np.ndarray:
        pbar.set_description(
            f"Stage 4/{self.TOTAL_STAGES}: Optimizing ({self.optimizer})"
        )
        # Temporarily patch tqdm for optimizer progress
        original = tqdm_module.tqdm
        tqdm_module.tqdm = SingleTqdm
        try:
            best_params = self.qaoa_orchestrator.optimize(qaoa, params)
            return best_params
        finally:
            tqdm_module.tqdm = original
            pbar.update(1)

    def _stage_execute_and_decode(
        self,
        qaoa: models.QAOA,
        best_params: np.ndarray,
        num_qubits: int,
        num_points: int,
        pbar: tqdm,
    ) -> List[int]:
        pbar.set_description(f"Stage 5/{self.TOTAL_STAGES}: Final execution & decoding")
        original = tqdm_module.tqdm
        tqdm_module.tqdm = SingleTqdm
        try:
            vec = self.qaoa_orchestrator.execute(qaoa, best_params)
            probs = np.abs(vec) ** 2
            indices = np.argsort(probs)[::-1]
            for idx in indices:
                if probs[idx] < 1e-6:
                    break
                bin_str = format(idx, f"0{num_qubits}b")
                if self._is_valid_permutation_matrix(bin_str, num_points):
                    return self._decode_binary_state_to_path(bin_str, num_points)
            most = int(np.argmax(probs))
            print("\nWarning: No valid TSP state found among highly probable states.")
            print(
                f"Reporting path from overall most probable state (index {most}, prob {probs[most]:.4f}), which is likely invalid."
            )
            return self._decode_binary_state_to_path(
                format(most, f"0{num_qubits}b"), num_points
            )
        except Exception as e:
            print(f"\nError during state vector decoding: {e}")
            raise RuntimeError("Failed to decode the final state vector.") from e
        finally:
            tqdm_module.tqdm = original
            pbar.update(1)

    def _calculate_distance_matrix(self, points: list) -> np.ndarray:
        """
        Delegates distance matrix computation and caching to `DistanceCalculator`.
        """
        return self.distance_calculator.calculate(points, self.numpy_real_dtype)

    def _tune_penalty_weight(self, distance_matrix: np.ndarray) -> float:
        """Stub for penalty weight tuning. Override or extend to customize weight selection. Returns float penalty weight."""
        # placeholder: return provided weight or heuristic based on distances
        max_dist = np.max(distance_matrix) if distance_matrix.size else 1.0
        return max_dist * (distance_matrix.shape[0] ** 2)

    def _create_tsp_hamiltonian(
        self, distance_matrix: np.ndarray
    ) -> hamiltonians.SymbolicHamiltonian:
        """Delegates Hamiltonian construction to `HamiltonianBuilder`."""
        return self.hamiltonian_builder.build(distance_matrix)

    def _decode_binary_state_to_path(
        self, state_binary: str, num_points: int
    ) -> List[int]:
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
            return False
