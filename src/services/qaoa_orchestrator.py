from typing import Tuple
import numpy as np
from qibo import models


class QaoaOrchestrator:
    """
    Handles QAOA initialization, parameter optimization, and state execution.
    """

    def __init__(self, depth: int, optimizer: str, dtype: np.dtype):
        self.depth = depth
        self.optimizer = optimizer
        self.dtype = dtype

    def initialize(self, hamiltonian) -> Tuple[models.QAOA, np.ndarray]:
        """
        Creates a QAOA model and generates initial random parameters.
        """
        qaoa = models.QAOA(hamiltonian)
        initial_params = np.random.uniform(0, 0.1, 2 * self.depth).astype(self.dtype)
        return qaoa, initial_params

    def optimize(self, qaoa, initial_params: np.ndarray) -> np.ndarray:
        """
        Minimizes the QAOA cost function to find optimal parameters.
        Returns the best parameters found.
        """
        _, best_params, _ = qaoa.minimize(
            initial_params, method=self.optimizer, options={"disp": False}
        )
        return best_params

    def execute(self, qaoa, best_params: np.ndarray) -> np.ndarray:
        """
        Executes the QAOA circuit with best parameters and returns the state vector.
        """
        qaoa.set_parameters(best_params)
        result = qaoa.execute()
        return result.get() if hasattr(result, "get") else result
