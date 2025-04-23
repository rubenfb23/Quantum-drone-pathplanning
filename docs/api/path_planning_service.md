# PathPlanningService

Service class for quantum path-planning using Qibo.

## Class: PathPlanningService

### Initialization

```python
PathPlanningService(
    depth: int = 4,
    optimizer: str = "BFGS",
    penalty_weight: Optional[float] = None,
    precision: str = "float32",
    gate_fusion: bool = True,
    devices: Optional[List[int]] = None,
    seed: Optional[int] = None
)
```
- **depth**: Depth of the QAOA circuit (default: 4).
- **optimizer**: Classical optimizer for parameter minimization (default: "BFGS").
- **penalty_weight**: Weight for constraint penalties; if `None`, computed automatically from max distance.
- **precision**: Numeric precision for quantum computations (`"float32"` or `"float64").
- **gate_fusion**: Enable gate fusion if supported by the backend (default: `True`).
- **devices**: List of GPU device indices; if `None`, uses default device.
- **seed**: Random seed for reproducibility; if `None`, results are nondeterministic.

Initializes the Qibo backend, sets precision, configures devices, and applies the random seed.

### Methods

#### find_optimal_path(points: List[Tuple[float, float]]) -> List[int]

- **points**: List of (x, y) coordinates to visit.
- **Returns**: Sequence of point indices representing the optimal path.
- **Raises**: `ValueError` if `points` is empty.
- **Notes**: Uses the configured parameters (`depth`, `optimizer`, `penalty_weight`, `precision`, `gate_fusion`, `devices`, `seed`) to run QAOA and sample the final circuit.

Steps:
1. Validates inputs.
2. Builds Hamiltonian for TSP.
3. Runs QAOA to minimize path length.
4. Extracts and decodes bitstring into path sequence.

#### _calculate_distance_matrix(points)

`_calculate_distance_matrix(points: List[Tuple[float, float]]) -> np.ndarray`

Computes and caches pairwise Euclidean distances between points.

#### _create_tsp_hamiltonian(distance_matrix)

`_create_tsp_hamiltonian(distance_matrix: np.ndarray) -> SymbolicHamiltonian`

Constructs the TSP Hamiltonian combining cost terms and constraint penalties.

#### _decode_binary_state_to_path(state_binary, num_points)

`_decode_binary_state_to_path(state_binary: str, num_points: int) -> List[int]`

Reshapes a binary state string into a permutation matrix and extracts the TSP path sequence.

#### _is_valid_permutation_matrix(state_binary, num_points)

`_is_valid_permutation_matrix(state_binary: str, num_points: int) -> bool`

Validates that the binary string represents a valid permutation matrix (each row and column sums to 1).
