# PathPlanningService

Service class for quantum path-planning using Qibo.

## Class: PathPlanningService

### Initialization

```python
PathPlanningService(depth: int = 2, optimizer: str = "BFGS", shots: int = 100)
```
- **depth**: Depth of the QAOA circuit (default: 2).
- **optimizer**: Classical optimizer for parameter minimization (default: "BFGS").
- **shots**: Number of circuit executions for sampling (default: 100).

Initializes the Qibo backend optimized for CPU execution.

### Methods

#### find_optimal_path(points)

`find_optimal_path(points: List[Tuple[float, float]]) -> List[int]`

- **points**: List of (x, y) coordinates to visit.
- **Returns**: Sequence of point indices representing the optimal path.
- **Raises**: `ValueError` if `points` is empty.
- **Notes**: Uses the configured `depth`, `optimizer`, and `shots` parameters to run QAOA and sample the final circuit.

Steps:
1. Validates inputs.
2. Builds Hamiltonian for TSP.
3. Runs QAOA to minimize path length.
4. Extracts and decodes bitstring into path sequence.

#### _calculate_distance_matrix(points)

`_calculate_distance_matrix(points: List[Tuple[float, float]]) -> np.ndarray`

Computes pairwise Euclidean distances between points.

#### _create_tsp_hamiltonian(distance_matrix)

`_create_tsp_hamiltonian(distance_matrix: np.ndarray) -> SymbolicHamiltonian`

Constructs the TSP Hamiltonian with cost and constraint terms using Qibo's symbolics.

#### _decode_result(result, points, num_qubits)

`_decode_result(result, points: List[Tuple[float, float]], num_qubits: int) -> List[int]`

Processes Qibo execution result to retrieve the most probable state and map it back to a TSP path.
