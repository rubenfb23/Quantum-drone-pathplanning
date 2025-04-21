# Quantum Drone Path-Planning with Qibo

This project demonstrates a quantum algorithm for drone path-planning using Qibo (Qiskit 2.0 compatible). The core logic is implemented in Python and leverages quantum optimization techniques to solve a simplified version of the Traveling Salesman Problem (TSP).

## Project Structure

- `main.py`: Entry point. Sets up the problem, runs the quantum algorithm, and visualizes the result.
- `services/path_planning_service.py`: Contains the `PathPlanningService` class, which encapsulates the quantum optimization logic.

## How It Works

1. **Define Points**: Specify a list of (x, y) coordinates representing locations the drone must visit.
2. **Quantum Optimization**: The service encodes the TSP as a quantum Hamiltonian and uses QAOA (Quantum Approximate Optimization Algorithm) to find an optimal path.
3. **Visualization**: The resulting path is plotted on a 2D map.

## Usage

1. Install dependencies:
   ```bash
   pip install qibo matplotlib
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

3. You can customize the QAOA depth and optimizer in `main.py`:
   ```python
   service = PathPlanningService(depth=2, optimizer="BFGS")
   ```

## Customization
- **Input Points**: Edit the `points` list in `main.py` to change the locations.
- **QAOA Parameters**: Adjust `depth` and `optimizer` when creating the service.

## Clean Code Improvements
- Clear and descriptive names for classes, methods, and variables.
- Comprehensive docstrings for all public and private methods.
- Input validation and error handling throughout the code.
- Configurable QAOA parameters (depth, optimizer).
- Separation of concerns: quantum logic in the service, visualization in `main.py`.
- Example usage and visualization included.

## Limitations
- The TSP Hamiltonian is a simplified placeholder. For real-world use, a full QUBO/Ising encoding is needed.
- The decoding from quantum output to path is not fully implemented (returns a trivial path).

## Example Output
A matplotlib window will display the points and the computed path.

---

For further development, implement a full TSP Hamiltonian and decoding logic in `services/path_planning_service.py`.

## License

This project is licensed under the MIT License.