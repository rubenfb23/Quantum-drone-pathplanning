# Quantum Drone Path-Planning

This project develops a quantum algorithm for drone path-planning using Qibo.

## Setup

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
2. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- Python 3.9+
- Qibo
- NumPy
- Matplotlib

## Usage

Start developing your quantum algorithm by running the main script:
```bash
python main.py
```

The project uses Qibo's implementation of QAOA (Quantum Approximate Optimization Algorithm) to solve the Traveling Salesperson Problem for drone path planning.

## Project Structure

- `main.py`: Entry point that initializes the path planning service and visualizes results
- `services/path_planning_service.py`: Core quantum algorithm implementation using Qibo

## Key Features

- Quantum solution for path optimization problems
- Visualization of calculated drone paths
- QAOA implementation for combinatorial optimization

## License

This project is licensed under the MIT License.