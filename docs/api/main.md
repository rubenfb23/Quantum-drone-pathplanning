# main.py


Module for entry-point orchestration and visualization.

## Functions

### load_points_from_csv(csv_file: str) -> List[Tuple[float, float]]
Loads point coordinates from a CSV file with headers 'x' and 'y'.

- **csv_file**: Path to the CSV file.
- **Returns**: List of (x, y) tuples.

### plot_path(points: List[Tuple[float, float]], path: List[int]) -> None
Visualizes a sequence of 2D points and the computed tour.

- **points**: Coordinates to plot.
- **path**: Ordered indices defining the tour.

### main() -> None
Executes example workflow:
1. Loads points from `points.csv` using `load_points_from_csv`.
2. Instantiates `PathPlanningService` with configurable depth, optimizer, and shots.
3. Computes optimal path via QAOA.
4. Calls `plot_path` to render the result.
