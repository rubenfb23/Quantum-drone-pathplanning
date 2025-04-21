# main.py


Module for entry-point orchestration and visualization.

## Functions

### plot_path(points: List[Tuple[float, float]], path: List[int]) -> None
Visualizes a sequence of 2D points and the computed tour.

- **points**: Coordinates to plot.
- **path**: Ordered indices defining the tour.

### main() -> None
Executes example workflow:
1. Defines sample points.
2. Instantiates `PathPlanningService`.
3. Computes optimal path.
4. Calls `plot_path` to render the result.
