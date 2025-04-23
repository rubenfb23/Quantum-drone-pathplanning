"""
Entry point for the quantum drone path-planning project.
This module handles CSV-loading, plotting, and high-level orchestration.
"""

import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import CubicSpline
import numpy as np
import argparse

from src.services.path_planning_service import PathPlanningService


def load_points_from_csv(csv_file: str) -> List[Tuple[float, float]]:
    """Load point coordinates from a CSV file with headers 'x' and 'y'."""
    points = []
    with open(csv_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Ensure data are loaded as floats.
            points.append((float(row["x"]), float(row["y"])))
    return points


def _load_and_validate_points(csv_filename: str) -> List[Tuple[float, float]]:
    """Load point coordinates from CSV and validate for TSP requirements."""
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, csv_filename)
    points = load_points_from_csv(csv_path)
    if not points:
        print(f"Error: No points found in {csv_path}.")
        raise ValueError("No points found.")
    num_points = len(points)
    if num_points < 2:
        print("Warning: TSP requires at least 2 points.")
    if num_points > 10:
        print(
            f"Warning: TSP with {num_points} points "
            f"({num_points**2} qubits) "
            "can be computationally intensive."
        )
    return points


def plot_path(points: List[Tuple[float, float]], path: List[int]) -> None:
    """
    Visualize the path on a 2D map, connecting points in order
    and returning to start with animation.
    """
    if not points or not path:
        raise ValueError("Both 'points' and 'path' must be non-empty.")

    if len(path) != len(points):
        print(
            f"Warning: Path length ({len(path)}) doesn't match number of "
            f"points ({len(points)}). "
            f"Plot might be incorrect."
        )

    points_np = np.array(points)
    plt.figure(figsize=(10, 8))
    plt.scatter(
        points_np[:, 0],
        points_np[:, 1],
        c="blue",
        label="Points",
        s=50,
        zorder=5,
    )

    for i in range(len(points)):
        plt.annotate(str(i), (points_np[i, 0] + 0.1, points_np[i, 1] + 0.1))

    valid_path_indices = [p for p in path if 0 <= p < len(points)]
    if len(valid_path_indices) != len(path):
        print("Warning: Path contains invalid indices.")
        path = valid_path_indices

    if not path:
        print("Cannot plot path: no valid points in the path list.")
        title_text = (
            "Drone Path Planning (TSP) - " f"{len(points)} points - NO VALID PATH"
        )
        plt.title(title_text)
        plt.show()
        return

    path_points_x = points_np[path, 0]
    path_points_y = points_np[path, 1]

    # Close the TSP loop by returning to the starting point.
    path_points_x = np.append(path_points_x, points_np[path[0], 0])
    path_points_y = np.append(path_points_y, points_np[path[0], 1])

    # Fit cubic spline and animate drone
    t = np.arange(len(path_points_x))
    cs_x = CubicSpline(t, path_points_x, bc_type="periodic")
    cs_y = CubicSpline(t, path_points_y, bc_type="periodic")
    t_new = np.linspace(t[0], t[-1], 200)
    xs_smooth = cs_x(t_new)
    ys_smooth = cs_y(t_new)
    (curve_line,) = plt.plot(
        xs_smooth,
        ys_smooth,
        "r-",
        label="Optimal Path (Cubic)",
        linewidth=1.5,
    )

    # Initialize drone marker
    (drone,) = plt.plot([], [], "ro", markersize=8, label="Drone")

    def init():
        drone.set_data([], [])
        return (drone,)

    def animate(i):
        # Animate drone marker with single-point sequences
        x = xs_smooth[i]
        y = ys_smooth[i]
        drone.set_data([x], [y])
        return (drone,)

    anim = FuncAnimation(
        plt.gcf(),
        animate,
        frames=len(xs_smooth),
        init_func=init,
        interval=50,
        blit=True,
    )

    plt.legend()
    plt.title(f"Drone Path Planning (TSP) - {len(points)} points")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axis("equal")
    # Save animation to GIF for visibility during headless or test runs
    output_file = os.path.join(os.getcwd(), "drone_path_plot.gif")
    try:
        anim.save(output_file, writer=PillowWriter(fps=20))
        print(f"Animation saved to {output_file}")
    except Exception as e:
        print(f"Warning: could not save animation gif: {e}")
    # Display plot (no-op under non-interactive backends)
    plt.show()


def main():
    """Main function to execute the path-planning service."""
    parser = argparse.ArgumentParser(description="Quantum Drone Path Planning")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Only plot the points from the CSV " "without running the quantum algorithm"
        ),
    )
    args = parser.parse_args()

    QAOA_DEPTH = 4  # Recommended: Increase depth for more complex problems.
    OPTIMIZER = "BFGS"
    PRECISION = "float32"  # Use single precision for GPU.
    GATE_FUSION = True  # Enable gate fusion if supported by the backend.
    DEVICES = None  # Use default GPU (or CPU if configuration fails).

    # Reproducibility seed
    SEED = 42  # set to desired integer for consistent runs

    try:
        points = _load_and_validate_points("points.csv")
        num_points = len(points)
        print(f"Loaded {num_points} points.")
    except FileNotFoundError:
        print("Error: points.csv not found at expected location.")
        print(
            (
                "Please create points.csv with 'x' and 'y' "
                "columns in the same directory."
            )
        )
        return
    except ValueError as ve:
        print(f"Input Error: {ve}")
        return

    if args.plot_only:
        plot_path(points, list(range(len(points))))
        return

    service = PathPlanningService(
        depth=QAOA_DEPTH,
        optimizer=OPTIMIZER,
        precision=PRECISION,
        gate_fusion=GATE_FUSION,
        devices=DEVICES,
        seed=SEED,
    )

    try:
        print(
            "\nFinding optimal path for %d points "
            "using QAOA (depth=%d)..." % (num_points, QAOA_DEPTH)
        )
        path = service.find_optimal_path(points)
        path_int = [int(p) for p in path]
        print(f"\nOptimal path found: {path_int}")
        plot_path(points, path_int)
    except ValueError as ve:
        print(f"\nInput Error: {ve}")
    except RuntimeError as re:
        print(f"\nRuntime Error (possibly Qibo/backend related): {re}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during path planning: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
