"""
Entry point for the quantum drone path-planning project.
This module handles CSV-loading, plotting, and high-level orchestration.
"""

import csv
import os
from typing import List, Tuple
import sys
from pathlib import Path

# Ensure project root is on sys.path for module imports when running as script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import CubicSpline
import numpy as np
import argparse

from src.services.path_planning_service import PathPlanningService

# Monkey-patch CuQuantumBackend to avoid AttributeError in destructor
try:
    import qibojit.backends.gpu as _qj_gpu

    def _safe_del(self):
        if hasattr(self, "handle"):
            try:
                self.cusv.destroy(self.handle)
            except Exception:
                pass

    _qj_gpu.CuQuantumBackend.__del__ = _safe_del
except ImportError:
    pass

# Default CSV filename
DEFAULT_CSV_FILENAME = "points.csv"


def parse_arguments():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Quantum Drone Path Planning")
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV_FILENAME,
        help="Path to CSV file with 'x' and 'y' columns (default: points.csv)",
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Only plot points without optimization"
    )
    return parser.parse_args()


def load_points_from_csv(csv_file: str) -> List[Tuple[float, float]]:
    """Load point coordinates from a CSV file with headers 'x' and 'y'."""
    points = []
    with open(csv_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Ensure data are loaded as floats.
            points.append((float(row["x"]), float(row["y"])))
    return points


def load_and_validate_points(csv_filename: str) -> List[Tuple[float, float]]:
    """
    Load and validate point coordinates from a CSV file.
    Raises ValueError if invalid.
    """
    csv_path = Path(__file__).parent / csv_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    points = load_points_from_csv(str(csv_path))
    if not points:
        raise ValueError("No points found in CSV.")
    if len(points) < 2:
        print("Warning: TSP requires at least 2 points.")
    if len(points) > 10:
        print(
            f"Warning: TSP with {len(points)} points ({len(points)**2} qubits) may be slow."
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
    # Save animation GIF to project root for host visibility
    output_file = project_root.joinpath("drone_path_plot.gif")
    try:
        anim.save(str(output_file), writer=PillowWriter(fps=20))
        print(f"Animation saved to {output_file}")
    except Exception as e:
        print(f"Warning: could not save animation gif: {e}")
    # Display plot (no-op under non-interactive backends)
    plt.show()


def main():
    """CLI entry point."""
    args = parse_arguments()
    try:
        points = load_and_validate_points(args.csv)
    except (FileNotFoundError, ValueError) as e:
        print(f"Input Error: {e}")
        sys.exit(1)

    if args.plot_only:
        plot_path(points, list(range(len(points))))
    else:
        # QAOA parameters
        QAOA_DEPTH = 4
        OPTIMIZER = "BFGS"
        PRECISION = "float32"
        GATE_FUSION = True
        DEVICES = None
        SEED = 42

        service = PathPlanningService(
            depth=QAOA_DEPTH,
            optimizer=OPTIMIZER,
            precision=PRECISION,
            gate_fusion=GATE_FUSION,
            devices=DEVICES,
            seed=SEED,
        )
        try:
            path = service.find_optimal_path(points)
            path_int = [int(p) for p in path]
            print(f"Optimal path: {path_int}")
            plot_path(points, path_int)
        except Exception as e:
            print(f"Runtime Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
