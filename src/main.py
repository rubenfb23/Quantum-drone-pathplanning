"""
Entry point for the quantum drone path-planning project.

This file initializes the quantum algorithm and
handles high-level orchestration.
"""

import matplotlib.pyplot as plt
from services.path_planning_service import PathPlanningService
from typing import List, Tuple
import csv
import os


def load_points_from_csv(csv_file: str) -> List[Tuple[float, float]]:
    """Load point coordinates from a CSV file with headers 'x' and 'y'."""
    points = []
    with open(csv_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            points.append((float(row["x"]), float(row["y"])))
    return points


def plot_path(points: List[Tuple[float, float]], path: List[int]) -> None:
    """Visualize the path on a 2D map."""
    if not points or not path:
        raise ValueError("Both 'points' and 'path' must be non-empty.")
    x, y = zip(*points)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, "bo", label="Points")
    for i, txt in enumerate(range(len(points))):
        plt.annotate(txt, (x[i], y[i]))
    plt.plot([x[i] for i in path], [y[i] for i in path], "r-", label="Path")
    plt.legend()
    plt.title("Drone Path Planning")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.show()


def main():
    """Main function to execute the path-planning service."""
    # Load points from CSV
    csv_path = os.path.join(os.path.dirname(__file__), "points.csv")
    points = load_points_from_csv(csv_path)

    service = PathPlanningService(depth=2, optimizer="BFGS")

    try:
        # Obtener el camino óptimo
        path = service.find_optimal_path(points)
        # Convertir cada elemento a entero estándar
        # para impresión legible
        path = [int(p) for p in path]
        print(f"Received path: {path}")
        plot_path(points, path)
    except Exception as e:
        print(f"Error during path planning: {e}")


if __name__ == "__main__":
    main()
