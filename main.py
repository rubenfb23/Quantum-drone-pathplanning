# main.py

"""
Entry point for the quantum drone path-planning project.
This file initializes the quantum algorithm and
handles high-level orchestration.
"""

import matplotlib.pyplot as plt
from services.path_planning_service import PathPlanningService
from typing import List, Tuple


def plot_path(points: List[Tuple[float, float]], path: List[int]) -> None:
    """Visualize the path on a 2D map."""
    if not points or not path:
        raise ValueError("Both 'points' and 'path' must be non-empty.")

    x, y = zip(*points)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, "bo-", label="Points")

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
    points = [
        (0, 0),
        (1, 5),
        (2, 3),
        (4, 1),
    ]  # Example points
    service = PathPlanningService(depth=2, optimizer="BFGS")
    try:
        path = service.find_optimal_path(points)
        plot_path(points, path)
    except Exception as e:
        print(f"Error during path planning: {e}")


if __name__ == "__main__":
    main()
