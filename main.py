# main.py

"""
Entry point for the quantum drone path-planning project.
This file initializes the quantum algorithm and handles high-level orchestration.
"""

import matplotlib.pyplot as plt
from services.path_planning_service import PathPlanningService


def plot_path(points, path):
    """Visualize the path on a 2D map."""
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
    points = [(0, 0), (1, 2), (2, 4)]  # Reduced to 3 points for smaller problem size
    service = PathPlanningService()
    path = service.execute(points)  # Use service result as the path
    plot_path(points, path)


if __name__ == "__main__":
    main()
