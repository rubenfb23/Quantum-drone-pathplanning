# main.py

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
import numpy as np  # Añadido para plotting


def load_points_from_csv(csv_file: str) -> List[Tuple[float, float]]:
    """Load point coordinates from a CSV file with headers 'x' and 'y'."""
    points = []
    with open(csv_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Asegurarse de que se cargan como floats
            points.append((float(row["x"]), float(row["y"])))
    return points


def plot_path(points: List[Tuple[float, float]], path: List[int]) -> None:
    """Visualize the path on a 2D map, connecting points in order and returning to start."""
    if not points or not path:
        raise ValueError("Both 'points' and 'path' must be non-empty.")
    if len(path) != len(points):
        print(
            f"Warning: Path length ({len(path)}) doesn't match number of points ({len(points)}). Plot might be incorrect."
        )
        # Decide how to handle, e.g., raise error or plot partial path.
        # For now, we proceed but it likely indicates an issue.

    # Convertir puntos a numpy array para facilitar indexación
    points_np = np.array(points)

    plt.figure(figsize=(10, 8))  # Aumentado tamaño para mejor visualización

    # Graficar puntos con etiquetas numéricas
    plt.scatter(
        points_np[:, 0], points_np[:, 1], c="blue", label="Points", s=50, zorder=5
    )
    for i, txt in enumerate(range(len(points))):
        plt.annotate(
            txt, (points_np[i, 0] + 0.1, points_np[i, 1] + 0.1)
        )  # Pequeño offset para legibilidad

    # Graficar la ruta conectando los puntos en el orden del 'path'
    # Asegurarse de que path contiene índices válidos y usar points_np
    # Filtrar índices inválidos si es necesario, aunque idealmente path debería ser correcto.
    valid_path_indices = [p for p in path if 0 <= p < len(points)]
    if len(valid_path_indices) != len(path):
        print("Warning: Path contains invalid indices.")
        path = valid_path_indices  # Use only valid indices

    if not path:  # Si no quedan puntos válidos en la ruta
        print("Cannot plot path: no valid points in the path list.")
        plt.title(f"Drone Path Planning (TSP) - {len(points)} points - NO VALID PATH")
        plt.show()
        return

    path_points_x = points_np[path, 0]
    path_points_y = points_np[path, 1]

    # Añadir el regreso al punto inicial para cerrar el ciclo TSP
    path_points_x = np.append(path_points_x, points_np[path[0], 0])
    path_points_y = np.append(path_points_y, points_np[path[0], 1])

    plt.plot(
        path_points_x, path_points_y, "r-", label="Optimal Path (QAOA)", linewidth=1.5
    )

    plt.legend()
    plt.title(f"Drone Path Planning (TSP) - {len(points)} points")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axis("equal")  # Asegura que las escalas X e Y sean iguales
    plt.show()


def main():
    """Main function to execute the path-planning service."""
    # --- Configuración ---
    QAOA_DEPTH = 4  # Recomendación: Aumentar profundidad (e.g., 4-6) para problemas más complejos
    OPTIMIZER = "BFGS"
    PRECISION = "float32"  # Recomendación: Usar precisión simple ("float32") para GPU
    GATE_FUSION = (
        True  # Recomendación: Activar fusión de puertas para backends que lo soporten
    )
    # DEVICES = [0, 1] # Opcional: Descomentar y ajustar si tienes múltiples GPUs
    DEVICES = None  # Usar GPU por defecto (o CPU si falla la configuración)
    # --- Fin Configuración ---

    # Cargar puntos desde CSV
    try:
        script_dir = os.path.dirname(__file__)  # Directorio del script actual
        # Asegúrate de que points.csv esté en el mismo directorio o ajusta la ruta
        csv_path = os.path.join(script_dir, "points.csv")
        points = load_points_from_csv(csv_path)
        num_points = len(points)
        print(f"Loaded {num_points} points from {csv_path}")
        if num_points == 0:
            print("Error: No points found in points.csv.")
            return
        if num_points > 10:  # n^2 qubits, >100 qubits puede ser lento
            print(
                f"Warning: TSP with {num_points} points ({num_points**2} qubits) can be computationally intensive."
            )
        if num_points < 2:
            print("Warning: TSP requires at least 2 points.")
            # Podríamos manejar el caso trivial aquí o dejarlo al servicio
            # plot_path(points, list(range(num_points))) # Graficar solo puntos
            # return

    except FileNotFoundError:
        print(f"Error: points.csv not found at {csv_path}")
        print(
            "Please create points.csv with 'x' and 'y' columns in the same directory."
        )
        return
    except Exception as e:
        print(f"Error loading points: {e}")
        return

    # Inicializar el servicio con parámetros optimizados
    service = PathPlanningService(
        depth=QAOA_DEPTH,
        optimizer=OPTIMIZER,
        precision=PRECISION,
        gate_fusion=GATE_FUSION,  # Pasamos el flag (aunque su efecto depende del backend)
        devices=DEVICES,
    )

    try:
        print(
            f"\nFinding optimal path for {num_points} points using QAOA (depth={QAOA_DEPTH})..."
        )
        # Obtener el camino óptimo
        path = service.find_optimal_path(points)

        # Convertir cada elemento a entero estándar para impresión legible y plotting
        # (El servicio ya debería devolver enteros, pero aseguramos)
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

        traceback.print_exc()  # Imprime el stack trace completo para depuración


if __name__ == "__main__":
    main()
