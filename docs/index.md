# Quantum Drone Path Planning Documentation

This documentation provides details on the quantum algorithm for drone path-planning using Qibo.

## Overview

The project implements a QAOA-based solution to the Traveling Salesman Problem (TSP) for drone route optimization.

## Getting Started

- Install dependencies: `pip install -r requirements.txt`
- Ensure you have a CUDA-compatible GPU and the CUDA toolkit installed
- Prepare input points:
  - Edit `points.csv` with `x,y` headers or supply your own CSV file.
- Run the main script:
  ```bash
  python main.py
  ```

## API Reference

- [main.py](api/main.md): Entry point functions for loading points, plotting, and running the service.
- [PathPlanningService](api/path_planning_service.md): Core service for QAOA-based path planning.
