# main.py

"""
Entry point for the quantum drone path-planning project.
This file initializes the quantum algorithm and handles high-level orchestration.
"""

from services.path_planning_service import PathPlanningService

def main():
    """Main function to execute the quantum path-planning algorithm."""
    service = PathPlanningService()
    service.execute()

if __name__ == "__main__":
    main()