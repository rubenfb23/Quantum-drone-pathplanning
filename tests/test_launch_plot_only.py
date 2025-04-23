import os
import sys
import subprocess
import pytest


def test_main_plot_only(monkeypatch, tmp_path):
    # Use non-interactive Agg backend to prevent GUI requirement
    monkeypatch.setenv("MPLBACKEND", "Agg")
    script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    )
    # Ensure src package is on PYTHONPATH for imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    monkeypatch.setenv("PYTHONPATH", project_root)
    # Run the script with --plot-only flag
    result = subprocess.run(
        [sys.executable, script, "--plot-only"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Script exited with {result.returncode}\n"
        f"STDOUT:\n{result.stdout.decode()}\n"
        f"STDERR:\n{result.stderr.decode()}"
    )
