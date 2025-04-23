import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
from src.main import plot_path


def test_plot_path(monkeypatch):
    points = [(0, 0), (3, 1), (2, 2), (5, 7)]
    path = [0, 2, 1, 3]
    called = []

    def fake_show():
        called.append(True)

    monkeypatch.setattr(plt, "show", fake_show)
    plot_path(points, path)
    assert called, "plt.show was not called"
