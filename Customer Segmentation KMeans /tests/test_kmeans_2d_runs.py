# 8.2 tests/test_kmeans_2d_runs.py

import os
import sys
import subprocess

def test_kmeans_2d_runs():
    """
    Smoke test: script should run and produce 2D cluster plot.
    """
    cmd = [sys.executable, "scripts/run_kmeans_2d.py"]
    subprocess.run(cmd, check=True)
    assert os.path.exists("clusters_2d.png")
