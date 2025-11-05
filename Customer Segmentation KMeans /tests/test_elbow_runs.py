# 8.1 tests/test_elbow_runs.py

import os
import sys
import subprocess

def test_elbow_runs():
    """
    Smoke test: script should run and produce elbow and silhouette plots.
    """
    cmd = [sys.executable, "scripts/run_elbow.py"]
    subprocess.run(cmd, check=True)
    assert os.path.exists("elbow_plot.png")
    assert os.path.exists("silhouette_plot.png")
