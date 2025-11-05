# tests/test_visualization_outputs.py

import os
import subprocess
import sys

def test_visualization_outputs(tmp_path):
    """
    This test runs the visualization script and checks that
    all four expected plot images are generated and non-empty.
    """

    # Run the visualization script from the repo root
    cmd = [sys.executable, "mobile_visualization.py"]
    subprocess.run(cmd, check=True, cwd=".")

    # Expected output images
    expected = [
        "plot_battery_vs_memory.png",
        "plot_frontcam_vs_bluetooth.png",
        "plot_memory_vs_frontcam.png",
        "plot_3d_battery_blue_frontcam.png",
    ]

    # Validate that each file exists and is not empty
    for fname in expected:
        assert os.path.exists(fname), f"Expected plot not found: {fname}"
        assert os.path.getsize(fname) > 0, f"Plot file appears empty: {fname}"
