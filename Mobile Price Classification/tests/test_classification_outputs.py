# tests/test_classification_outputs.py

import os
import subprocess
import sys

def test_classification_confusion_plots(tmp_path):
    """
    This test runs the classification script and verifies that
    the confusion matrix images for KNN and RandomForest are
    created and non-empty.
    """

    # Run the classification script
    cmd = [sys.executable, "mobile_classification.py"]
    subprocess.run(cmd, check=True, cwd=".")

    # Expected confusion matrix plots
    expected = ["confusion_knn.png", "confusion_rf.png"]

    # Verify existence and content
    for fname in expected:
        assert os.path.exists(fname), f"Expected confusion matrix not found: {fname}"
        assert os.path.getsize(fname) > 0, f"Confusion matrix appears empty: {fname}"
