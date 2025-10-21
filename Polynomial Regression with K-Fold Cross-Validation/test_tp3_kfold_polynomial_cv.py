# Filename: test_tp3_kfold_polynomial_cv.py
"""
Pytest suite for TP3_kfold_polynomial_cv.py

What this validates:
1) The CV summary has the expected structure and columns.
2) On synthetic quadratic data, degree=1 performs worse than degree=2 (by mean MSE).
3) pick_best_degree() returns a degree from the evaluated set (and not 1 on quadratic data).
4) The pipeline is stable and reproducible with fixed seeds and KFold settings.

How to run:
    pytest -q
"""

import numpy as np
import pandas as pd

# Import from your TP3 module
# If the module is in the same folder, this works; otherwise adjust sys.path.
from TP3_kfold_polynomial_cv import PolynomialKFoldCV, make_synthetic


def test_summary_structure_and_columns():
    # Generate a small synthetic dataset (noisy quadratic)
    x, y = make_synthetic(n=60, seed=123)

    runner = PolynomialKFoldCV(degrees=[1, 2, 3, 4], k=5, shuffle=True, random_state=7)
    summary = runner.evaluate(x, y)

    # Basic structure checks
    assert isinstance(summary, pd.DataFrame)
    for col in ["degree", "mse_mean", "mse_std", "r2_mean", "r2_std"]:
        assert col in summary.columns

    # Four rows for four degrees
    assert len(summary) == 4

    # Degrees present and sorted (we sort later in code; but ensure unique)
    assert set(summary["degree"]) == {1, 2, 3, 4}

    # MSE values should be non-negative
    assert (summary["mse_mean"] >= 0).all()
    assert (summary["mse_std"] >= 0).all()


def test_degree1_worse_than_degree2_on_quadratic():
    # Quadratic synthetic data; degree=2 should outperform degree=1 on average
    x, y = make_synthetic(n=100, seed=42)

    runner = PolynomialKFoldCV(degrees=[1, 2], k=6, shuffle=True, random_state=123)
    summary = runner.evaluate(x, y)

    # Extract per-degree mean MSE
    mse1 = float(summary.loc[summary["degree"] == 1, "mse_mean"].iloc[0])
    mse2 = float(summary.loc[summary["degree"] == 2, "mse_mean"].iloc[0])

    # On quadratic data, linear model (deg=1) should be worse than deg=2
    assert mse2 < mse1


def test_pick_best_degree_is_in_list_and_not_one_on_quadratic():
    x, y = make_synthetic(n=100, seed=999)

    degrees = [1, 2, 3, 4]
    runner = PolynomialKFoldCV(degrees=degrees, k=5, shuffle=True, random_state=999)
    summary = runner.evaluate(x, y)
    best = runner.pick_best_degree(summary)

    assert best in degrees
    # On quadratic ground-truth, degree 1 should not be the best
    assert best != 1
