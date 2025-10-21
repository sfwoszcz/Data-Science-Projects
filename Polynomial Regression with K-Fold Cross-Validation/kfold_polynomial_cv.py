# Filename: kfold_polynomial_cv.py
"""
Task: "Write code that performs k-fold cross-validation for the four possible polynomial degrees."

This script evaluates polynomial regression models of degrees [1, 2, 3, 4]
using k-fold cross-validation and reports mean±std MSE and R² per degree.
It selects the best degree by the lowest mean MSE.

Key features
------------
- OOP design via `PolynomialKFoldCV`
- Uses scikit-learn: Pipeline(PolynomialFeatures, LinearRegression)
- Metrics: Mean Squared Error (MSE) and R² (coefficient of determination)
- k-fold CV with reproducible shuffling
- Optional CSV input (columns: x, y) or built-in synthetic demo

Run examples
------------
1) With synthetic data (default):
    python TP3_kfold_polynomial_cv.py

2) With your CSV file (must contain columns named 'x' and 'y'):
    python TP3_kfold_polynomial_cv.py --csv path/to/data.csv --k 5

3) Change degrees or k:
    python TP3_kfold_polynomial_cv.py --degrees 1 2 3 4 --k 10

Author’s note
-------------
- This fulfills the exam requirement exactly: k-fold CV over four polynomial degrees.
- Metric reported is MSE (lower is better), plus R² for insight.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------------
# Core OOP: K-fold cross-validation for polynomial regression
# ---------------------------------------------------------------------
@dataclass
class PolynomialKFoldCV:
    """
    K-fold cross-validation runner for polynomial regression degrees.

    Parameters
    ----------
    degrees : Iterable[int]
        Polynomial degrees to evaluate (e.g., [1, 2, 3, 4]).
    k : int
        Number of folds for cross-validation (k ≥ 2).
    shuffle : bool
        Whether to shuffle before splitting into batches.
    random_state : int
        Seed for reproducible shuffling when shuffle=True.

    Notes
    -----
    - We evaluate each degree with the same KFold splits for fairness.
    - We compute both MSE and R² per fold, then aggregate mean±std.
    - Best degree is selected by lowest mean MSE.
    """

    degrees: Iterable[int] = (1, 2, 3, 4)
    k: int = 5
    shuffle: bool = True
    random_state: int = 42

    def _build_model(self, degree: int) -> Pipeline:
        """
        Build a regression pipeline: PolynomialFeatures(degree) -> LinearRegression.

        We include a bias term (include_bias=True), and LinearRegression fits the intercept;
        this is mathematically redundant but harmless—scikit-learn handles this consistently.
        """
        return Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
                ("linreg", LinearRegression()),
            ]
        )

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Run k-fold CV for each degree and return a summary table.

        Parameters
        ----------
        x : np.ndarray, shape (n,)
            One-dimensional feature values.
        y : np.ndarray, shape (n,)
            Target values.

        Returns
        -------
        pd.DataFrame
            Columns:
                - degree
                - mse_mean, mse_std
                - r2_mean, r2_std
        """
        # Ensure shapes
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        if x.size != y.size or x.size < self.k:
            raise ValueError("x and y must have same length >= k.")

        # Prepare CV splitter (reused across degrees to keep comparability)
        kf = KFold(n_splits=self.k, shuffle=self.shuffle, random_state=self.random_state)
        results: List[Dict[str, float]] = []

        # Evaluate each degree
        for deg in self.degrees:
            mse_scores: List[float] = []
            r2_scores: List[float] = []

            model = self._build_model(deg)

            # K-fold loop
            for train_idx, test_idx in kf.split(x):
                x_tr, y_tr = x[train_idx], y[train_idx]
                x_te, y_te = x[test_idx], y[test_idx]

                # scikit-learn expects X to be 2D; reshape to (n_samples, 1)
                X_tr = x_tr.reshape(-1, 1)
                X_te = x_te.reshape(-1, 1)

                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)

                mse_scores.append(mean_squared_error(y_te, y_pred))
                r2_scores.append(r2_score(y_te, y_pred))

            # Aggregate per degree
            results.append(
                {
                    "degree": int(deg),
                    "mse_mean": float(np.mean(mse_scores)),
                    "mse_std": float(np.std(mse_scores, ddof=1)) if len(mse_scores) > 1 else 0.0,
                    "r2_mean": float(np.mean(r2_scores)),
                    "r2_std": float(np.std(r2_scores, ddof=1)) if len(r2_scores) > 1 else 0.0,
                }
            )

        # Construct a tidy summary table sorted by mse_mean (ascending)
        summary = pd.DataFrame(results).sort_values(by="mse_mean", ascending=True).reset_index(drop=True)
        return summary

    @staticmethod
    def pick_best_degree(summary_df: pd.DataFrame) -> int:
        """
        Choose the best degree by the lowest mean MSE.
        """
        if summary_df.empty:
            raise ValueError("Empty summary table.")
        return int(summary_df.loc[summary_df["mse_mean"].idxmin(), "degree"])


# ---------------------------------------------------------------------
# Utility: Load CSV (columns must be named 'x' and 'y')
# ---------------------------------------------------------------------
def load_xy_from_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load x and y from a CSV file that contains columns 'x' and 'y'.

    Returns
    -------
    x : np.ndarray, shape (n,)
    y : np.ndarray, shape (n,)
    """
    df = pd.read_csv(path)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("CSV must contain columns named 'x' and 'y'.")
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return x, y


# ---------------------------------------------------------------------
# Demo data (if no CSV provided): a noisy quadratic ground-truth
# ---------------------------------------------------------------------
def make_synthetic(n: int = 80, seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic 1D data with quadratic trend:
        y = 1.0 + 2.0*x - 0.5*x^2 + noise
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-3.0, 3.0, n)
    y_true = 1.0 + 2.0 * x - 0.5 * x**2
    y = y_true + rng.normal(scale=0.6, size=n)
    return x, y


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TP_3: k-fold cross-validation for polynomial degrees [1,2,3,4]."
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional CSV with columns 'x' and 'y'. If omitted, synthetic data is used.")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for CV (k ≥ 2).")
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Polynomial degrees to evaluate, e.g., --degrees 1 2 3 4")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling before folding.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    args = parser.parse_args()

    # Load data
    if args.csv:
        x, y = load_xy_from_csv(args.csv)
    else:
        x, y = make_synthetic()

    # Configure runner
    runner = PolynomialKFoldCV(
        degrees=args.degrees,
        k=args.k,
        shuffle=not args.no_shuffle,
        random_state=args.seed
    )

    # Evaluate
    summary = runner.evaluate(x, y)
    best_deg = runner.pick_best_degree(summary)

    # Pretty print
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 10)
    print("\n=== K-Fold CV Summary (lower MSE is better) ===")
    print(summary.to_string(index=False))
    print(f"\nBest degree by mean MSE: {best_deg}")

if __name__ == "__main__":
    main()
