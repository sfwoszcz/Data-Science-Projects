# Polynomial Regression with K-Fold Cross-Validation

**Goal:** Evaluate polynomial regression models of degrees **{1, 2, 3, 4}** using **k-fold cross-validation**, report **MSE** (mean ± std) and **R²**, and select the best degree by **lowest mean MSE**.

This is a compact, exam-grade **Data Science Analysis Project**:
- Clear problem definition
- Robust modeling & evaluation (CV)
- Reproducibility (tests & CI)
- Best practices (OOP, pipelines, code style)

---

## Features
- Degrees evaluated: 1, 2, 3, 4 (configurable)
- **K-Fold CV** with shuffling and fixed seeds
- Metrics: **MSE** (selection), **R²** (interpretability)
- OOP class: `PolynomialKFoldCV`
- Test suite (pytest)
- GitHub Actions CI
- Pre-commit hooks (Black, isort, Flake8)

---

## Installation

```bash
git clone https://github.com/<your-username>/tp3-polynomial-cv.git
cd tp3-polynomial-cv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

---

## Optional (recommended):
pip install pre-commit
pre-commit install

---

## Quickstart
# Run with synthetic data:
python TP3_kfold_polynomial_cv.py

---

# Run with your own CSV (must contain columns x,y):
python TP3_kfold_polynomial_cv.py --csv examples/sample_xy.csv --k 5 --degrees 1 2 3 4

---

# Flags:

   * --csv PATH Load CSV with columns x,y (optional)

   * --k INT Number of folds (default: 5)

   * --degrees D1 D2 ... Degrees to evaluate (default: 1 2 3 4)

   * --seed INT Random seed (default: 42)

   * --no-shuffle Disable shuffling before folding

---

# Output example:
=== K-Fold CV Summary (lower MSE is better) ===
 degree  mse_mean  mse_std  r2_mean  r2_std
      2    0.3612   0.0815   0.9287   0.0229
      3    0.3725   0.0891   0.9260   0.0241
      4    0.3884   0.0972   0.9225   0.0269
      1    1.9821   0.3746   0.5012   0.0813

Best degree by mean MSE: 2

---

## Tests
pytest -q

---

## Data format
Provide a CSV with two columns:

x,y
-3.0,-2.1
-2.9,-1.8
...

A sample is in examples/sample_xy.csv.

---

## Project structure

 * TP3_kfold_polynomial_cv.py — main module with PolynomialKFoldCV

 * test_tp3_kfold_polynomial_cv.py — pytest suite

 * examples/sample_xy.csv — demo CSV

 * .github/workflows/ci.yml — CI (tests on push/PR)

 * pyproject.toml — Black, isort, Flake8 config

 * .pre-commit-config.yaml — pre-commit hooks

---

## Citation
If you use this in teaching or learning materials, please cite:

TP3 Polynomial CV (2025). A compact DS project for model selection via K-Fold CV on polynomial regression.

---

## License (MIT)

---

## 2) `requirements.txt`
```txt
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
pytest>=7.0

---

## 3) .gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
*.egg-info/
.eggs/
.build/
dist/
.DS_Store

# Environments
.venv/
venv/
env/

# VSCode / IDE
.vscode/
.idea/

# Test / coverage
.coverage
htmlcov/
.pytest_cache/

# OS files
Thumbs.db

---

## 4) LICENSE (MIT)
MIT License

Copyright (c) 2025 submarine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

---

## 5) pyproject.toml (Black, isort, Flake8)
[tool.black]
line-length = 100
target-version = ["py310"]

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true

[tool.flake8]
max-line-length = 100
extend-ignore = ["E203","W503"]
exclude = [".venv",".git","__pycache__","build","dist"]

---

## 6) .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8

---

## 7) .github/workflows/ci.yml (GitHub Actions)
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Lint & Test
        run: |
          pip install black isort flake8
          black --check .
          isort --check-only .
          flake8 .
          pytest -q

---

## 8) examples/sample_xy.csv
x,y
-3.0,-2.7
-2.5,-2.0
-2.0,-1.1
-1.5,-0.4
-1.0,0.2
-0.5,0.6
0.0,0.9
0.5,1.0
1.0,0.8
1.5,0.3
2.0,-0.6
2.5,-1.7
3.0,-2.8

---

## 9) CONTRIBUTING.md
# Contributing

Thanks for considering a contribution! Please:

1. Open an issue describing the proposed change.
2. Fork the repo and create a feature branch.
3. Keep code style: Black + isort + Flake8 (see `pyproject.toml`).
4. Add or update tests in `test_tp3_kfold_polynomial_cv.py`.
5. Run locally:
   ```bash
   pip install -r requirements.txt
   pip install black isort flake8
   black .
   isort .
   flake8 .
   pytest -q
6. Open a PR referencing the issue.

We appreciate documentation improvements and examples too!

---

## 10) `CODE_OF_CONDUCT.md`
```markdown
# Code of Conduct

Be respectful, helpful, and inclusive. Discrimination, harassment, or toxicity are not tolerated. Violations may result in limitation of participation.

---

## 11) CHANGELOG.md
# Changelog

## [0.1.0] - 2025-10-21
### Added
- Initial release: Polynomial K-Fold CV for degrees {1,2,3,4}
- Metrics: MSE (mean ± std), R² (mean ± std)
- OOP core class `PolynomialKFoldCV`
- Pytest suite
- GitHub Actions CI
- Pre-commit hooks and style config

---

## 12) Core Code & Tests (already provided)
Ensure these two files are present (as we wrote earlier):

 * kfold_polynomial_cv.py
 
 * test_kfold_polynomial_cv.py
-------------------------------------------------------------------------------
