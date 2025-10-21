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