#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cmake -S . -B build
cmake --build build -j
pytest -q || true
ctest --test-dir build --output-on-failure || true
echo 'Setup complete.'
