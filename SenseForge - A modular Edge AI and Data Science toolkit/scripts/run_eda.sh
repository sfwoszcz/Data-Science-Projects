#!/usr/bin/env bash
set -euo pipefail
python -m sensorkit.cli --config configs/default.yaml plot data/sample_vibration.csv --out artifacts/eda.png
