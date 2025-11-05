#!/usr/bin/env bash
set -euo pipefail
python -m sensorkit.cli --config configs/default.yaml summarize data/sample_vibration.csv
python -m sensorkit.cli --config configs/default.yaml features data/sample_vibration.csv --out artifacts/features.csv
