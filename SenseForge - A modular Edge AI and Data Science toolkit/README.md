Week 1 — Sensor Log Toolkit (v2)

Python (OOP) + C parity + CI + docs.

---

##Quickstart (Ubuntu 24.04.3)
unzip week1-sensor-toolkit.zip -d ~/dev
cd ~/dev/week1-sensor-toolkit

# Optional venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build C lib and run tests
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure

# Python tests (parity)
pytest -q

# Try the CLI on sample data
python -m sensorkit.cli --config configs/default.yaml summarize data/sample_vibration.csv
python -m sensorkit.cli --config configs/default.yaml plot data/sample_vibration.csv --out artifacts/eda.png
python -m sensorkit.cli --config configs/default.yaml features data/sample_vibration.csv --out artifacts/features.csv

---


