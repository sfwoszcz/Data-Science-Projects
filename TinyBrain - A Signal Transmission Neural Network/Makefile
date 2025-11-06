# ============================================================
# Makefile for Signal Transmission Neural Network
# ============================================================

.PHONY: install run test docker-build docker-run plot

install:
	pip install -r requirements.txt

run:
	python scripts/run_signal_transmission.py

plot:
	python scripts/plot_activations.py

test:
	pytest -v

docker-build:
	docker build -t signal_transmission_nn .

docker-run:
	docker run --rm signal_transmission_nn
