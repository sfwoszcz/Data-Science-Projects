.PHONY: setup build test
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
build:
	cmake -S . -B build && cmake --build build -j
test:
	pytest -q && ctest --test-dir build --output-on-failure || true
