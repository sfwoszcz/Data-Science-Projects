FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
RUN cmake -S . -B build && cmake --build build -j
RUN pytest -q && ctest --test-dir build --output-on-failure
ENTRYPOINT ["python","-m","sensorkit.cli"]
CMD ["--help"]
