# ==========================================================
# Dockerfile for Case Study 2.C.02 — Mobile Phone Classification
# ----------------------------------------------------------
# Build:
#   docker build -t mobile-price-classification .
#
# Run classification (default):
#   docker run --rm -it mobile-price-classification
#
# Run visualization (saves PNGs in /app):
#   docker run --rm -it mobile-price-classification python mobile_visualization.py
#
# Save plots to host:
#   docker run --rm -it -v "$PWD":/app mobile-price-classification python mobile_visualization.py
# ==========================================================

FROM python:3.11-slim

# Avoid interactive prompts, make Python output unbuffered
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build tools (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (better layer caching)
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY mobile_visualization.py /app/mobile_visualization.py
COPY mobile_classification.py /app/mobile_classification.py
COPY mobile_price_classification.ipynb /app/mobile_price_classification.ipynb
COPY MobilePhone.csv /app/MobilePhone.csv
COPY README.md /app/README.md

# Optional: copy src/ and tests/ if you want them inside the image
COPY src /app/src
COPY tests /app/tests

# Default command: run classification
CMD ["python", "mobile_classification.py"]
