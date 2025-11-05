.PHONY: install viz classify all clean docker-build docker-run

# Install Python dependencies
install:
	pip install -r requirements.txt

# Generate all required plots (2D + 3D)
viz:
	python mobile_visualization.py

# Train and evaluate KNN and RandomForest
classify:
	python mobile_classification.py

# Do everything
all: viz classify

# Remove generated images
clean:
	rm -f *.png

# Build docker image
docker-build:
	docker build -t mobile-price-classification .

# Run default container (classification)
docker-run:
	docker run --rm -it mobile-price-classification
