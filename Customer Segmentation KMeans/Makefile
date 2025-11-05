.PHONY: install elbow kmeans2d kmeans3d test clean checkdata preview all

# Install all Python dependencies
install:
	pip install -r requirements.txt

# Run Elbow and Silhouette analysis to determine optimal k
elbow:
	python scripts/run_elbow.py

# Run 2D K-Means clustering (Annual Income vs Spending Score)
kmeans2d:
	python scripts/run_kmeans_2d.py

# Run 3D K-Means clustering (Age, Income, Spending)
kmeans3d:
	python scripts/run_kmeans_3d.py

# Run tests with pytest
test:
	pytest -q

# Remove generated images
clean:
	rm -f *.png

# Check whether the dataset is present
checkdata:
	@echo "Checking for dataset..."
	@if [ -f data/Mall_Customers.csv ]; then \
		echo " Dataset found: data/Mall_Customers.csv"; \
	else \
		echo " Dataset not found! Please download it from Kaggle and place it in the data/ folder."; \
		exit 1; \
	fi

# Preview the first 5 rows of the dataset
preview:
	@echo "Previewing dataset..."
	@if [ -f data/Mall_Customers.csv ]; then \
		python -c "import pandas as pd; df = pd.read_csv('data/Mall_Customers.csv'); print(df.head());"; \
	else \
		echo ' Dataset not found! Please download it from Kaggle and place it in data/Mall_Customers.csv'; \
		exit 1; \
	fi

# Run the full project workflow in order
all: checkdata elbow kmeans2d kmeans3d
	@echo " Full workflow completed successfully!"
