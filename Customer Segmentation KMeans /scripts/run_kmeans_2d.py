# 4.2 scripts/run_kmeans_2d.py

from src.customer_segmentation.data_loader import load_customer_data
from src.customer_segmentation.clustering import run_kmeans_2d

# You can change this after checking the elbow/silhouette plots
BEST_K = 5

def main():
    df = load_customer_data()
    run_kmeans_2d(df, k=BEST_K, outpath="clusters_2d.png")
    print(f"2D clusters saved to clusters_2d.png with k={BEST_K}")

if __name__ == "__main__":
    main()
