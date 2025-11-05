# 4.3 scripts/run_kmeans_3d.py

from src.customer_segmentation.data_loader import load_customer_data
from src.customer_segmentation.clustering import run_kmeans_3d
from src.customer_segmentation.visualize import scatter_3d_age_income_spending

BEST_K = 5

def main():
    df = load_customer_data()
    labels = run_kmeans_3d(df, k=BEST_K)
    scatter_3d_age_income_spending(df, labels, outpath="clusters_3d.png")
    print(f"3D clusters saved to clusters_3d.png with k={BEST_K}")

if __name__ == "__main__":
    main()
