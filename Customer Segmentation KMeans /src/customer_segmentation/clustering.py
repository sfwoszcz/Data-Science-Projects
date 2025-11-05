# 3.8 src/customer_segmentation/clustering.py

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def run_kmeans_2d(df: pd.DataFrame, k: int, outpath: str = "clusters_2d.png"):
    """
    Run K-Means on 2 features:
    - Annual Income (k$)
    - Spending Score (1-100)
    and plot the result.
    """
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X)

    plt.figure(figsize=(7, 5))
    plt.scatter(
        X[:, 0], X[:, 1],
        c=labels,
        s=40
    )
    plt.scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        s=200,
        c="red",
        marker="X",
        label="Centroids"
    )
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title(f"K-Means Clusters (k={k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return labels

def run_kmeans_3d(df: pd.DataFrame, k: int):
    """
    Run K-Means on 3 features:
    - Age
    - Annual Income (k$)
    - Spending Score (1-100)
    Returns labels so the caller can plot with 3D helper.
    """
    X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X)
    return labels