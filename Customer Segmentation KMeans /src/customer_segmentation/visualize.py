# 3.6 src/customer_segmentation/visualize.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd

def scatter_2d_income_spending(df: pd.DataFrame, outpath: str = "plot_income_spending.png"):
    """
    Plot Annual Income (k$) vs Spending Score (1-100).
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], s=40)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Annual Income vs Spending Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def scatter_3d_age_income_spending(df: pd.DataFrame, labels, outpath: str = "plot_3d_clusters.png"):
    """
    3D scatter: Age, Annual Income, Spending Score, colored by cluster labels.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        df["Age"],
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=labels,
        s=40
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    ax.set_title("3D Clusters (Age, Income, Spending)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
