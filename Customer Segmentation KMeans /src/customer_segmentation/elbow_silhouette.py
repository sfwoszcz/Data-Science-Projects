# 3.7 src/customer_segmentation/elbow_silhouette.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def compute_inertia_over_k(X, k_range=range(2, 11)):
    """
    Compute inertia for each k in k_range.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(X)
        results.append((k, km.inertia_))
    return results

def plot_elbow(inertia_results, outpath="elbow_plot.png"):
    ks = [k for k, _ in inertia_results]
    inertias = [i for _, i in inertia_results]
    plt.figure(figsize=(7, 5))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method (K-Means)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def compute_silhouette_over_k(X, k_range=range(2, 11)):
    """
    Compute silhouette score for each k.
    """
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append((k, score))
    return scores

def plot_silhouette(scores, outpath="silhouette_plot.png"):
    ks = [k for k, _ in scores]
    vals = [s for _, s in scores]
    plt.figure(figsize=(7, 5))
    plt.plot(ks, vals, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette coefficient")
    plt.title("Silhouette Scores for K-Means")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()