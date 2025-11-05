# ==========================================================
# Program: mobile_visualization.py
# (Mobile Phone Price Classification)
# Part: Visualization
# ----------------------------------------------------------
# This script:
#   - Loads the CSV dataset "MobilePhone.csv" (from repo root by default)
#   - Creates the following plots with the price class encoded in colors:
#       1) Battery Power vs Internal Memory (2D Scatter)
#       2) Front Camera Megapixels vs Bluetooth (2D Scatter)
#       3) Internal Memory vs Front Camera Megapixels (2D Scatter)
#       4) 3D Scatter: Battery Power, Bluetooth, Front Camera Megapixels
#   - Saves figures as PNG files in the current directory.
# ==========================================================
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI / headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Prefer repo-root CSV; fallback to /mnt/data for this chat environment.
DEFAULT_PATHS = [
    "MobilePhone.csv",
    "/mnt/data/MobilePhone.csv",
]

def resolve_data_path() -> str:
    for p in DEFAULT_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"CSV not found in: {DEFAULT_PATHS}")

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {
        "battery_power", "blue", "dual_sim", "frontcamermegapixels",
        "four_g", "int_memory", "three_g", "touch_screen", "Price Range"
    }
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df

def encode_price_range(df: pd.DataFrame) -> np.ndarray:
    mapping = {'l': 0, 'm': 1, 'h': 2}
    y = df["Price Range"].map(mapping).values
    if np.any(pd.isna(y)):
        bad = df["Price Range"][pd.isna(df["Price Range"].map(mapping))].unique()
        raise ValueError(f"Unexpected labels in 'Price Range': {bad}")
    return y

def scatter_2d(x, y, c, xlabel, ylabel, title, filename):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, c=c, s=30)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def scatter_3d(x, y, z, c, labels, title, filename):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=c, s=30)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    data_path = resolve_data_path()
    df = load_data(data_path)
    y_class = encode_price_range(df)

    scatter_2d(
        x=df["battery_power"].values,
        y=df["int_memory"].values,
        c=y_class,
        xlabel="Battery Power",
        ylabel="Internal Memory (GB)",
        title="Battery Power vs Internal Memory (colored by Price Range)",
        filename="plot_battery_vs_memory.png"
    )

    scatter_2d(
        x=df["frontcamermegapixels"].values,
        y=df["blue"].values,
        c=y_class,
        xlabel="Front Camera Megapixels",
        ylabel="Bluetooth (0=no, 1=yes)",
        title="Front Camera Megapixels vs Bluetooth (colored by Price Range)",
        filename="plot_frontcam_vs_bluetooth.png"
    )

    scatter_2d(
        x=df["int_memory"].values,
        y=df["frontcamermegapixels"].values,
        c=y_class,
        xlabel="Internal Memory (GB)",
        ylabel="Front Camera Megapixels",
        title="Internal Memory vs Front Camera Megapixels (colored by Price Range)",
        filename="plot_memory_vs_frontcam.png"
    )

    scatter_3d(
        x=df["battery_power"].values,
        y=df["blue"].values,
        z=df["frontcamermegapixels"].values,
        c=y_class,
        labels=("Battery Power", "Bluetooth (0/1)", "Front Camera Megapixels"),
        title="3D: Battery Power, Bluetooth, Front Camera Megapixels (colored by Price Range)",
        filename="plot_3d_battery_blue_frontcam.png"
    )

    print("Saved plots:")
    print(" - plot_battery_vs_memory.png")
    print(" - plot_frontcam_vs_bluetooth.png")
    print(" - plot_memory_vs_frontcam.png")
    print(" - plot_3d_battery_blue_frontcam.png")

if __name__ == "__main__":
    main()
