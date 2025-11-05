# ==========================================================
# Program: mobile_classification.py
# (Mobile Phone Price Classification)
# Part: Supervised Classification
# ----------------------------------------------------------
# - Uses ONLY: battery_power and frontcamermegapixels
# - Trains and evaluates:
#     * KNeighborsClassifier (with StandardScaler)
#     * RandomForestClassifier
# - Prints accuracy + classification report
# - Saves confusion matrix plots for both models
# ==========================================================
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive for CI/headless
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    expected_cols = {"battery_power", "frontcamermegapixels", "Price Range"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df

def prepare_xy(df: pd.DataFrame):
    X = df[["battery_power", "frontcamermegapixels"]].values
    y_str = df["Price Range"].values
    le = LabelEncoder().fit(y_str)
    y = le.transform(y_str)
    return X, y, le

def plot_confusion(cm: np.ndarray, class_names, title: str, filename: str):
    """Save a simple confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    data_path = resolve_data_path()
    df = load_data(data_path)
    X, y, le = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 1) KNN with scaling
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])
    knn_pipeline.fit(X_train, y_train)
    y_pred_knn = knn_pipeline.predict(X_test)

    print("=== KNeighborsClassifier ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_knn, target_names=le.classes_))
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    plot_confusion(cm_knn, le.classes_, title="Confusion Matrix - KNN", filename="confusion_knn.png")

    # 2) RandomForest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("\n=== RandomForestClassifier ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plot_confusion(cm_rf, le.classes_, title="Confusion Matrix - RandomForest", filename="confusion_rf.png")

    print("\nSaved confusion matrices:")
    print(" - confusion_knn.png")
    print(" - confusion_rf.png")

if __name__ == "__main__":
    main()
