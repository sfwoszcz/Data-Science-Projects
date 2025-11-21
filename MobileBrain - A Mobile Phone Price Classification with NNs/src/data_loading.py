# src/data_loading.py

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_mobilephone_data(
    csv_path: str
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Loads the mobile phone dataset.

    Label:
        Price Range (low/medium/high) -> encoded as 0/1/2.

    Returns:
        X: features (float32)
        y: encoded labels (int64)
        label_encoder
    """
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c != "Price Range"]
    X = df[feature_cols].to_numpy().astype(np.float32)

    y_raw = df["Price Range"].astype(str).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)

    return X, y, le


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train/test split + StandardScaler feature scaling.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
