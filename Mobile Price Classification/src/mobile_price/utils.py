from typing import Tuple
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DEFAULT_PATHS = [
    "MobilePhone.csv",
    "/mnt/data/MobilePhone.csv",
]

def resolve_data_path() -> str:
    for p in DEFAULT_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"CSV not found in: {DEFAULT_PATHS}")

def load_df() -> pd.DataFrame:
    path = resolve_data_path()
    df = pd.read_csv(path)
    return df

def prepare_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    X = df[["battery_power", "frontcamermegapixels"]].values
    y_str = df["Price Range"].values
    le = LabelEncoder().fit(y_str)
    y = le.transform(y_str)
    return X, y, le
