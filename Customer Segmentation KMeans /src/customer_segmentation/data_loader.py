# 3.5 src/customer_segmentation/data_loader.py

import os
import pandas as pd

def load_customer_data(
    path: str = "data/Mall_Customers.csv"
) -> pd.DataFrame:
    """
    Load the mall customer dataset from CSV.

    Expected columns (typical for this Kaggle dataset):
    - CustomerID
    - Gender
    - Age
    - Annual Income (k$)
    - Spending Score (1-100)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Please download from Kaggle and place it there."
        )
    df = pd.read_csv(path)
    return df
