# src/data_loading.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_mnist_from_csv(
    csv_path: str = "data/mnist_data.csv",
    batch_size: int = 100,
    train_split: float = 0.8,
):
    df = pd.read_csv(csv_path, header=None)

    data = df.to_numpy().astype(np.float32)
    labels = data[:, 0].astype(np.int64)
    pixels = data[:, 1:]  # shape: (N, 784)

    # scale pixels from [0,255] to [0.01,1.0]
    pixels = (pixels / 255.0) * 0.99 + 0.01

    X = torch.tensor(pixels, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    num_samples = X.shape[0]
    train_size = int(train_split * num_samples)
    test_size = num_samples - train_size

    indices = torch.randperm(num_samples)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
