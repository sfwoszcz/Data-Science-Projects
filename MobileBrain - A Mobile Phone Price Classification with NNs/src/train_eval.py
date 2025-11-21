# src/train_eval.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
import numpy as np


def make_loaders(X_train, X_test, y_train, y_test, batch_size=32) -> Tuple[DataLoader, DataLoader]:
    """
    Converts numpy arrays to PyTorch TensorDataset + DataLoaders.
    """
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_model(model: nn.Module,
                train_loader: DataLoader,
                test_loader: DataLoader,
                epochs: int = 50,
                lr: float = 0.01,
                optimizer_type: str = "adam",
                device: str = "cpu") -> float:
    """
    Trains the model and returns test accuracy.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

    return evaluate_model(model, test_loader, device=device)


def evaluate_model(model: nn.Module, test_loader: DataLoader, device="cpu") -> float:
    """
    Computes accuracy on the test set.
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            preds = logits.argmax(dim=1)

            total += yb.size(0)
            correct += (preds == yb).sum().item()

    acc = correct / total
    print(f"Test accuracy: {acc:.4f}")
    return acc

