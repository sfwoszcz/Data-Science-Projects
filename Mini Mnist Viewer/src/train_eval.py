# src/train_eval.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 5,
    lr: float = 0.1,
    device: str = "cpu",
) -> float:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")

    performance = evaluate_performance(model, test_loader, device=device)
    print(f"Final test performance: {performance:.4f}")
    return performance


def evaluate_performance(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs, dim=1)

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    return correct / total
