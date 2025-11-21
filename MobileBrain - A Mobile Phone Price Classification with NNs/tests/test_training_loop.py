# tests/test_training_loop.py

import torch.nn as nn
from src.model import MobileNetNN
from src.train_eval import train_model, make_loaders
import numpy as np


def test_training_one_epoch():
    # tiny fake dataset to sanity-check training loop
    X = np.random.randn(50, 8).astype(np.float32)
    y = np.random.randint(0, 3, size=50)

    train_loader, test_loader = make_loaders(X[:40], X[40:], y[:40], y[40:], batch_size=8)

    model = MobileNetNN(8, 16, 3, nn.ReLU())
    acc = train_model(model, train_loader, test_loader, epochs=1, lr=0.01)
    assert 0.0 <= acc <= 1.0
