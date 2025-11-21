# tests/test_model_forward.py

import torch
import torch.nn as nn
from src.model import MobileNetNN


def test_model_forward_shape():
    model = MobileNetNN(8, 32, 3, nn.ReLU())
    x = torch.randn(4, 8)
    y = model(x)
    assert y.shape == (4, 3)


