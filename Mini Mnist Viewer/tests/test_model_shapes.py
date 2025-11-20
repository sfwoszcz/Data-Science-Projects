# tests/test_model_shapes.py

import torch
import torch.nn as nn
from src.model import NN


def test_model_forward_shape():
    input_size = 784
    hidden_size = 100
    output_size = 10

    model = NN(input_size, hidden_size, output_size, nn.Sigmoid())
    x = torch.randn(16, input_size)
    y = model(x)

    assert y.shape == (16, output_size)
