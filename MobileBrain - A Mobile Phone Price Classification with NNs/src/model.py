# src/model.py

import torch
import torch.nn as nn


class MobileNetNN(nn.Module):
    """
    Simple MLP for tabular phone data.

    input_size -> hidden_size -> output_size
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, activation_fn: nn.Module):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = activation_fn
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        x = self.act(self.fc1(x))
        x = self.fc2(x)  # logits
        return x
