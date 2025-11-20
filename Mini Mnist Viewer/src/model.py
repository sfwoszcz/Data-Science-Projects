# src/model.py

import torch.nn as nn


class NN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, activation_fn: nn.Module):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = activation_fn

    def forward(self, x):
        # x expected shape: (batch_size, 784)
        x = self.activation(self.layer1(x))
        x = self.layer2(x)  # logits for CrossEntropyLoss
        return x
