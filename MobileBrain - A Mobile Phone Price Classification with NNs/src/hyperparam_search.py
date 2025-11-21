# src/hyperparam_search.py

import torch.nn as nn
from typing import Dict
from .model import MobileNetNN
from .train_eval import train_model


def search_hyperparams(input_size, output_size, train_loader, test_loader, device="cpu") -> Dict[str, float]:
    """
    Lightweight hyperparameter search.
    """
    results = {}
    hidden_sizes = [16, 32, 64]
    lrs = [0.1, 0.01, 0.001]
    epochs_list = [30, 60]
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leakyrelu": nn.LeakyReLU(0.01)
    }
    optimizers = ["sgd", "adam"]

    for hs in hidden_sizes:
        for lr in lrs:
            for ep in epochs_list:
                for opt in optimizers:
                    for act_name, act_fn in activations.items():
                        name = f"hs{hs}_lr{lr}_ep{ep}_{opt}_{act_name}"
                        print("\n---", name, "---")
                        model = MobileNetNN(input_size, hs, output_size, act_fn)
                        acc = train_model(model, train_loader, test_loader,
                                          epochs=ep, lr=lr, optimizer_type=opt, device=device)
                        results[name] = acc

    return results
