# scripts/run_activation_experiments.py

import torch
import torch.nn as nn

from src.data_loading import load_mnist_from_csv
from src.model import NN
from src.train_eval import train_one_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader = load_mnist_from_csv("data/mnist_data.csv", batch_size=100)

    input_size = 784
    hidden_size = 100
    output_size = 10
    num_epochs = 5
    learning_rate = 0.1

    performances = {}

    print("\n=== Sigmoid activation ===")
    sigmoid_model = NN(input_size, hidden_size, output_size, nn.Sigmoid())
    performances["Sigmoid"] = train_one_model(
        sigmoid_model, train_loader, test_loader,
        num_epochs=num_epochs, lr=learning_rate, device=device
    )

    leaky_slopes = [0.01, 0.05, 0.1, 0.5]
    for slope in leaky_slopes:
        print(f"\n=== LeakyReLU activation (negative_slope={slope}) ===")
        leaky_model = NN(input_size, hidden_size, output_size, nn.LeakyReLU(negative_slope=slope))
        performances[f"LeakyReLU_{slope}"] = train_one_model(
            leaky_model, train_loader, test_loader,
            num_epochs=num_epochs, lr=learning_rate, device=device
        )

    print("\n=== PReLU activation ===")
    prelu_model = NN(input_size, hidden_size, output_size, nn.PReLU())
    performances["PReLU"] = train_one_model(
        prelu_model, train_loader, test_loader,
        num_epochs=num_epochs, lr=learning_rate, device=device
    )

    elu_alphas = [0.1, 0.2, 0.3]
    for alpha in elu_alphas:
        print(f"\n=== ELU activation (alpha={alpha}) ===")
        elu_model = NN(input_size, hidden_size, output_size, nn.ELU(alpha=alpha))
        performances[f"ELU_{alpha}"] = train_one_model(
            elu_model, train_loader, test_loader,
            num_epochs=num_epochs, lr=learning_rate, device=device
        )

    print("\n=== Summary of performances ===")
    for name, perf in performances.items():
        print(f"{name}: {perf:.4f}")

    best_name = max(performances, key=performances.get)
    print(f"\nBest activation configuration: {best_name} "
          f"with performance {performances[best_name]:.4f}")


if __name__ == "__main__":
    main()
