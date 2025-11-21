# scripts/run_baseline_nn.py

import torch.nn as nn
import torch

from src.data_loading import load_mobilephone_data, split_and_scale
from src.train_eval import make_loaders, train_model
from src.model import MobileNetNN


def main():
    csv_path = "data/MobilePhone.csv"

    X, y, le = load_mobilephone_data(csv_path)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)

    train_loader, test_loader = make_loaders(X_train, X_test, y_train, y_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MobileNetNN(
        input_size=X_train.shape[1],
        hidden_size=32,
        output_size=len(set(y)),
        activation_fn=nn.ReLU()
    )

    acc = train_model(model, train_loader, test_loader,
                      epochs=50, lr=0.01, optimizer_type="adam", device=device)

    print("\nBaseline NN accuracy:", acc)


if __name__ == "__main__":
    main()

