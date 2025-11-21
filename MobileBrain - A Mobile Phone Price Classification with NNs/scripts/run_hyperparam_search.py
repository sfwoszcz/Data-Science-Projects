# scripts/run_hyperparam_search.py

import torch

from src.data_loading import load_mobilephone_data, split_and_scale
from src.train_eval import make_loaders
from src.hyperparam_search import search_hyperparams


def main():
    csv_path = "data/MobilePhone.csv"

    X, y, le = load_mobilephone_data(csv_path)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)

    train_loader, test_loader = make_loaders(X_train, X_test, y_train, y_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = search_hyperparams(
        input_size=X_train.shape[1],
        output_size=len(set(y)),
        train_loader=train_loader,
        test_loader=test_loader,
        device=device
    )

    best_name = max(results, key=results.get)
    print("\n=== Hyperparameter Search Summary ===")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
    print(f"\nBest configuration: {best_name} with {results[best_name]:.4f}")


if __name__ == "__main__":
    main()
