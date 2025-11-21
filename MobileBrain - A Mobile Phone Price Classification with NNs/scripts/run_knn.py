# scripts/run_knn.py

from src.data_loading import load_mobilephone_data, split_and_scale
from src.knn_baseline import run_knn


def main():
    csv_path = "data/MobilePhone.csv"

    X, y, le = load_mobilephone_data(csv_path)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)

    acc = run_knn(X_train, X_test, y_train, y_test, k=5)
    print("\nBaseline KNN accuracy:", acc)


if __name__ == "__main__":
    main()
