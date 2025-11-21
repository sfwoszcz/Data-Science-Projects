# src/knn_baseline.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def run_knn(X_train, X_test, y_train, y_test, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"KNN (k={k}) accuracy: {acc:.4f}")
    return acc
