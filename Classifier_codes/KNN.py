import numpy as np
from collections import Counter

class KNN:
    """K-Nearest Neighbors classifier from scratch."""
    def __init__(self, k=3):
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        if self.X_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty.")
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        print(f"KNN fitted with {self.X_train.shape[0]} samples.")

    def _euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """Predict labels for new data."""
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_test = np.array(X)
        if X_test.shape[1] != self.X_train.shape[1]:
             raise ValueError(f"Input feature dimension mismatch: Expected {self.X_train.shape[1]}, got {X_test.shape[1]}")
        print(f"Predicting for {X_test.shape[0]} samples using KNN (k={self.k})...")
        y_pred = [self._predict_single(x) for x in X_test]
        return np.array(y_pred)

    def _predict_single(self, x):
        """Predict label for a single test point."""
        # Calculate distances to all training points
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get indices of k nearest neighbors
        # Ensure k is not larger than the number of training samples
        effective_k = min(self.k, len(self.X_train))
        k_indices = np.argsort(distances)[:effective_k]

        # Get labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class label (majority vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        if not most_common:
             # This should only happen if k_nearest_labels is empty, which means k=0 or data error
             print("Warning: Could not determine most common label for a sample.")
             # Return a default or handle as error - returning first training label as fallback
             return self.y_train[0] if len(self.y_train) > 0 else 0
        return most_common[0][0]