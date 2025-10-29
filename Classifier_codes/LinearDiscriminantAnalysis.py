import numpy as np

class LDA:
    """Linear Discriminant Analysis classifier from scratch."""
    def __init__(self):
        self.priors = None      # Shape: (n_classes,)
        self.means = None       # Shape: (n_classes, n_features)
        self.shared_covariance = None # Shape: (n_features, n_features)
        self.shared_cov_inv = None # Inverse of shared covariance
        self.classes = None

    def fit(self, X, y):
        """Calculate priors, means, and the shared covariance matrix."""
        X = np.array(X)
        y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize
        self.priors = np.zeros(n_classes, dtype=np.float64)
        self.means = np.zeros((n_classes, n_features), dtype=np.float64)
        # Calculate shared covariance matrix (pooled covariance)
        S = np.zeros((n_features, n_features), dtype=np.float64)

        print("Fitting LDA...")
        for idx, c in enumerate(self.classes):
            X_c = X[y == c] # Samples belonging to class c
            n_c = X_c.shape[0]
            if n_c == 0:
                print(f"Warning: No samples for class {c}. Skipping.")
                continue

            # Calculate prior P(c)
            self.priors[idx] = n_c / float(n_samples)
            # Calculate mean vector for class c
            self.means[idx, :] = np.mean(X_c, axis=0)

            # Accumulate sum of squares for pooled covariance
            # S += sum((x - mean_c).T * (x - mean_c) for x in X_c)
            # More efficient way: S += (n_c - 1) * np.cov(X_c.T)
            if n_c > 1:
                 # Need at least 2 samples to calculate covariance for a class
                 S += (n_c - 1) * np.cov(X_c, rowvar=False) # rowvar=False because features are columns

        # Calculate the pooled/shared covariance matrix
        # Denominator is (n_samples - n_classes) if using unbiased sample cov estimates
        if n_samples > n_classes:
            self.shared_covariance = S / (n_samples - n_classes)
        else:
            # Handle cases with very few samples per class
            print("Warning: Not enough samples relative to classes to compute unbiased pooled covariance. Using n_samples as denominator.")
            self.shared_covariance = S / n_samples if n_samples > 0 else np.eye(n_features) * 1e-6


        # Calculate the inverse of the shared covariance matrix (needed for prediction)
        # Add small identity matrix (regularization) for numerical stability if determinant is near zero
        try:
            # Regularization term (ridge)
            reg = 1e-6 * np.eye(n_features)
            self.shared_cov_inv = np.linalg.inv(self.shared_covariance + reg)
        except np.linalg.LinAlgError:
            print("Error: Shared covariance matrix is singular. Cannot compute inverse.")
            # Fallback: Use pseudo-inverse or handle error
            print("Using pseudo-inverse as fallback.")
            self.shared_cov_inv = np.linalg.pinv(self.shared_covariance + reg)


        print("LDA fitting complete.")

    def predict(self, X):
        """Predict class labels using the linear discriminant function."""
        if self.priors is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_test = np.array(X)
        if X_test.shape[1] != self.means.shape[1]:
             raise ValueError(f"Input feature dimension mismatch: Expected {self.means.shape[1]}, got {X_test.shape[1]}")

        print(f"Predicting labels for {X_test.shape[0]} samples using LDA...")
        n_samples = X_test.shape[0]
        n_classes = len(self.classes)
        discriminant_scores = np.zeros((n_samples, n_classes))

        # Calculate discriminant function score for each class for each sample
        # score(c) = x @ inv(Cov) @ mean_c - 0.5 * mean_c @ inv(Cov) @ mean_c + log(P(c))
        for idx, c in enumerate(self.classes):
            mean_c = self.means[idx]
            prior_c = self.priors[idx]
            # Avoid log(0) for prior
            log_prior_c = np.log(prior_c + 1e-9)

            # Term 1: x @ inv(Cov) @ mean_c
            term1 = np.dot(X_test, np.dot(self.shared_cov_inv, mean_c))
            # Term 2: - 0.5 * mean_c @ inv(Cov) @ mean_c
            term2 = -0.5 * np.dot(mean_c.T, np.dot(self.shared_cov_inv, mean_c))

            discriminant_scores[:, idx] = term1 + term2 + log_prior_c

        # Predict the class with the highest discriminant score
        predictions = self.classes[np.argmax(discriminant_scores, axis=1)]
        return predictions