import numpy as np

class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier from scratch."""
    def __init__(self, epsilon=1e-9):
        self.classes = None
        self.mean = None    # Shape: (n_classes, n_features)
        self.var = None     # Shape: (n_classes, n_features)
        self.priors = None  # Shape: (n_classes,)
        self.epsilon = epsilon # Small value added to variance for numerical stability

    def fit(self, X, y):
        """Calculate mean, variance, and priors for each class."""
        X = np.array(X)
        y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize storage for parameters
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        print("Fitting Gaussian Naive Bayes...")
        # Calculate parameters for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c] # Get samples belonging to the current class 'c'
            if X_c.shape[0] == 0:
                 print(f"Warning: No samples found for class {c}. Skipping parameter calculation for this class.")
                 # Handle this case: maybe assign default small priors/variances, or raise error
                 self.mean[idx, :] = 0
                 self.var[idx, :] = self.epsilon
                 self.priors[idx] = self.epsilon # Small prior
                 continue

            # Calculate mean and variance for each feature within this class
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + self.epsilon # Add epsilon for stability

            # Calculate prior probability P(c) = (samples in class c) / (total samples)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

        # Normalize priors if needed (though they should sum to 1 if no classes were empty)
        if np.sum(self.priors) < (1.0 - 1e-6): # Check if priors sum is significantly less than 1
             print("Warning: Priors do not sum to 1, possibly due to empty classes. Normalizing.")
             self.priors /= np.sum(self.priors)

        print("Fitting complete.")

    def _pdf(self, class_idx, x):
        """
        Gaussian Probability Density Function (PDF).
        Calculates P(x_i | C) for each feature x_i of a single sample x, given class C.
        Returns an array of probabilities, one for each feature.
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx] # Variance already includes epsilon

        # Ensure variance is positive before sqrt
        safe_var = np.maximum(var, self.epsilon)

        numerator = np.exp(-((x - mean) ** 2) / (2 * safe_var))
        denominator = np.sqrt(2 * np.pi * safe_var)

        # PDF value for each feature. Add epsilon to avoid exact zero.
        return (numerator / denominator) + self.epsilon

    def predict(self, X):
        """Predict class labels for new data."""
        if self.classes is None:
             raise RuntimeError("Model has not been fitted yet.")
        X_test = np.array(X)
        if X_test.shape[1] != self.mean.shape[1]:
             raise ValueError(f"Input feature dimension mismatch: Expected {self.mean.shape[1]}, got {X_test.shape[1]}")

        print(f"Predicting labels for {X_test.shape[0]} samples using GNB...")
        y_pred = [self._predict_single(x) for x in X_test]
        return np.array(y_pred)

    def _predict_single(self, x):
        """Predict the class for a single sample 'x' using Bayes' theorem with logs."""
        log_posteriors = []

        # Calculate log posterior probability for each class:
        # log(P(C | x)) ~ log(P(C)) + sum(log(P(x_i | C)))
        for idx, c in enumerate(self.classes):
            # Log prior: log(P(C))
            # Add epsilon to prior before log to handle potential zero priors (e.g., empty class)
            log_prior = np.log(self.priors[idx] + self.epsilon)

            # Log likelihood: sum over features of log(P(x_i | C))
            # P(x_i | C) is calculated using the Gaussian PDF (_pdf)
            pdfs_for_features = self._pdf(idx, x)
            log_likelihood = np.sum(np.log(pdfs_for_features)) # log(pdf) handles epsilon added in _pdf

            # Log posterior (proportional)
            log_posterior = log_prior + log_likelihood
            log_posteriors.append(log_posterior)

        # Return the class index corresponding to the maximum log posterior probability
        return self.classes[np.argmax(log_posteriors)]