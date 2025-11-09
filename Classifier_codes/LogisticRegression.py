import numpy as np

class LogisticRegression:
    """Binary Logistic Regression with L2 Regularization and Gradient Descent."""
    
    def __init__(self, learning_rate=0.01, n_iters=1000, alpha=0.01, verbose=False, tol=1e-4):
        """
        alpha (float): L2 regularization strength (lambda). 
                       Set to 0 for no regularization.
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if n_iters <= 0:
            raise ValueError("Number of iterations must be positive.")
        self.lr = learning_rate
        self.n_iters = n_iters
        self.alpha = alpha 
        self.weights = None
        self.bias = None
        self.verbose = verbose
        self.tol = tol # Tolerance for convergence check
        self.cost_history = []

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        # Clip to avoid overflow/underflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Train the model using gradient descent."""
        X = np.array(X)
        y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        if np.any((y != 0) & (y != 1)):
            raise ValueError("Logistic Regression currently supports binary labels (0 or 1).")

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        prev_cost = float('inf')

        print(f"Starting Logistic Regression training for up to {self.n_iters} iterations (alpha={self.alpha})...")
        # Gradient Descent
        for i in range(self.n_iters):
            # Linear model: z = X.w + b
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid: y_predicted = h(z)
            y_predicted = self._sigmoid(linear_model)

            dw_base = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Add L2 regularization term to weight gradient (bias is not regularized)
            l2_grad_term = (self.alpha / n_samples) * self.weights
            dw = dw_base + l2_grad_term

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Add epsilon (1e-9) to prevent log(0)
            log_loss = - (1 / n_samples) * np.sum(y * np.log(y_predicted + 1e-9) + (1 - y) * np.log(1 - y_predicted + 1e-9))
            
            # Add L2 cost term
            l2_cost_term = (self.alpha / (2 * n_samples)) * np.sum(self.weights**2)
            cost = log_loss + l2_cost_term
            
            self.cost_history.append(cost)

            # Optional: Check for convergence
            if abs(prev_cost - cost) < self.tol:
                print(f"Convergence reached at iteration {i}. Cost = {cost:.6f}")
                break
            prev_cost = cost

            # Print progress
            if self.verbose and (i % max(1, self.n_iters // 10) == 0 or i == self.n_iters - 1):
                print(f"Iteration {i}: Cost = {cost:.6f}")

        if i == self.n_iters - 1:
            print(f"Training finished after {self.n_iters} iterations. Final Cost = {self.cost_history[-1]:.6f}")

    def predict_proba(self, X):
        """Predict probability of class 1."""
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = np.array(X)
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Input feature dimension mismatch: Expected {self.weights.shape[0]}, got {X.shape[1]}")

        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted_proba = self._sigmoid(linear_model)
        return y_predicted_proba

    def predict(self, X, threshold=0.5):
        """Predict class labels (0 or 1)."""
        print(f"Predicting labels for {len(X)} samples using Logistic Regression...")
        y_predicted_proba = self.predict_proba(X)
        y_predicted_labels = (y_predicted_proba > threshold).astype(int)
        return y_predicted_labels