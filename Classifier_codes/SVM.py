import numpy as np

class LinearSVM:
    """Linear Support Vector Machine (Binary) with Gradient Descent from scratch."""
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, verbose=False, tol=1e-5):
        if learning_rate <= 0: raise ValueError("Learning rate must be positive.")
        if lambda_param < 0: raise ValueError("Lambda (regularization) parameter cannot be negative.")
        if n_iters <= 0: raise ValueError("Number of iterations must be positive.")

        self.lr = learning_rate
        self.lambda_param = lambda_param # Regularization strength ( inversely related to C)
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.verbose = verbose
        self.tol = tol # Tolerance for convergence check based on weight change
        self.cost_history = []


    def fit(self, X, y):
        """Train the SVM using gradient descent and hinge loss."""
        X = np.array(X)
        y_input = np.array(y)
        if X.shape[0] != y_input.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        if len(np.unique(y_input)) != 2:
             print("Warning: SVM currently implemented for binary classification. Ensure labels are {0, 1} or {-1, 1}.")

        # Convert labels to {-1, 1} which is standard for SVM formulation
        y_ = np.where(y_input <= 0, -1, 1)
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        prev_weights = self.weights.copy()

        print(f"Starting Linear SVM training: iters={self.n_iters}, lr={self.lr}, lambda={self.lambda_param}...")
        # Gradient Descent
        for i in range(self.n_iters):
            cost_epoch = 0
            dw_epoch = np.zeros_like(self.weights)
            db_epoch = 0

            # Iterate over each sample (Stochastic Gradient Descent like approach)
            for idx, x_i in enumerate(X):
                y_i = y_[idx]
                # Condition for hinge loss: y_i * (w.x_i - b) >= 1
                score = y_i * (np.dot(x_i, self.weights) - self.bias)
                is_correct_margin = score >= 1

                # Calculate gradients based on hinge loss
                if is_correct_margin:
                    # Point is correctly classified and outside margin
                    # Gradient only from regularization term
                    dw = 2 * self.lambda_param * self.weights
                    db = 0
                    # Cost contribution: regularization only
                    cost_epoch += self.lambda_param * np.dot(self.weights, self.weights)
                else:
                    # Point is misclassified or inside margin
                    # Gradient from both loss and regularization
                    dw = 2 * self.lambda_param * self.weights - y_i * x_i
                    db = y_i # Gradient w.r.t bias is -y_i
                    # Cost contribution: hinge loss + regularization
                    cost_epoch += self.lambda_param * np.dot(self.weights, self.weights) + (1 - score)

                # --- Update parameters ---
                # This is closer to SGD as updates happen per sample.
                # For batch GD, accumulate dw and db over all samples then update once.
                self.weights -= self.lr * dw
                self.bias -= self.lr * (-db) # Bias update uses -gradient; gradient was y_i


            # Store average cost for the epoch
            avg_cost = cost_epoch / n_samples
            self.cost_history.append(avg_cost)

            # Optional: Check for convergence based on weight change
            weight_change = np.linalg.norm(self.weights - prev_weights)
            if weight_change < self.tol:
                 print(f"Convergence reached at iteration {i}. Weight change = {weight_change:.6f}")
                 break
            prev_weights = self.weights.copy()

            # Print progress
            if self.verbose and (i % max(1, self.n_iters // 10) == 0 or i == self.n_iters - 1):
                print(f"Iteration {i}: Average Cost = {avg_cost:.6f}, Weight Change = {weight_change:.6f}")

        if i == self.n_iters - 1:
             print(f"Training finished after {self.n_iters} iterations. Final Cost = {self.cost_history[-1]:.6f}")


    def predict(self, X):
        """Predict class labels (0 or 1)."""
        if self.weights is None or self.bias is None:
             raise RuntimeError("Model has not been fitted yet.")
        X = np.array(X)
        if X.shape[1] != self.weights.shape[0]:
             raise ValueError(f"Input feature dimension mismatch: Expected {self.weights.shape[0]}, got {X.shape[1]}")

        print(f"Predicting labels for {X.shape[0]} samples using Linear SVM...")
        # Calculate the decision function: score = w.x - b
        linear_output = np.dot(X, self.weights) - self.bias

        # Predict based on the sign of the score
        predictions_svm = np.sign(linear_output) # Output is -1 or 1

        # Convert back from {-1, 1} to {0, 1} for consistency
        y_predicted = np.where(predictions_svm <= 0, 0, 1) # Map -1 to 0, 1 to 1
        return y_predicted