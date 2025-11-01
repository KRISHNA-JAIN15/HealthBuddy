import numpy as np

class LinearSVM:
    """
    Linear Support Vector Machine (Binary) with Batch Gradient Descent from scratch.
    
    EDITED: 
    1. Fixed the bias update rule in the gradient descent step.
    2. Converted the 'fit' method from Stochastic (per-sample) to 
       Batch (per-epoch) Gradient Descent for stability and speed.
    3. Added a 'decision_function' method, which is standard for SVMs.
    """
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
        """Train the SVM using Batch Gradient Descent and hinge loss."""
        X = np.array(X)
        y_input = np.array(y)
        if X.shape[0] != y_input.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        if len(np.unique(y_input)) > 2: # Allow for single-class data during OvR
             print("Warning: SVM currently implemented for binary classification. Ensure labels are {0, 1} or {-1, 1}.")

        # Convert labels to {-1, 1} which is standard for SVM formulation
        y_ = np.where(y_input <= 0, -1, 1)
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        prev_weights = self.weights.copy()

        print(f"Starting Linear SVM (Batch GD) training: iters={self.n_iters}, lr={self.lr}, lambda={self.lambda_param}...")
        
        # Batch Gradient Descent
        for i in range(self.n_iters):
            
            # --- 1. Calculate Scores and Cost ---
            # Linear model score: z = w.x - b (using -b for convention)
            # (n_samples,) array
            linear_output = np.dot(X, self.weights) - self.bias
            
            # Hinge loss condition: score = y_i * (w.x_i - b)
            # (n_samples,) array
            scores = y_ * linear_output
            
            # Find all samples that violate the margin (score < 1)
            # (n_samples,) boolean mask
            misclassified_mask = (scores < 1)
            
            # Calculate total cost for this epoch
            # Cost = Regularization Loss + Hinge Loss
            reg_cost = self.lambda_param * np.dot(self.weights, self.weights)
            hinge_loss = np.sum(1 - scores[misclassified_mask])
            
            # Average cost across all samples
            total_cost = reg_cost + (hinge_loss / n_samples)
            self.cost_history.append(total_cost)

            # --- 2. Calculate Gradients (Averaged over Batch) ---
            
            # Gradient of Regularization Term (w.r.t w)
            dw_reg = 2 * self.lambda_param * self.weights
            
            # Gradient of Hinge Loss Term (w.r.t w)
            # d(Hinge)/dw = sum(-y_i * x_i) for misclassified samples
            dw_hinge = -np.dot(y_[misclassified_mask], X[misclassified_mask])
            
            # Gradient of Hinge Loss Term (w.r.t b)
            # d(Hinge)/db = sum(y_i) for misclassified samples
            # Note: The original code's update was mathematically incorrect.
            # This is the correct gradient for the bias term.
            db_hinge = np.sum(y_[misclassified_mask])

            # Average gradients over the batch
            dw_total = dw_reg + (dw_hinge / n_samples)
            db_total = (db_hinge / n_samples) # Bias is not regularized

            # --- 3. Update Parameters ---
            self.weights -= self.lr * dw_total
            self.bias -= self.lr * db_total # Correct update: b_new = b_old - lr * gradient

            # --- 4. Check for Convergence ---
            weight_change = np.linalg.norm(self.weights - prev_weights)
            if weight_change < self.tol and i > 0:
                 print(f"Convergence reached at iteration {i}. Weight change = {weight_change:.6f}")
                 break
            prev_weights = self.weights.copy()

            if self.verbose and (i % max(1, self.n_iters // 10) == 0 or i == self.n_iters - 1):
                print(f"Iteration {i}: Average Cost = {total_cost:.6f}, Weight Change = {weight_change:.6f}")

        if i == self.n_iters - 1:
             print(f"Training finished after {self.n_iters} iterations. Final Cost = {self.cost_history[-1]:.6f}")


    def decision_function(self, X):
        """Returns the raw decision scores (w.x - b)."""
        if self.weights is None or self.bias is None:
             raise RuntimeError("Model has not been fitted yet.")
        X = np.array(X)
        if X.shape[1] != self.weights.shape[0]:
             raise ValueError(f"Input feature dimension mismatch: Expected {self.weights.shape[0]}, got {X.shape[1]}")
        
        # Calculate the decision function: score = w.x - b
        linear_output = np.dot(X, self.weights) - self.bias
        return linear_output


    def predict(self, X):
        """Predict class labels (0 or 1)."""
        print(f"Predicting labels for {X.shape[0]} samples using Linear SVM...")
        
        # Get the raw decision scores
        linear_output = self.decision_function(X)

        # Predict based on the sign of the score
        predictions_svm = np.sign(linear_output) # Output is -1 or 1

        # Convert back from {-1, 1} to {0, 1} for consistency
        y_predicted = np.where(predictions_svm <= 0, 0, 1) # Map -1 to 0, 1 to 1
        return y_predicted