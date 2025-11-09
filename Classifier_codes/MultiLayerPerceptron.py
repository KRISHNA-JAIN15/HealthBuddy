import numpy as np
import random # For potential weight initialization variations

class SimpleMLP:
    """
    Simple Multi-Layer Perceptron (1 hidden layer) for binary classification
    with L2 Regularization.
    """
    
    def __init__(self, n_input, n_hidden, n_output=1, learning_rate=0.01, n_iters=1000, 
                 alpha=0.01, verbose=False, random_state=None, activation='sigmoid'):
        """
        Initialize the MLP.
        
        Parameters:
        n_input (int): Number of features in the input data.
        n_hidden (int): Number of nodes in the hidden layer.
        n_output (int): Number of output nodes (should be 1 for binary classification).
        learning_rate (float): Step size for gradient descent.
        n_iters (int): Number of passes over the training data.
        alpha (float): L2 regularization strength (lambda). Set to 0 for no regularization.
        verbose (bool): If True, prints cost during training.
        random_state (int): Seed for random number generator for reproducible weight initialization.
        activation (str): Activation function for the hidden layer: 'sigmoid' or 'relu'.
        """
        
        if n_input <= 0 or n_hidden <= 0 or n_output <= 0:
            raise ValueError("Layer sizes must be positive.")
        if learning_rate <= 0: raise ValueError("Learning rate must be positive.")
        if n_iters <= 0: raise ValueError("Number of iterations must be positive.")

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = learning_rate
        self.n_iters = n_iters
        self.alpha = alpha # L2 Regularization strength
        self.verbose = verbose
        self.activation_type = activation.lower()
        if self.activation_type not in ['sigmoid', 'relu']:
            raise ValueError("Activation must be 'sigmoid' or 'relu'.")

        # Initialize weights and biases
        if random_state:
            np.random.seed(random_state)

        # Scaled (He/Glorot) initialization helps with training
        # Weights from input to hidden layer (shape: n_input x n_hidden)
        self.W1 = np.random.randn(self.n_input, self.n_hidden) * np.sqrt(1. / self.n_input) 
        self.b1 = np.zeros((1, self.n_hidden))
        # Weights from hidden to output layer (shape: n_hidden x n_output)
        self.W2 = np.random.randn(self.n_hidden, self.n_output) * np.sqrt(1. / self.n_hidden)
        self.b2 = np.zeros((1, self.n_output))

        self.cost_history = []

    # --- Activation Functions and Derivatives ---

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        z = np.clip(z, -500, 500) # Prevent overflow in exp()
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        """Derivative of sigmoid (where 'a' is the activation, sigmoid(z))."""
        return a * (1 - a)

    def _relu(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        """Derivative of ReLU (where 'z' is the pre-activation)."""
        # Returns 1 for z > 0, 0 otherwise
        return (z > 0).astype(float)

    def _apply_activation(self, z):
        """Applies the chosen activation function for the hidden layer."""
        if self.activation_type == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation_type == 'relu':
            return self._relu(z)

    def _apply_activation_derivative(self, cache_val):
        """
        Applies derivative of the chosen activation. 
        Takes Z or A depending on the function's requirement.
        """
        if self.activation_type == 'sigmoid':
            # Derivative needs the activation A = sigmoid(Z)
            return self._sigmoid_derivative(cache_val) # Pass A1
        elif self.activation_type == 'relu':
            # Derivative needs the pre-activation Z
            return self._relu_derivative(cache_val) # Pass Z1

    # --- Forward and Backward Propagation ---

    def _forward_propagation(self, X):
        """Perform forward pass through the network."""
        # Input to Hidden Layer
        Z1 = np.dot(X, self.W1) + self.b1        # Linear step
        A1 = self._apply_activation(Z1)         # Activation step (hidden layer)

        # Hidden to Output Layer
        Z2 = np.dot(A1, self.W2) + self.b2       # Linear step
        A2 = self._sigmoid(Z2)                  # Activation step (output layer - always sigmoid for binary prob)

        # Cache values needed for backpropagation
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def _compute_cost(self, A2, Y):
        """Compute Binary Cross-Entropy Cost with L2 Regularization."""
        m = Y.shape[0]
        if m == 0: return 0
        
        # Log Loss (Binary Cross-Entropy)
        # Add epsilon to prevent log(0)
        log_loss = - (1 / m) * np.sum(Y * np.log(A2 + 1e-9) + (1 - Y) * np.log(1 - A2 + 1e-9))
        
        # L2 Regularization Cost
        l2_cost = (self.alpha / (2 * m)) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        cost = log_loss + l2_cost
        
        return np.squeeze(cost) # Ensure cost is a scalar

    def _backward_propagation(self, cache, X, Y):
        """Perform backward pass to calculate gradients."""
        m = X.shape[0]
        if m == 0: # Avoid division by zero if batch is empty
            return {"dW1": np.zeros_like(self.W1), "db1": np.zeros_like(self.b1),
                    "dW2": np.zeros_like(self.W2), "db2": np.zeros_like(self.b2)}

        Z1, A1, A2 = cache['Z1'], cache['A1'], cache['A2']

        # --- Output Layer Gradients ---
        # Error gradient w.r.t Z2 (pre-activation of output)
        # For Binary Cross-Entropy with Sigmoid output, dCost/dZ2 = A2 - Y
        dZ2 = A2 - Y
        
        # Gradient w.r.t weights W2 (with L2 regularization term)
        dW2_base = (1 / m) * np.dot(A1.T, dZ2)
        dW2 = dW2_base + (self.alpha / m) * self.W2
        
        # Gradient w.r.t bias b2 (biases are not regularized)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # --- Hidden Layer Gradients ---
        # Error gradient w.r.t A1 (activation of hidden layer)
        dA1 = np.dot(dZ2, self.W2.T)
        
        # Error gradient w.r.t Z1 (pre-activation of hidden layer)
        # dZ1 = dA1 * derivative_of_activation(Z1 or A1)
        if self.activation_type == 'sigmoid':
            activation_deriv = self._apply_activation_derivative(A1) # Pass A1
        else: # ReLU
            activation_deriv = self._apply_activation_derivative(Z1) # Pass Z1
            
        dZ1 = dA1 * activation_deriv # Element-wise product
        
        # Gradient w.r.t weights W1 (with L2 regularization term)
        dW1_base = (1 / m) * np.dot(X.T, dZ1)
        dW1 = dW1_base + (self.alpha / m) * self.W1
        
        # Gradient w.r.t bias b1
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    def _update_parameters(self, grads):
        """Update weights and biases using gradients and learning rate."""
        self.W1 -= self.lr * grads['dW1']
        self.b1 -= self.lr * grads['db1']
        self.W2 -= self.lr * grads['dW2']
        self.b2 -= self.lr * grads['db2']

    def fit(self, X, y):
        """Train the MLP using batch gradient descent."""
        X = np.array(X)
        # Reshape y to be a column vector (m, 1) for calculations
        Y = np.array(y).reshape(-1, 1) 

        if X.shape[1] != self.n_input:
            raise ValueError(f"Input feature dimension mismatch: Expected {self.n_input}, got {X.shape[1]}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        
        self.cost_history = []
        print(f"Starting MLP training: {self.n_iters} iterations, lr={self.lr}, hidden_nodes={self.n_hidden}, activation={self.activation_type}, alpha={self.alpha}...")

        for i in range(self.n_iters):
            # Forward propagation
            A2, cache = self._forward_propagation(X)

            # Compute cost
            cost = self._compute_cost(A2, Y)
            self.cost_history.append(cost)

            # Backward propagation
            grads = self._backward_propagation(cache, X, Y)

            # Update parameters
            self._update_parameters(grads)

            # Print cost periodically
            if self.verbose and (i % max(1, self.n_iters // 10) == 0 or i == self.n_iters - 1):
                print(f"Iteration {i}: Cost = {cost:.6f}")

        print(f"Training complete. Final Cost = {self.cost_history[-1]:.6f}")

    def predict_proba(self, X):
       """Predict probability of class 1."""
       if self.W1 is None: raise RuntimeError("Model has not been fitted yet.")
       X = np.array(X)
       if X.shape[1] != self.n_input:
           raise ValueError(f"Input feature dimension mismatch: Expected {self.n_input}, got {X.shape[1]}")

       A2, _ = self._forward_propagation(X)
       return A2.flatten() # Return as a 1D array of probabilities

    def predict(self, X, threshold=0.5):
        """Predict class labels (0 or 1)."""
        print(f"Predicting labels for {len(X)} samples using MLP...")
        probabilities = self.predict_proba(X)
        predictions = (probabilities > threshold).astype(int)
        return predictions