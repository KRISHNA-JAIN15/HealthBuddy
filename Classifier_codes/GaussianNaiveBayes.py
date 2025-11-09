import numpy as np

class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier from scratch."""
    
    def __init__(self, var_smoothing=1e-9):
        """
        Initialize the classifier.
        
        Parameters:
        var_smoothing (float): Portion of the largest variance of all features 
                               that is added to variances for calculation stability.
        """
        self.classes = None
        self.mean = None       # Shape: (n_classes, n_features)
        self.var = None        # Shape: (n_classes, n_features)
        self.priors = None     # Shape: (n_classes,)
        self.var_smoothing = var_smoothing
        self.epsilon_ = None    # Will store the calculated smoothing term

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
        
        if n_samples > 0:
            max_data_var = np.var(X, axis=0).max()
            self.epsilon_ = self.var_smoothing * max_data_var
        else:
            self.epsilon_ = self.var_smoothing # Fallback
        
        # Add a small floor to epsilon_ to prevent it from being exactly zero
        # if max_data_var was zero (e.g., all features are constant)
        if self.epsilon_ == 0:
            self.epsilon_ = self.var_smoothing

        # Calculate parameters for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c] # Get samples belonging to the current class 'c'
            
            if X_c.shape[0] == 0:
                print(f"Warning: No samples found for class {c}.")
                # Assign 0 mean, and just the smoothing term for variance
                self.mean[idx, :] = 0
                self.var[idx, :] = self.epsilon_ 
                self.priors[idx] = 0 # Prior is 0
                continue

            # Calculate mean and variance for each feature within this class
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            
            # Calculate prior probability P(c)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

        # --- CHANGE: Add the adaptive epsilon AFTER calculating all variances ---
        self.var += self.epsilon_
        
        # Normalize priors if they don't sum to 1 (e.g., due to empty classes)
        p_sum = np.sum(self.priors)
        if p_sum == 0:
            print("Warning: All classes were empty. Assigning uniform priors.")
            self.priors = np.full(n_classes, 1.0 / n_classes)
        elif not np.isclose(p_sum, 1.0):
             print("Warning: Priors do not sum to 1. Normalizing.")
             self.priors /= p_sum

        print("Fitting complete.")

    def predict(self, X):
        """
        Predict class labels for new data.
        This method is vectorized and replaces _predict_single and _pdf.
        """
        if self.classes is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_test = np.array(X)
        if X_test.shape[1] != self.mean.shape[1]:
            raise ValueError(f"Input feature dimension mismatch: Expected {self.mean.shape[1]}, got {X_test.shape[1]}")

        print(f"Predicting labels for {X_test.shape[0]} samples using GNB...")

        n_samples = X_test.shape[0]
        n_classes = len(self.classes)
        
        # Store log posteriors for each sample, for each class
        log_posteriors = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            # 1. Get parameters for this class
            mean = self.mean[idx]
            var = self.var[idx] # Already includes epsilon_
            
            # 2. Calculate log prior: log(P(C))
            # We add epsilon_ before the log to prevent -inf if prior is 0
            log_prior = np.log(self.priors[idx] + self.epsilon_)
            
            # 3. Calculate log likelihood: sum(log(P(x_i | C)))
            # This is done for all samples at once.
            
            # Log PDF formula: -0.5 * log(2*pi*var) - 0.5 * ((x-mean)^2 / var)
            # We calculate this for all features, then sum them up.
            
            # Term 1: -0.5 * sum(log(2*pi*var_i)) over all features i
            # This is a single scalar value.
            term1 = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            
            # Term 2: -0.5 * sum(((x_i - mean_i)^2 / var_i)) over all features i
            # This is calculated for ALL samples at once.
            # (X_test - mean) -> shape (n_samples, n_features)
            # ... / var       -> shape (n_samples, n_features)
            # np.sum(..., axis=1) -> shape (n_samples,)
            term2_numerator = (X_test - mean) ** 2
            term2_denominator = 2 * var
            term2 = -np.sum(term2_numerator / term2_denominator, axis=1)

            log_likelihood = term1 + term2
            
            # 4. Calculate log posterior
            # log_prior is scalar, log_likelihood is (n_samples,)
            # Broadcasting adds them correctly.
            log_posteriors[:, idx] = log_prior + log_likelihood

        # Find the class with the highest log posterior for each sample
        indices = np.argmax(log_posteriors, axis=1)
        return self.classes[indices]
