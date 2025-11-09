# import numpy as np
# from collections import Counter

# # --- Decision Stump Node (Simpler than full DT Node) ---
# class StumpNode:
#     def __init__(self, feature_idx=None, threshold=None, left_val=None, right_val=None, *, value=None):
#         self.feature_idx = feature_idx # Feature to split on
#         self.threshold = threshold     # Threshold value
#         self.left_val = left_val       # Prediction if feature <= threshold
#         self.right_val = right_val     # Prediction if feature > threshold
#         self.value = value             # Prediction if it's a leaf (no split possible/needed)

#     def is_leaf(self):
#         return self.value is not None

# # --- Decision Stump Classifier (Weak Learner) ---
# class DecisionStump:
#     """A Decision Tree with max_depth=1, used as a weak learner for AdaBoost."""
#     def __init__(self):
#         self.root = None
#         self._tree_n_features = None # Store expected features for prediction check


#     def fit(self, X, y, sample_weights=None):
#         """Find the best single split based on weighted Gini impurity or error."""
#         X = np.array(X)
#         y = np.array(y)
#         n_samples, n_features = X.shape
#         self._tree_n_features = n_features

#         # If no sample weights provided, assume uniform weights
#         if sample_weights is None:
#             sample_weights = np.ones(n_samples) / n_samples
#         else:
#              # Ensure weights sum to 1 (important for weighted impurity calculations)
#              sample_weights = np.array(sample_weights) / np.sum(sample_weights)


#         best_gain = -float('inf')
#         best_feat, best_thresh = None, None
#         best_left_val, best_right_val = None, None

#         # Check if the node is already pure or has too few samples (optional for stump)
#         if len(np.unique(y)) == 1:
#              self.root = StumpNode(value=self._weighted_majority_vote(y, sample_weights))
#              return

#         # Iterate through all features and potential thresholds
#         for feat_idx in range(n_features):
#             X_column = X[:, feat_idx]
#             thresholds = np.unique(X_column)

#             for threshold in thresholds:
#                 # Split data based on threshold
#                 left_idxs = np.argwhere(X_column <= threshold).flatten()
#                 right_idxs = np.argwhere(X_column > threshold).flatten()

#                 # Skip invalid splits
#                 if len(left_idxs) == 0 or len(right_idxs) == 0:
#                     continue

#                 # Calculate weighted Gini gain for this split
#                 gain = self._weighted_gini_gain(y, sample_weights, left_idxs, right_idxs)

#                 if gain > best_gain:
#                     best_gain = gain
#                     best_feat = feat_idx
#                     best_thresh = threshold
#                     # Determine predictions for left/right nodes based on weighted majority
#                     best_left_val = self._weighted_majority_vote(y[left_idxs], sample_weights[left_idxs])
#                     best_right_val = self._weighted_majority_vote(y[right_idxs], sample_weights[right_idxs])

#         # If a split was found, create an internal node
#         if best_feat is not None:
#              self.root = StumpNode(feature_idx=best_feat, threshold=best_thresh,
#                                    left_val=best_left_val, right_val=best_right_val)
#         else:
#              # No split improved purity, create a leaf node
#              self.root = StumpNode(value=self._weighted_majority_vote(y, sample_weights))


#     def _weighted_gini_impurity(self, y, sample_weights):
#         """Calculate Gini impurity considering sample weights."""
#         total_weight = np.sum(sample_weights)
#         if total_weight == 0: return 0

#         impurity = 1.0
#         classes = np.unique(y)
#         for cls in classes:
#             # Sum weights of samples belonging to this class
#             weight_cls = np.sum(sample_weights[y == cls])
#             prob_cls = weight_cls / total_weight
#             impurity -= prob_cls**2
#         return impurity

#     def _weighted_gini_gain(self, y, sample_weights, left_idxs, right_idxs):
#         """Calculate Gini gain using weighted impurities."""
#         parent_impurity = self._weighted_gini_impurity(y, sample_weights)

#         weight_left = np.sum(sample_weights[left_idxs])
#         weight_right = np.sum(sample_weights[right_idxs])
#         total_weight = weight_left + weight_right

#         if total_weight == 0: return 0 # Avoid division by zero

#         impurity_left = self._weighted_gini_impurity(y[left_idxs], sample_weights[left_idxs])
#         impurity_right = self._weighted_gini_impurity(y[right_idxs], sample_weights[right_idxs])

#         weighted_child_impurity = (weight_left / total_weight) * impurity_left + \
#                                   (weight_right / total_weight) * impurity_right

#         gain = parent_impurity - weighted_child_impurity
#         return gain

#     def _weighted_majority_vote(self, y, sample_weights):
#         """Determine the class label with the highest total weight."""
#         if len(y) == 0: return 0 # Default if empty
#         classes = np.unique(y)
#         max_weight = -1
#         major_class = classes[0] # Default
#         for cls in classes:
#              weight_cls = np.sum(sample_weights[y == cls])
#              if weight_cls > max_weight:
#                   max_weight = weight_cls
#                   major_class = cls
#         return major_class


#     def predict(self, X):
#         """Predict labels for new data using the single split rule."""
#         if self.root is None: raise RuntimeError("Stump has not been fitted yet.")
#         X = np.array(X)
#         if self._tree_n_features is not None and X.shape[1] != self._tree_n_features:
#              raise ValueError(f"Input feature dimension mismatch: Expected {self._tree_n_features}, got {X.shape[1]}")

#         # Apply prediction rule for each sample
#         predictions = np.array([self._traverse_stump(x, self.root) for x in X])
#         return predictions.astype(int) # Ensure integer output


#     def _traverse_stump(self, x, node):
#         """Apply the stump's rule to a single sample."""
#         if node.is_leaf():
#             return node.value

#         feature_val = x[node.feature_idx]
#         if feature_val <= node.threshold:
#             return node.left_val
#         else:
#             return node.right_val

# # --- AdaBoost Classifier ---
# class AdaBoost:
#     """AdaBoost classifier using Decision Stumps from scratch (for binary classification)."""
#     def __init__(self, n_estimators=50, learning_rate=1.0):
#         if n_estimators <= 0: raise ValueError("n_estimators must be positive.")
#         self.n_estimators = n_estimators
#         self.learning_rate = learning_rate # Often not used explicitly in basic AdaBoost, but can be for shrinkage
#         self.estimators = [] # List to store weak learners (stumps)
#         self.estimator_weights = [] # List to store alpha weights for each learner
#         self._boost_n_features = None # Store expected features for prediction check


#     def fit(self, X, y):
#         """Train the AdaBoost ensemble."""
#         X = np.array(X)
#         y_input = np.array(y)
#         if X.shape[0] != y_input.shape[0]:
#             raise ValueError("Number of samples in X and y must match.")
#         if len(np.unique(y_input)) != 2:
#             raise ValueError("AdaBoost implementation currently supports binary labels (e.g., 0 and 1).")

#         # Convert labels to {-1, 1} for AdaBoost calculation convenience
#         y_ = np.where(y_input <= 0, -1, 1)
#         n_samples, n_features = X.shape
#         self._boost_n_features = n_features

#         # Initialize sample weights uniformly
#         sample_weights = np.ones(n_samples) / n_samples

#         self.estimators = []
#         self.estimator_weights = []

#         print(f"Starting AdaBoost training with {self.n_estimators} estimators...")
#         for i in range(self.n_estimators):
#             # 1. Train a weak learner (Decision Stump) on weighted data
#             stump = DecisionStump()
#             stump.fit(X, y_, sample_weights) # Pass y_ {-1, 1} and current weights
#             stump_pred = stump.predict(X) # Get predictions {-1, 1} from stump

#             # 2. Calculate weighted error of the stump
#             # Error is the sum of weights of misclassified samples
#             incorrect = (stump_pred != y_)
#             error = np.sum(sample_weights[incorrect])

#             # Add epsilon to avoid division by zero or log(0) if error is 0 or 1
#             error = np.clip(error, 1e-10, 1 - 1e-10)

#             # 3. Calculate estimator weight (alpha)
#             # alpha = 0.5 * log((1 - error) / error)
#             alpha = 0.5 * np.log((1.0 - error) / error)

#             # 4. Update sample weights
#             # Increase weights for misclassified, decrease for correctly classified
#             # w_new = w_old * exp(-alpha * y_true * y_pred)
#             update_factor = np.exp(-alpha * y_ * stump_pred)
#             sample_weights *= update_factor

#             # Normalize sample weights to sum to 1
#             sample_weights /= np.sum(sample_weights)

#             # Store the trained stump and its weight
#             self.estimators.append(stump)
#             self.estimator_weights.append(alpha)

#             if (i + 1) % max(1, self.n_estimators // 10) == 0 or i == self.n_estimators - 1:
#                 print(f"  Estimator {i+1}/{self.n_estimators} fitted. Error={error:.4f}, Alpha={alpha:.4f}")

#              # Optional: Break if error is too high (e.g., > 0.5) or too low (perfect fit)
#             if error >= 0.5:
#                  print(f"  Stopping early: Error ({error:.4f}) >= 0.5 at estimator {i+1}.")
#                  # Remove the last estimator as it's not better than random
#                  if self.estimators: self.estimators.pop()
#                  if self.estimator_weights: self.estimator_weights.pop()
#                  break
#             if error <= 1e-10:
#                  print(f"  Stopping early: Perfect fit achieved at estimator {i+1}.")
#                  break

#         print("AdaBoost training complete.")


#     def predict(self, X):
#         """Predict labels using the weighted vote of all estimators."""
#         if not self.estimators:
#              raise RuntimeError("Model has not been fitted yet.")
#         X = np.array(X)
#         if self._boost_n_features is not None and X.shape[1] != self._boost_n_features:
#              raise ValueError(f"Input feature dimension mismatch: Expected {self._boost_n_features}, got {X.shape[1]}")

#         print(f"Predicting labels for {X.shape[0]} samples using AdaBoost...")

#         # Get weighted predictions from each estimator
#         n_samples = X.shape[0]
#         estimator_preds = np.array([est.predict(X) for est in self.estimators]) # Shape: (n_estimators, n_samples)
#         alphas = np.array(self.estimator_weights) # Shape: (n_estimators,)

#         # Calculate final score: sum(alpha_i * prediction_i) for each sample
#         # Need to reshape alphas for broadcasting: (n_estimators, 1)
#         final_scores = np.dot(alphas.reshape(1, -1), estimator_preds) # Result shape: (1, n_samples)
#         final_scores = final_scores.flatten() # Shape: (n_samples,)

#         # Final prediction is the sign of the aggregated score
#         # Sign gives {-1, 1}, convert back to {0, 1}
#         predictions_boost = np.sign(final_scores)
#         y_predicted = np.where(predictions_boost <= 0, 0, 1) # Map -1 to 0, 1 to 1
#         return y_predicted


import numpy as np
from collections import Counter

# --- Decision Stump Node (Unchanged) ---
class StumpNode:
    """A node for the Decision Stump."""
    def __init__(self, feature_idx=None, threshold=None, left_val=None, right_val=None, *, value=None):
        self.feature_idx = feature_idx # Feature to split on
        self.threshold = threshold   # Threshold value
        self.left_val = left_val     # Prediction if feature <= threshold
        self.right_val = right_val   # Prediction if feature > threshold
        self.value = value           # Prediction if it's a leaf (no split)

    def is_leaf(self):
        """Check if the node is a leaf (a prediction value) or an internal node (a split)."""
        return self.value is not None

# --- Decision Stump Classifier (Refactored) ---
class DecisionStump:
    """
    A Decision Tree with max_depth=1, used as a weak learner for AdaBoost.
    This implementation uses weighted Gini impurity to find the best split.
    """
    def __init__(self):
        self.root = None
        self._tree_n_features = None # Store expected features for prediction check

    def fit(self, X, y, sample_weights=None):
        """Find the best single split based on weighted Gini impurity."""
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self._tree_n_features = n_features

        # If no sample weights provided, assume uniform weights
        if sample_weights is None:
            sample_weights = np.ones(n_samples) / n_samples
        else:
            # Ensure weights sum to 1 (important for weighted impurity calculations)
            total_weight = np.sum(sample_weights)
            if total_weight == 0:
                 # If all weights are zero, re-initialize to uniform
                 sample_weights = np.ones(n_samples) / n_samples
            else:
                 sample_weights = np.array(sample_weights) / total_weight


        best_gain = -float('inf')
        best_feat, best_thresh = None, None
        best_left_val, best_right_val = None, None

        # Check if the node is already pure
        if len(np.unique(y)) == 1:
            self.root = StumpNode(value=self._weighted_majority_vote(y, sample_weights))
            return

        # Iterate through all features and potential thresholds
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # Split data based on threshold
                left_idxs = np.argwhere(X_column <= threshold).flatten()
                right_idxs = np.argwhere(X_column > threshold).flatten()

                # Skip invalid splits (where one side is empty)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                # Calculate weighted Gini gain for this split
                gain = self._weighted_gini_gain(y, sample_weights, left_idxs, right_idxs)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = threshold
                    # Determine predictions for left/right nodes based on weighted majority
                    best_left_val = self._weighted_majority_vote(y[left_idxs], sample_weights[left_idxs])
                    best_right_val = self._weighted_majority_vote(y[right_idxs], sample_weights[right_idxs])

        # If a split was found, create an internal node
        if best_feat is not None:
            self.root = StumpNode(feature_idx=best_feat, threshold=best_thresh,
                                  left_val=best_left_val, right_val=best_right_val)
        else:
            # No split improved purity, create a leaf node
            self.root = StumpNode(value=self._weighted_majority_vote(y, sample_weights))


    def _weighted_gini_impurity(self, y, sample_weights):
        """Calculate Gini impurity considering sample weights."""
        total_weight = np.sum(sample_weights)
        if total_weight == 0: return 0

        impurity = 1.0
        classes = np.unique(y)
        for cls in classes:
            # Sum weights of samples belonging to this class
            weight_cls = np.sum(sample_weights[y == cls])
            prob_cls = weight_cls / total_weight
            impurity -= prob_cls**2
        return impurity

    def _weighted_gini_gain(self, y, sample_weights, left_idxs, right_idxs):
        """Calculate Gini gain using weighted impurities."""
        parent_impurity = self._weighted_gini_impurity(y, sample_weights)

        weight_left = np.sum(sample_weights[left_idxs])
        weight_right = np.sum(sample_weights[right_idxs])
        total_weight = weight_left + weight_right

        if total_weight == 0: return 0 # Avoid division by zero

        impurity_left = self._weighted_gini_impurity(y[left_idxs], sample_weights[left_idxs])
        impurity_right = self._weighted_gini_impurity(y[right_idxs], sample_weights[right_idxs])

        weighted_child_impurity = (weight_left / total_weight) * impurity_left + \
                                  (weight_right / total_weight) * impurity_right

        gain = parent_impurity - weighted_child_impurity
        return gain

    def _weighted_majority_vote(self, y, sample_weights):
        """Determine the class label with the highest total weight."""
        if len(y) == 0: 
            return 0 # Default if empty (should not happen in valid splits)
        
        classes = np.unique(y)
        if len(classes) == 0:
            return 0 # Default if y is empty
            
        max_weight = -1
        major_class = classes[0] # Default
        for cls in classes:
            weight_cls = np.sum(sample_weights[y == cls])
            if weight_cls > max_weight:
                max_weight = weight_cls
                major_class = cls
        return major_class


    def predict(self, X):
        """Predict labels for new data using the single split rule (vectorized)."""
        if self.root is None: raise RuntimeError("Stump has not been fitted yet.")
        X = np.array(X)
        if self._tree_n_features is not None and X.shape[1] != self._tree_n_features:
            raise ValueError(f"Input feature dimension mismatch: Expected {self._tree_n_features}, got {X.shape[1]}")

        if self.root.is_leaf():
            # If root is a leaf, all predictions are the same
            return np.full(X.shape[0], self.root.value, dtype=int)
        else:
            # Apply the split rule to the whole column at once
            feature_idx = self.root.feature_idx
            threshold = self.root.threshold
            left_val = self.root.left_val
            right_val = self.root.right_val
            
            X_column = X[:, feature_idx]
            
            # np.where is the vectorized equivalent of the if/else
            predictions = np.where(X_column <= threshold, left_val, right_val)
            return predictions.astype(int)

# --- AdaBoost Classifier (Refactored) ---
class AdaBoost:
    """AdaBoost classifier using Decision Stumps from scratch (for binary classification)."""
    
    def __init__(self, n_estimators=50, learning_rate=1.0):
        if n_estimators <= 0: raise ValueError("n_estimators must be positive.")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate # Used for shrinkage
        self.estimators = [] # List to store weak learners (stumps)
        self.estimator_weights = [] # List to store alpha weights for each learner
        self._boost_n_features = None # Store expected features for prediction check
        self.classes_ = None # To store original class labels (e.g., [0, 1] or ['A', 'B'])


    def fit(self, X, y):
        """Train the AdaBoost ensemble."""
        X = np.array(X)
        y_input = np.array(y)
        if X.shape[0] != y_input.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        
        # Store the original class labels
        self.classes_ = np.unique(y_input)
        if len(self.classes_) != 2:
            raise ValueError(f"AdaBoost implementation supports binary classification. Found {len(self.classes_)} unique classes.")
        
        # Convert labels to {-1, 1} internally for AdaBoost math
        # self.classes_[0] becomes -1, self.classes_[1] becomes 1
        y_ = np.where(y_input == self.classes_[0], -1, 1)
        # --- End Label Handling ---

        n_samples, n_features = X.shape
        self._boost_n_features = n_features

        # Initialize sample weights uniformly
        sample_weights = np.ones(n_samples) / n_samples

        self.estimators = []
        self.estimator_weights = []

        print(f"Starting AdaBoost training with {self.n_estimators} estimators...")
        for i in range(self.n_estimators):
            # 1. Train a weak learner (Decision Stump) on weighted data
            stump = DecisionStump()
            stump.fit(X, y_, sample_weights) # Pass internal y_ {-1, 1} and current weights
            stump_pred = stump.predict(X) # Get predictions {-1, 1} from stump

            # 2. Calculate weighted error of the stump
            # Error is the sum of weights of misclassified samples
            incorrect = (stump_pred != y_)
            error = np.sum(sample_weights[incorrect])

            # Add epsilon to avoid division by zero or log(0) if error is 0 or 1
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # 3. Calculate estimator weight (alpha)
            # alpha = 0.5 * log((1 - error) / error)
            # Scale alpha by the learning rate (shrinkage)
            alpha = self.learning_rate * (0.5 * np.log((1.0 - error) / error))

            # 4. Update sample weights
            # Increase weights for misclassified, decrease for correctly classified
            # w_new = w_old * exp(-alpha * y_true * y_pred)
            update_factor = np.exp(-alpha * y_ * stump_pred)
            sample_weights *= update_factor

            # Normalize sample weights to sum to 1
            sample_weights /= np.sum(sample_weights)

            # Store the trained stump and its weight
            self.estimators.append(stump)
            self.estimator_weights.append(alpha)

            # Print progress
            if (i + 1) % max(1, self.n_estimators // 10) == 0 or i == self.n_estimators - 1:
                print(f"  Estimator {i+1}/{self.n_estimators} fitted. Error={error:.4f}, Alpha={alpha:.4f}")

            # Optional: Break if error is no better than random
            # (Only if learning_rate is 1.0, otherwise shrinkage might still help)
            if error >= 0.5 and self.learning_rate == 1.0:
                print(f"  Stopping early: Error ({error:.4f}) >= 0.5 at estimator {i+1}.")
                # Remove the last estimator as it's not better than random
                if self.estimators: self.estimators.pop()
                if self.estimator_weights: self.estimator_weights.pop()
                break
            # Optional: Break if a perfect fit is achieved
            if error <= 1e-10:
                print(f"  Stopping early: Perfect fit achieved at estimator {i+1}.")
                break

        print("AdaBoost training complete.")


    def predict(self, X):
        """Predict labels using the weighted vote of all estimators."""
        if not self.estimators:
            raise RuntimeError("Model has not been fitted yet.")
        X = np.array(X)
        if self._boost_n_features is not None and X.shape[1] != self._boost_n_features:
            raise ValueError(f"Input feature dimension mismatch: Expected {self._boost_n_features}, got {X.shape[1]}")

        print(f"Predicting labels for {X.shape[0]} samples using AdaBoost...")

        # Accumulate scores instead of building a large (n_estimators, n_samples) matrix
        n_samples = X.shape[0]
        final_scores = np.zeros(n_samples)
        
        for alpha, stump in zip(self.estimator_weights, self.estimators):
            # stump.predict() is now fast and vectorized
            final_scores += alpha * stump.predict(X) 
        # --- End Change ---

        # Final prediction is the sign of the aggregated score
        # np.sign() will return {-1, 0, 1}
        predictions_boost = np.sign(final_scores)

        # --- Map {-1, 1} back to original classes ---
        # We map scores <= 0 to the first class (e.g., 0)
        # and scores > 0 to the second class (e.g., 1)
        y_predicted = np.where(predictions_boost <= 0, self.classes_[0], self.classes_[1])
        # --- End Change ---

        return y_predicted