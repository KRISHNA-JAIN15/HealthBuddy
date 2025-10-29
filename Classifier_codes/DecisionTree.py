import numpy as np
from collections import Counter

class Node:
    """Represents a node in the decision tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None, info_gain=None):
        self.feature_idx = feature_idx # Index of feature to split on
        self.threshold = threshold     # Threshold for the split
        self.left = left               # Left child node (samples <= threshold)
        self.right = right             # Right child node (samples > threshold)
        self.value = value             # Class label if it's a leaf node
        self.info_gain = info_gain     # Information gain achieved by this split (optional, for analysis)

    def is_leaf_node(self):
        """Check if the node is a leaf."""
        return self.value is not None

class DecisionTreeClassifier:
    """Decision Tree Classifier using Gini Impurity from scratch."""
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        if min_samples_split <= 1:
            raise ValueError("min_samples_split must be >= 2.")
        if max_depth <= 0:
             raise ValueError("max_depth must be positive.")
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats # Number of features to consider at each split (for Random Forest)
        self.root = None

    def fit(self, X, y):
        """Build the decision tree."""
        X = np.array(X)
        y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        if X.shape[0] < self.min_samples_split:
            raise ValueError(f"Number of samples ({X.shape[0]}) is less than min_samples_split ({self.min_samples_split}).")

        # Determine number of features to use at each split:
        # If n_feats is None or >= total features, use all (standard DT).
        # Otherwise, use specified subset (for Random Forest).
        n_total_features = X.shape[1]
        self.n_feats = n_total_features if self.n_feats is None or self.n_feats >= n_total_features else self.n_feats

        print(f"Fitting Decision Tree: max_depth={self.max_depth}, min_split={self.min_samples_split}, features_per_split={self.n_feats}...")
        self.root = self._grow_tree(X, y)
        print("Tree fitting complete.")

    def _grow_tree(self, X, y, depth=0):
        """Recursively build the tree node by node."""
        n_samples, n_features_in_data = X.shape
        n_labels = len(np.unique(y))

        # --- Stopping criteria ---
        # 1. Max depth reached
        # 2. Node is pure (only one class)
        # 3. Not enough samples to split further
        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value) # Create a leaf node

        # --- Find the best split ---
        # Randomly select a subset of feature indices without replacement
        feat_idxs = np.random.choice(n_features_in_data, self.n_feats, replace=False)

        # Find the best feature index and threshold among the selected features
        best_feat, best_thresh, best_gain = self._best_criteria(X, y, feat_idxs)

        # --- Second Stopping criteria ---
        # 4. No split provides any information gain (e.g., all features identical, or gain is <= 0)
        if best_gain <= 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # --- Split the data and recurse ---
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        # Handle potential empty splits (rare, but possible with certain data distributions)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Recursively build the left and right subtrees
        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        # Return the internal node representing the best split found
        return Node(best_feat, best_thresh, left_child, right_child, info_gain=best_gain)

    def _best_criteria(self, X, y, feat_idxs):
        """Find the best split (feature index, threshold) by maximizing Gini gain."""
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            # Consider unique values in the column as potential split thresholds
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # Calculate Gini gain for splitting on this feature at this threshold
                gain = self._gini_gain(y, X_column, threshold)

                # Update if this split is better than the current best
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh, best_gain

    def _gini_impurity(self, y):
        """Calculate Gini impurity for a set of labels. Gini = 1 - sum(p_i^2)."""
        n_samples = len(y)
        if n_samples == 0:
            return 0
        # Count occurrences of each class label
        _, counts = np.unique(y, return_counts=True)
        # Calculate probabilities
        probabilities = counts / n_samples
        # Calculate Gini impurity
        gini = 1.0 - np.sum(probabilities**2)
        return gini

    def _gini_gain(self, y, X_column, split_thresh):
        """Calculate Information Gain based on Gini impurity reduction."""
        # Parent Gini impurity
        parent_gini = self._gini_impurity(y)

        # Generate split based on threshold
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        # If split results in empty child, gain is 0 (invalid split)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate weighted average Gini impurity of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        gini_l, gini_r = self._gini_impurity(y[left_idxs]), self._gini_impurity(y[right_idxs])
        weighted_child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r

        # Gini gain = Parent Gini - Weighted Child Gini
        gini_gain = parent_gini - weighted_child_gini
        return gini_gain

    def _split(self, X_column, split_thresh):
        """Split a feature column into two arrays of indices based on a threshold."""
        # Indices where feature value is less than or equal to threshold
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        # Indices where feature value is greater than threshold
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        """Find the most frequent label in a set (for leaf node prediction)."""
        if len(y) == 0:
            # This case should be rare if min_samples_split >= 2
            print("Warning: Trying to find most common label in empty set.")
            return 0 # Or handle appropriately
        counter = Counter(y)
        # Returns the most common class label
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """Predict class labels for new data by traversing the tree."""
        if self.root is None:
             raise RuntimeError("Model has not been fitted yet.")
        X = np.array(X)
        if X.shape[1] != self._get_num_features(self.root):
             # Basic check, assumes root node isn't a leaf immediately
              num_expected = self._get_num_features(self.root)
              if num_expected is not None:
                   raise ValueError(f"Input feature dimension mismatch: Expected {num_expected}, got {X.shape[1]}")
              else: # Root is leaf
                   print("Warning: Tree root is a leaf node. Prediction will be constant.")

        print(f"Predicting labels for {X.shape[0]} samples using Decision Tree...")
        predictions = np.array([self._traverse_tree(x, self.root) for x in X])
        # Handle cases where traversal might return None (shouldn't happen with proper fit)
        if np.any(predictions == None):
             print("Warning: Some predictions resulted in None. Replacing with default (0).")
             predictions[predictions == None] = 0 # Fallback
        return predictions.astype(int) # Ensure integer output


    def _traverse_tree(self, x, node):
        """Recursively navigate the tree for a single data point's prediction."""
        if node is None: # Should not happen in a fitted tree unless root was None
             print("Error: Encountered None node during traversal.")
             return None # Or raise error
        if node.is_leaf_node():
            return node.value # Reached a leaf, return its class label

        # Decide whether to go left or right based on the split criteria
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _get_num_features(self, node):
        """Helper to find the expected number of features (for predict check)."""
        if node is None or node.is_leaf_node():
            return None # Cannot determine from leaf or None
        # Go down left/right until we find an internal node or run out
        q = [node]
        while q:
            curr = q.pop(0)
            if not curr.is_leaf_node():
                # We need to know how many features the data *had* when this node was made.
                # This simple implementation doesn't store that.
                # A more robust check would involve storing n_features_in at fit time.
                # For now, we return None, making the check in predict less strict.
                return None # Cannot reliably determine expected features easily after fitting.
            if curr.left: q.append(curr.left)
            if curr.right: q.append(curr.right)
        return None