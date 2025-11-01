import numpy as np
from collections import Counter

# --- Node Class (for Regressor) ---
class Node:
    """Represents a node in the decision tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None, var_reduction=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        # Value is now a continuous number (the average), not a class label
        self.value = value
        self.var_reduction = var_reduction # How much this split reduced variance

    def is_leaf_node(self):
        """Check if the node is a leaf."""
        return self.value is not None

# --- Decision Tree Regressor Class ---
class DecisionTreeRegressor:
    """Decision Tree Regressor using Variance Reduction from scratch."""
    
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None, min_samples_leaf=1, random_state=None):
        if min_samples_split <= 1:
            raise ValueError("min_samples_split must be >= 2.")
        if max_depth <= 0:
             raise ValueError("max_depth must be positive.")
        
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self._tree_n_features = None 
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state) 
        self.root = None

    def fit(self, X, y):
        """Build the decision tree."""
        X = np.array(X)
        y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        if X.shape[0] < self.min_samples_split:
             self.root = Node(value=self._mean_value(y)) # Leaf value is the mean
             self._tree_n_features = X.shape[1]
             return

        n_total_features = X.shape[1]
        self._tree_n_features = n_total_features
        self.n_feats = n_total_features if self.n_feats is None or self.n_feats >= n_total_features else self.n_feats
        self.rng = np.random.RandomState(self.random_state)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """Recursively build the tree node by node."""
        n_samples, n_features_in_data = X.shape
        
        # Stop if min samples, max depth, or node is "pure" (all y values are the same)
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            leaf_value = self._mean_value(y)
            return Node(value=leaf_value)

        feat_idxs = self.rng.choice(n_features_in_data, self.n_feats, replace=False)
        
        # Find best split based on variance reduction
        best_feat, best_thresh, best_gain = self._best_criteria(X, y, feat_idxs)

        # Stop if no split provides positive gain
        if best_gain <= 0:
            leaf_value = self._mean_value(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        # Stop if a split results in an empty or too-small child
        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            leaf_value = self._mean_value(y)
            return Node(value=leaf_value)

        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left_child, right_child, var_reduction=best_gain)

    def _best_criteria(self, X, y, feat_idxs):
        """Find the best split (feature index, threshold) by maximizing Variance Reduction."""
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # Use variance reduction instead of Gini gain
                gain = self._variance_reduction(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh, best_gain

    def _variance(self, y):
        """Calculate variance of a set of values."""
        if len(y) == 0: return 0
        return np.var(y)

    def _variance_reduction(self, y, X_column, split_thresh):
        """Calculate Variance Reduction for a potential split."""
        parent_variance = self._variance(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0: return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        
        var_l, var_r = self._variance(y[left_idxs]), self._variance(y[right_idxs])
        weighted_child_variance = (n_l / n) * var_l + (n_r / n) * var_r
        
        var_reduction = parent_variance - weighted_child_variance
        return var_reduction

    def _split(self, X_column, split_thresh):
        """Split a feature column into two arrays of indices based on a threshold."""
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _mean_value(self, y):
        """Find the mean value of a set (for leaf node prediction)."""
        if len(y) == 0: return 0 
        return np.mean(y)

    def predict(self, X):
        """Predict continuous values for new data by traversing the tree."""
        if self.root is None: raise RuntimeError("Model has not been fitted yet.")
        X = np.array(X)
        if self._tree_n_features is not None and X.shape[1] != self._tree_n_features:
             raise ValueError(f"Input feature dimension mismatch: Expected {self._tree_n_features}, got {X.shape[1]}")
        
        predictions = np.array([self._traverse_tree(x, self.root) for x in X])
        return predictions

    def _traverse_tree(self, x, node):
        """Recursively navigate the tree for a single data point's prediction."""
        if node.is_leaf_node():
            return node.value # Returns the average value stored at the leaf
        
        if node.feature_idx >= len(x):
             print(f"Warning: Feature index {node.feature_idx} out of bounds for sample.")
             return 0

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# --- Random Forest Regressor Class ---
class RandomForestRegressor:
    """Random Forest Regressor from scratch (uses the DecisionTreeRegressor above)."""
    
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None, min_samples_leaf=1, random_state=None):
        if n_trees <= 0:
            raise ValueError("n_trees must be positive.")
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []
        self._forest_n_features = None 
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state) 
        

    def fit(self, X, y):
        """Train the random forest by building multiple decision trees."""
        X = np.array(X)
        y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        self.trees = []
        n_samples, n_features_total = X.shape
        self._forest_n_features = n_features_total 

        num_feats_for_tree = self.n_feats
        if num_feats_for_tree is None:
             num_feats_for_tree = int(np.sqrt(n_features_total))
             num_feats_for_tree = max(1, min(num_feats_for_tree, n_features_total)) 

        self.rng = np.random.RandomState(self.random_state)
        
        print(f"Fitting Random Forest Regressor: {self.n_trees} trees, max_depth={self.max_depth}, min_split={self.min_samples_split}, features_per_split={num_feats_for_tree}...")
        for i in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            # Use DecisionTreeRegressor
            tree = DecisionTreeRegressor(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=num_feats_for_tree,
                min_samples_leaf=self.min_samples_leaf,
                
                # --- THIS IS THE FIX ---
                # Use 2**31 - 1 (max for signed 32-bit int)
                random_state=self.rng.randint(0, 2**31 - 1) 
            )

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        print(f"Random Forest Regressor fitting complete. ({self.n_trees} trees fitted)")


    def _bootstrap_sample(self, X, y):
        """Create a random sample of the data with replacement."""
        n_samples = X.shape[0]
        idxs = self.rng.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def _mean_value(self, y_predictions):
        """Calculate the mean of predictions from all trees."""
        if y_predictions.size == 0:
             print("Warning: Received empty array for averaging.")
             return 0 
        return np.mean(y_predictions)

    def predict(self, X):
        """Predict values by averaging predictions from all trees."""
        if not self.trees:
             raise RuntimeError("Model has not been fitted yet.")
        X = np.array(X)
        if self._forest_n_features is not None and X.shape[1] != self._forest_n_features:
             raise ValueError(f"Input feature dimension mismatch: Expected {self._forest_n_features}, got {X.shape[1]}")

        print(f"Predicting labels for {X.shape[0]} samples using Random Forest Regressor ({self.n_trees} trees)...")
        
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds_transposed = tree_preds.T

        # For each sample, calculate the mean of all its tree predictions
        y_pred = [self._mean_value(sample_preds) for sample_preds in tree_preds_transposed]

        return np.array(y_pred)