"""
src/decision_tree.py
--------------------
Decision Tree classifier built from scratch using NumPy.
Uses Gini impurity for split selection and supports
sample weights for class imbalance handling.
"""

import numpy as np


class Node:
    """Represents a single node in the decision tree."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        class_counts=None,
    ):
        self.feature = feature          # Feature index to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left child (feature <= threshold)
        self.right = right              # Right child (feature > threshold)
        self.value = value              # Predicted class (leaf nodes only)
        self.class_counts = class_counts  # Weighted class distribution (for predict_proba)


class DecisionTree:
    """
    Binary decision tree trained via recursive Gini-based splitting.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.
    min_samples_split : int
        Minimum samples required to attempt a split.
    random_state : int or None
        Seed for feature subsampling (used when embedded in RandomForest).
    """

    def __init__(self, max_depth=10, min_samples_split=2, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.root = None
        if random_state is not None:
            np.random.seed(random_state)

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(self, X, y, sample_weight=None):
        """Train the tree on feature matrix X and labels y."""
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self.root = self._grow_tree(X, y, sample_weight, depth=0)

    def predict(self, X):
        """Return predicted class labels for each row in X."""
        return np.array([self._traverse(x, self.root) for x in X])

    def predict_proba(self, X):
        """Return probability of class 1 for each row in X."""
        return np.array([self._traverse_proba(x, self.root) for x in X])

    # ── Tree building ────────────────────────────────────────────────────────

    def _grow_tree(self, X, y, sample_weight, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            value, counts = self._leaf_value(y, sample_weight)
            return Node(value=value, class_counts=counts)

        best_feature, best_threshold = self._best_split(X, y, sample_weight, n_features)

        if best_feature is None:
            value, counts = self._leaf_value(y, sample_weight)
            return Node(value=value, class_counts=counts)

        left_mask  = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            value, counts = self._leaf_value(y, sample_weight)
            return Node(value=value, class_counts=counts)

        left  = self._grow_tree(X[left_mask],  y[left_mask],  sample_weight[left_mask],  depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, sample_weight, n_features):
        """Scan a random subset of features and all thresholds; return best split."""
        best_gain = -1
        best_feature = best_threshold = None

        # Random feature subset (sqrt heuristic — standard for random forests)
        feature_indices = np.random.choice(
            n_features, int(np.sqrt(n_features)), replace=False
        )

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask  = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue

                gain = self._information_gain(y, left_mask, right_mask, sample_weight)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold

        return best_feature, best_threshold

    # ── Impurity helpers ─────────────────────────────────────────────────────

    def _information_gain(self, y, left_mask, right_mask, w):
        n = len(y)
        n_l, n_r = left_mask.sum(), right_mask.sum()
        parent = self._gini(y, w)
        child  = (n_l / n) * self._gini(y[left_mask], w[left_mask]) \
               + (n_r / n) * self._gini(y[right_mask], w[right_mask])
        return parent - child

    def _gini(self, y, w):
        total = w.sum()
        if total == 0:
            return 1.0
        impurity = 1.0
        for label in np.unique(y):
            p = w[y == label].sum() / total
            impurity -= p ** 2
        return impurity

    def _leaf_value(self, y, w):
        labels = np.unique(y)
        weighted = {label: w[y == label].sum() for label in labels}
        value = max(weighted, key=weighted.get)
        return value, weighted

    # ── Inference helpers ────────────────────────────────────────────────────

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def _traverse_proba(self, x, node):
        if node.class_counts is not None:
            total = sum(node.class_counts.values())
            return node.class_counts.get(1, 0) / total if total > 0 else 0.0
        if x[node.feature] <= node.threshold:
            return self._traverse_proba(x, node.left)
        return self._traverse_proba(x, node.right)
