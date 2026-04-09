import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, class_counts=None):
        self.feature = feature  # Feature index to split on
        self.threshold = threshold  # Threshold for split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Predicted class (for leaf nodes)
        self.class_counts = class_counts  # Class distribution for probability estimation

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.root = None
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self.root = self._grow_tree(X, y, sample_weight, depth=0)

    def _grow_tree(self, X, y, sample_weight, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                n_classes == 1):
            leaf_value, class_counts = self._most_common_label(y, sample_weight)
            return Node(value=leaf_value, class_counts=class_counts)

        # Find best split
        best_feature, best_threshold = self._best_split(X, y, sample_weight, n_features)

        if best_feature is None:  # No valid split found
            leaf_value, class_counts = self._most_common_label(y, sample_weight)
            return Node(value=leaf_value, class_counts=class_counts)

        # Split data
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs

        if sum(left_idxs) == 0 or sum(right_idxs) == 0:
            leaf_value, class_counts = self._most_common_label(y, sample_weight)
            return Node(value=leaf_value, class_counts=class_counts)

        # Create child nodes
        left = self._grow_tree(X[left_idxs], y[left_idxs], sample_weight[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], sample_weight[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, sample_weight, n_features):
        best_gain = -1
        best_feature = None
        best_threshold = None

        # Randomly select features to consider (Random Forest feature)
        feature_indices = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:  # Removed thresholds[::2] for better accuracy
                left_idxs = X[:, feature] <= threshold
                right_idxs = ~left_idxs

                if sum(left_idxs) < self.min_samples_split or sum(right_idxs) < self.min_samples_split:
                    continue

                gain = self._information_gain(y, left_idxs, right_idxs, sample_weight)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, left_idxs, right_idxs, sample_weight):
        parent_gini = self._gini(y, sample_weight)
        n = len(y)
        n_left = sum(left_idxs)
        n_right = n - n_left

        if n_left == 0 or n_right == 0:
            return 0

        child_gini = (n_left / n) * self._gini(y[left_idxs], sample_weight[left_idxs]) + \
                     (n_right / n) * self._gini(y[right_idxs], sample_weight[right_idxs])

        return parent_gini - child_gini

    def _gini(self, y, sample_weight):
        _, counts = np.unique(y, return_counts=True)
        weighted_counts = np.zeros_like(counts, dtype=float)
        total_weight = np.sum(sample_weight)
        if total_weight == 0:  # Avoid division by zero
            return 1.0
        for i, label in enumerate(np.unique(y)):
            weighted_counts[i] = np.sum(sample_weight[y == label])
        weighted_counts /= total_weight
        return 1 - np.sum(weighted_counts ** 2)

    def _most_common_label(self, y, sample_weight):
        labels, counts = np.unique(y, return_counts=True)
        weighted_counts = np.zeros_like(counts, dtype=float)
        for i, label in enumerate(labels):
            weighted_counts[i] = np.sum(sample_weight[y == label])
        most_common = labels[np.argmax(weighted_counts)]
        class_counts = {label: weighted_counts[i] for i, label in enumerate(labels)}
        return most_common, class_counts

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict_proba(self, X):
        # For binary classification, return probability of class 1
        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            predictions[i] = self._traverse_tree_proba(x, self.root)
        return predictions

    def _traverse_tree_proba(self, x, node):
        if node.class_counts is not None:
            total = sum(node.class_counts.values())
            if total == 0:
                return 0.0
            return node.class_counts.get(1, 0) / total  # Probability of class 1
        if x[node.feature] <= node.threshold:
            return self._traverse_tree_proba(x, node.left)
        return self._traverse_tree_proba(x, node.right)