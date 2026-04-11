"""
src/random_forest.py
--------------------
Random Forest classifier built from scratch on top of the custom
DecisionTree. Combines bootstrap sampling, feature subsampling
(handled inside DecisionTree), and majority-vote aggregation.
"""

import numpy as np
from src.decision_tree import DecisionTree


class RandomForest:
    """
    Ensemble of DecisionTree classifiers trained via bootstrap sampling.

    Parameters
    ----------
    n_trees : int
        Number of trees in the ensemble.
    max_depth : int
        Maximum depth of each tree.
    min_samples_split : int
        Minimum samples required to split a node.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, n_trees=50, max_depth=10, min_samples_split=2, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        if random_state is not None:
            np.random.seed(random_state)

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Train the forest.

        Class weights are computed automatically to handle imbalance
        (fraud class is upweighted proportionally to its rarity).
        """
        self.trees = []
        n_samples = X.shape[0]

        # Compute inverse-frequency class weights
        classes, counts = np.unique(y, return_counts=True)
        freq = dict(zip(classes, counts))
        class_weights = {c: n_samples / (len(classes) * freq[c]) for c in classes}
        sample_weight = np.array([class_weights[yi] for yi in y])

        for i in range(self.n_trees):
            # Bootstrap sample (with replacement)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            w_boot = sample_weight[indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
            tree.fit(X_boot, y_boot, w_boot)
            self.trees.append(tree)

    def predict(self, X):
        """Majority vote across all trees."""
        votes = np.array([tree.predict(X) for tree in self.trees])  # (n_trees, n_samples)
        return np.apply_along_axis(
            lambda col: np.bincount(col.astype(int)).argmax(), axis=0, arr=votes
        )

    def predict_proba(self, X):
        """
        Average fraud probability across all trees.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Columns: [P(legitimate), P(fraud)]
        """
        proba_fraud = np.mean(
            [tree.predict_proba(X) for tree in self.trees], axis=0
        )
        return np.column_stack([1 - proba_fraud, proba_fraud])
