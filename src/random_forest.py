import numpy as np
from src.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        
        # Compute class weights for imbalanced data
        classes, counts = np.unique(y, return_counts=True)
        class_weights = {0: 1.0, 1: counts[0] / counts[1]}  # Weight fraud class higher
        sample_weight = np.array([class_weights[yi] for yi in y])
        
        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            sample_weight_sample = sample_weight[indices]
            
            # Train a decision tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )
            tree.fit(X_sample, y_sample, sample_weight_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Majority voting
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    def predict_proba(self, X):
        # Average probabilities
        proba = np.mean([tree.predict_proba(X) for tree in self.trees], axis=0)
        return proba