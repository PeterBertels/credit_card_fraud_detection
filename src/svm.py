"""
src/svm.py
----------
Support Vector Machine (SVM) classifier built from scratch.

Implements a soft-margin linear SVM using sub-gradient descent
on the hinge loss objective:

    L(w, b) = (1/2)||w||² + C * Σ max(0, 1 - yᵢ(w·xᵢ + b))

where C controls the trade-off between margin width and
classification error. Labels are internally mapped to {-1, +1}.

This is a pure NumPy implementation — no scikit-learn used.
"""

import numpy as np


class SVM:
    """
    Soft-margin linear SVM trained via sub-gradient descent.

    Parameters
    ----------
    C : float
        Regularization parameter. Smaller = wider margin, more tolerance
        for misclassifications. Larger = tighter fit to training data.
    learning_rate : float
        Step size for gradient descent updates.
    n_epochs : int
        Number of full passes over the training data.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, C=1.0, learning_rate=0.001, n_epochs=1000, random_state=None):
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.w = None       # Weight vector
        self.b = None       # Bias term
        self.classes_ = None

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Train the SVM on feature matrix X and binary labels y (0 or 1).

        Internally maps labels to {-1, +1} for the hinge loss formulation.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        # Map {0, 1} → {-1, +1}
        y_svm = np.where(y == 1, 1, -1).astype(float)

        # Initialise weights near zero
        self.w = np.zeros(n_features)
        self.b = 0.0

        # Class weights to handle imbalance (mirror sklearn's balanced mode)
        classes, counts = np.unique(y, return_counts=True)
        freq = dict(zip(classes, counts))
        sample_weight = np.array(
            [n_samples / (len(classes) * freq[yi]) for yi in y]
        )

        # Sub-gradient descent
        for epoch in range(self.n_epochs):
            # Shuffle each epoch for stochastic behaviour
            indices = np.random.permutation(n_samples)
            for i in indices:
                xi, yi, wi = X[i], y_svm[i], sample_weight[i]
                margin = yi * (np.dot(self.w, xi) + self.b)

                if margin >= 1:
                    # Correctly classified with sufficient margin — only regularise
                    self.w -= self.learning_rate * self.w
                else:
                    # Hinge loss active — update toward correct side
                    self.w -= self.learning_rate * (self.w - self.C * wi * yi * xi)
                    self.b += self.learning_rate * self.C * wi * yi

        return self

    def predict(self, X):
        """
        Predict binary class labels for each row in X.

        Returns
        -------
        np.ndarray of int (0 or 1)
        """
        scores = np.dot(X, self.w) + self.b
        # Map {-1, +1} → {0, 1}
        return np.where(scores >= 0, 1, 0)

    def predict_proba(self, X):
        """
        Approximate class probabilities using the decision function score.

        Note: SVMs don't natively produce probabilities. We use a sigmoid
        transformation of the margin distance as a calibrated approximation.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Columns: [P(legitimate), P(fraud)]
        """
        scores = np.dot(X, self.w) + self.b
        proba_fraud = self._sigmoid(scores)
        return np.column_stack([1 - proba_fraud, proba_fraud])

    def decision_function(self, X):
        """
        Return the raw signed distance from the decision boundary.
        Positive = fraud side, negative = legitimate side.
        """
        return np.dot(X, self.w) + self.b

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x):
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x)),
        )
