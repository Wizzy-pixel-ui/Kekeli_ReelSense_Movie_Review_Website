import numpy as np
from scipy.sparse import csr_matrix


class NegationAwareContrastiveCentroid:
    def __init__(self, neutral_margin=0.025, contrast_strength=0.50):
        self.neutral_margin = neutral_margin
        self.contrast_strength = contrast_strength

    def fit(self, X, y):
        X = csr_matrix(X)
        y = np.asarray(y)
        self.classes_ = np.sort(np.unique(y))
        self.centroids_ = np.vstack([
            np.asarray(X[y == class_id].mean(axis=0)).ravel()
            for class_id in self.classes_
        ])
        norms = np.linalg.norm(self.centroids_, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normalized_centroids_ = self.centroids_ / norms
        contrast = np.max(self.centroids_, axis=0) - np.min(self.centroids_, axis=0)
        if contrast.max() - contrast.min() < 1e-12:
            normalized_contrast = np.ones_like(contrast)
        else:
            normalized_contrast = (contrast - contrast.min()) / (contrast.max() - contrast.min())
        self.feature_weights_ = 1.0 + self.contrast_strength * normalized_contrast
        self.class_priors_ = np.array([(y == class_id).mean() for class_id in self.classes_])
        return self

    def _weighted_normalize(self, X):
        X = csr_matrix(X)
        weighted = X.multiply(self.feature_weights_)
        norms = np.asarray(np.sqrt(weighted.multiply(weighted).sum(axis=1))).ravel()
        norms[norms == 0] = 1.0
        return weighted.multiply(1.0 / norms[:, None])

    def decision_scores(self, X):
        normalized_X = self._weighted_normalize(X)
        weighted_centroids = self.normalized_centroids_ * self.feature_weights_[None, :]
        norms = np.linalg.norm(weighted_centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        weighted_centroids = weighted_centroids / norms
        return np.asarray(normalized_X @ weighted_centroids.T)

    def predict_proba(self, X):
        scores = self.decision_scores(X) * 8.0
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exponentials = np.exp(scores)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict_with_margin(self, X):
        scores = self.decision_scores(X)
        sorted_scores = np.sort(scores, axis=1)
        margins = sorted_scores[:, -1] - sorted_scores[:, -2]
        predictions = self.classes_[np.argmax(scores, axis=1)].copy()
        neutral_position = np.where(self.classes_ == 1)[0][0]
        predictions[margins < self.neutral_margin] = self.classes_[neutral_position]
        return predictions, margins

    def predict(self, X):
        predictions, _ = self.predict_with_margin(X)
        return predictions
