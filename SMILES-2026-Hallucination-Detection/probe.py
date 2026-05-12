"""
probe.py — Hallucination probe classifier.

Three independent pipelines (StandardScaler → PCA → classifier) whose
probability outputs are averaged before thresholding:

  Pipeline A: LogisticRegression(C=0.1)  on PCA(128) features
  Pipeline B: LogisticRegression(C=0.3)  on PCA(192) features
  Pipeline C: SVC(rbf, C=0.3)            on PCA(192) features

PCA whitening is critical: with ~440 training samples and 34118 input
dimensions the feature space is far too large to fit classifiers directly.
Whitening also stabilises the RBF-SVM by normalising the principal components.

Wrapped in nn.Module to satisfy the competition API contract.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_clf_pipeline(n_components: int, clf) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, whiten=True, random_state=42)),
        ("clf",    clf),
    ])


PROBE_CONFIGS = [
    (128, LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000,
                             solver="lbfgs", random_state=42)),
    (192, LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000,
                             solver="lbfgs", random_state=42)),
    (192, SVC(C=0.3, kernel="rbf", class_weight="balanced",
              probability=True, random_state=42)),
]


class HallucinationProbe(nn.Module):
    """Soft-voting ensemble of three PCA-based classifiers.

    Each pipeline operates on an independently scaled and whitened PCA projection,
    so the three classifiers explore slightly different subspaces of the feature
    distribution. The final score is the arithmetic mean of their probabilities.
    """

    def __init__(self) -> None:
        super().__init__()
        self._threshold: float = 0.5
        self._pipelines: list[Pipeline] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "HallucinationProbe does not use forward(). "
            "Use predict() or predict_proba() instead."
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        self._pipelines = []
        for n_pca, clf in PROBE_CONFIGS:
            n = min(n_pca, X.shape[0] - 1, X.shape[1])
            pipe = make_clf_pipeline(n, clf)
            pipe.fit(X, y)
            self._pipelines.append(pipe)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Choose the decision threshold that maximises F1 on a held-out set."""
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            f = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = float(t)
        self._threshold = best_t
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean([p.predict_proba(X) for p in self._pipelines], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)
