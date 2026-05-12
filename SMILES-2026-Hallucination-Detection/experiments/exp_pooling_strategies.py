"""
Experiment: compare different response token pooling strategies.

Starting hypothesis: mean-pooling over response tokens should outperform
last-token alone, because it captures the whole generation trajectory.
Also tests whether restricting to response tokens (vs all tokens) helps.
"""
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = "./data/dataset.csv"
CACHE = "features_cache.npz"

data = pd.read_csv(DATA)
labels = data["label"].values
cache = np.load(CACHE)
X = cache["X"]
y = labels[:len(X)]
print(f"Loaded X={X.shape}, positive_rate={y.mean():.2f}\n")

# The cache contains last-token + response_mean_14 + geo = 34118 dims.
# We approximate different strategies by slicing the cached features.
HIDDEN_DIM = 896
LAST_TOKEN_DIMS = 24 * HIDDEN_DIM       # 21504
RESP_MEAN_DIMS  = 14 * HIDDEN_DIM       # 12544
GEO_DIMS        = 70

X_last   = X[:, :LAST_TOKEN_DIMS]
X_mean   = X[:, LAST_TOKEN_DIMS:LAST_TOKEN_DIMS + RESP_MEAN_DIMS]
X_geo    = X[:, LAST_TOKEN_DIMS + RESP_MEAN_DIMS:]
X_full   = X


def cv_auroc(Xf, n_pca=192, C=0.1, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, te in kf.split(Xf, y):
        n = min(n_pca, Xf[tr].shape[0] - 1, Xf.shape[1])
        pipe = Pipeline([
            ("sc",  StandardScaler()),
            ("pca", PCA(n, whiten=True, random_state=42)),
            ("clf", LogisticRegression(C=C, class_weight="balanced",
                                       max_iter=2000, random_state=42)),
        ])
        pipe.fit(Xf[tr], y[tr])
        proba = pipe.predict_proba(Xf[te])[:, 1]
        scores.append(roc_auc_score(y[te], proba))
    return float(np.mean(scores))


print("Strategy                          CV AUROC")
print("-" * 45)
print(f"last-token only              {cv_auroc(X_last):.4f}")
print(f"response mean only           {cv_auroc(X_mean):.4f}")
print(f"last + response mean         {cv_auroc(np.hstack([X_last, X_mean])):.4f}")
print(f"last + response mean + geo   {cv_auroc(X_full):.4f}")
print(f"geo only                     {cv_auroc(X_geo, n_pca=64):.4f}")
