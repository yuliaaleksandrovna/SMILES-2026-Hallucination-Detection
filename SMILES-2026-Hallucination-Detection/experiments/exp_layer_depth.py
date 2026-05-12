"""
Experiment: how many deep transformer layers to include in response_embedding().

Sweeps L from 6 to 24 (number of layers from the bottom of the transformer,
i.e. layers (24-L+1)..24). The hypothesis is that early layers encode shared
prompt/context syntax and contribute noise to the response embedding.
"""
import numpy as np
import pandas as pd
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
X_full = cache["X"]
y = labels[:len(X_full)]

HIDDEN_DIM = 896
LAST_TOKEN_END = 24 * HIDDEN_DIM   # 21504
RESP_START     = LAST_TOKEN_END
RESP_END       = LAST_TOKEN_END + 14 * HIDDEN_DIM  # full 14-layer block

# resp_embedding layout (deep layers first in our cache):
# X_full[:, RESP_START : RESP_START + 14*896]
# where position 0 corresponds to layer 11, position 13*896 to layer 24.

def cv_auroc_with_L(L, n_pca=192, n_splits=5):
    """Use only the last L layers of the response embedding."""
    # Take last L*HIDDEN_DIM dims of the response block
    resp_block = X_full[:, RESP_START:RESP_END]  # (N, 14*896)
    resp_deep  = resp_block[:, (14 - L) * HIDDEN_DIM:]  # last L layers
    geo_block  = X_full[:, RESP_END:]
    last_block = X_full[:, :LAST_TOKEN_END]
    Xf = np.hstack([last_block, resp_deep, geo_block])

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, te in kf.split(Xf, y):
        n = min(n_pca, Xf[tr].shape[0] - 1, Xf.shape[1])
        pipe = Pipeline([
            ("sc",  StandardScaler()),
            ("pca", PCA(n, whiten=True, random_state=42)),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                       max_iter=2000, random_state=42)),
        ])
        pipe.fit(Xf[tr], y[tr])
        proba = pipe.predict_proba(Xf[te])[:, 1]
        scores.append(roc_auc_score(y[te], proba))
    return float(np.mean(scores))


print(f"{'L (layers)':>12}  {'feature_dim':>12}  {'CV AUROC':>10}")
print("-" * 40)
for L in [4, 6, 8, 10, 12, 13, 14]:
    dim = 24 * HIDDEN_DIM + L * HIDDEN_DIM + 70
    score = cv_auroc_with_L(L)
    marker = " ←" if L == 14 else ""
    print(f"{L:>12}  {dim:>12}  {score:>10.4f}{marker}")
