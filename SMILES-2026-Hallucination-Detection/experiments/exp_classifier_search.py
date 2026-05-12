"""
Experiment: search over probe ensemble configurations.

Tests different combinations of PCA dimensionality and classifier type/regularisation.
All experiments use the full 34118-dim feature cache.
"""
import copy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA  = "./data/dataset.csv"
CACHE = "features_cache.npz"

data = pd.read_csv(DATA)
y = data["label"].values
cache = np.load(CACHE)
X = cache["X"]
y = y[:len(X)]
print(f"X={X.shape}, pos={y.mean():.2f}\n")


def run(specs, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, te in kf.split(X, y):
        pipes = []
        for n_pca, clf in specs:
            n = min(n_pca, X[tr].shape[0] - 1, X.shape[1])
            p = Pipeline([("sc", StandardScaler()),
                          ("pca", PCA(n, whiten=True, random_state=42)),
                          ("clf", copy.deepcopy(clf))])
            p.fit(X[tr], y[tr])
            pipes.append(p)
        proba = np.mean([p.predict_proba(X[te]) for p in pipes], axis=0)[:, 1]
        scores.append(roc_auc_score(y[te], proba))
    return float(np.mean(scores))


LR = lambda C: LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                   solver="lbfgs", random_state=42)
SVM = lambda C: SVC(C=C, kernel="rbf", class_weight="balanced",
                    probability=True, random_state=42)

experiments = [
    ("single LR(192, C=0.1)",                   [(192, LR(0.1))]),
    ("single SVM(192, C=0.3)",                  [(192, SVM(0.3))]),
    ("LR(128)+LR(192)",                         [(128, LR(0.1)), (192, LR(0.3))]),
    ("LR(128)+SVM(192, C=0.1)",                 [(128, LR(0.1)), (192, SVM(0.1))]),
    ("LR(128)+LR(192)+SVM(192, C=0.1)",         [(128, LR(0.1)), (192, LR(0.3)), (192, SVM(0.1))]),
    ("LR(128)+LR(192)+SVM(192, C=0.3) [best]",  [(128, LR(0.1)), (192, LR(0.3)), (192, SVM(0.3))]),
    ("LR(128)+LR(192)+SVM(192, C=1.0)",         [(128, LR(0.1)), (192, LR(0.3)), (192, SVM(1.0))]),
]

for name, specs in experiments:
    score = run(specs)
    print(f"{name:<48}  {score:.4f}")
