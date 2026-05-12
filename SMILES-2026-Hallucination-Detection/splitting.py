"""
splitting.py — Train / validation / test split utilities.

Returns 5 stratified folds via StratifiedKFold. Each fold has an inner
validation split (~15% of total) carved from the training portion for
decision-threshold tuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into 5 stratified folds with inner val splits.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
        df:           Unused; kept for API compatibility.
        test_size:    Unused; fold test size is 1/5 = 20% by construction.
        val_size:     Target fraction of *total* dataset reserved for val.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of 5 ``(idx_train, idx_val, idx_test)`` tuples.
    """
    n_total = len(y)
    idx = np.arange(n_total)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = []

    for fold_train_idx, fold_test_idx in skf.split(idx, y):
        # Inner val: target ~val_size fraction of total, taken from fold train
        n_val_target = max(1, int(n_total * val_size))
        inner_val_frac = min(n_val_target / len(fold_train_idx), 0.5)

        fold_train_inner, fold_val_inner = train_test_split(
            fold_train_idx,
            test_size=inner_val_frac,
            random_state=random_state,
            stratify=y[fold_train_idx],
        )
        splits.append((fold_train_inner, fold_val_inner, fold_test_idx))

    return splits
