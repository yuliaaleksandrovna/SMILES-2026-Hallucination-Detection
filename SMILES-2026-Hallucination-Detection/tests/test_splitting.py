import numpy as np
import pytest
from splitting import split_data

N = 100
Y = np.array([0] * 30 + [1] * 70)  # mirrors 30/70 train imbalance


def test_returns_five_folds():
    splits = split_data(Y)
    assert len(splits) == 5, f"Expected 5 folds, got {len(splits)}"


def test_each_fold_has_three_parts():
    for idx_train, idx_val, idx_test in split_data(Y):
        assert idx_train is not None
        assert idx_val is not None
        assert idx_test is not None


def test_no_overlap_within_fold():
    for idx_train, idx_val, idx_test in split_data(Y):
        all_idx = np.concatenate([idx_train, idx_val, idx_test])
        assert len(all_idx) == len(np.unique(all_idx)), "Overlapping indices found"


def test_full_coverage():
    for idx_train, idx_val, idx_test in split_data(Y):
        all_idx = np.sort(np.concatenate([idx_train, idx_val, idx_test]))
        np.testing.assert_array_equal(all_idx, np.arange(N))


def test_test_fold_stratified():
    """Each test fold class ratio should be within 10pp of overall ratio."""
    overall_pos = Y.mean()
    for idx_train, idx_val, idx_test in split_data(Y):
        fold_pos = Y[idx_test].mean()
        assert abs(fold_pos - overall_pos) < 0.10, \
            f"Test fold class ratio {fold_pos:.2f} deviates from {overall_pos:.2f}"


def test_val_not_empty():
    for idx_train, idx_val, idx_test in split_data(Y):
        assert len(idx_val) > 0, "Val split is empty"


def test_reproducible():
    splits_a = split_data(Y, random_state=42)
    splits_b = split_data(Y, random_state=42)
    for (ta, va, tea), (tb, vb, teb) in zip(splits_a, splits_b):
        np.testing.assert_array_equal(ta, tb)
        np.testing.assert_array_equal(va, vb)
        np.testing.assert_array_equal(tea, teb)
