import numpy as np
import pytest
from probe import HallucinationProbe

N_SAMPLES = 120
FEATURE_DIM = 50


def make_dataset(n=N_SAMPLES, d=FEATURE_DIM, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = np.array([0] * (n // 3) + [1] * (n - n // 3))
    rng.shuffle(y)
    return X, y


def test_probe_is_nn_module():
    import torch.nn as nn
    probe = HallucinationProbe()
    assert isinstance(probe, nn.Module)


def test_fit_returns_self():
    X, y = make_dataset()
    probe = HallucinationProbe()
    result = probe.fit(X, y)
    assert result is probe


def test_predict_shape_and_values():
    X, y = make_dataset()
    probe = HallucinationProbe()
    probe.fit(X, y)
    preds = probe.predict(X)
    assert preds.shape == (N_SAMPLES,), f"Expected ({N_SAMPLES},), got {preds.shape}"
    assert set(preds).issubset({0, 1}), f"Unexpected labels: {set(preds)}"


def test_predict_proba_shape_and_sums():
    X, y = make_dataset()
    probe = HallucinationProbe()
    probe.fit(X, y)
    proba = probe.predict_proba(X)
    assert proba.shape == (N_SAMPLES, 2), f"Expected ({N_SAMPLES}, 2), got {proba.shape}"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "Probabilities do not sum to 1"
    assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities out of [0, 1]"


def test_fit_hyperparameters_updates_threshold():
    X, y = make_dataset()
    split = int(0.8 * N_SAMPLES)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    probe = HallucinationProbe()
    probe.fit(X_train, y_train)
    probe.fit_hyperparameters(X_val, y_val)
    assert 0.0 <= probe._threshold <= 1.0


def test_probe_better_than_majority():
    """Probe on train data should beat majority-class baseline."""
    X, y = make_dataset(n=200, d=100, seed=42)
    probe = HallucinationProbe()
    probe.fit(X, y)
    preds = probe.predict(X)
    acc = (preds == y).mean()
    majority_acc = max(y.mean(), 1 - y.mean())
    assert acc > majority_acc, f"Probe acc {acc:.3f} <= majority {majority_acc:.3f}"


def test_predict_proba_col1_matches_predict():
    """predict() must be consistent with predict_proba() at stored threshold."""
    X, y = make_dataset()
    probe = HallucinationProbe()
    probe.fit(X, y)
    proba = probe.predict_proba(X)[:, 1]
    expected = (proba >= probe._threshold).astype(int)
    actual = probe.predict(X)
    np.testing.assert_array_equal(actual, expected)
