import torch
import pytest
from aggregation import aggregate, extract_geometric_features, aggregation_and_feature_extraction

# Qwen2.5-0.5B: 24 transformer layers + 1 embedding = 25 total, hidden_dim=896
N_LAYERS = 25
HIDDEN_DIM = 896
SEQ_LEN = 20
N_TRANSFORMER = 24   # indices 1..24 (skip embedding at 0)
GEO_DIM = N_TRANSFORMER + (N_TRANSFORMER - 1) + (N_TRANSFORMER - 1)  # 24+23+23 = 70


def make_hidden(seq_len=SEQ_LEN, n_padding=3):
    hidden = torch.randn(N_LAYERS, seq_len, HIDDEN_DIM)
    mask = torch.ones(seq_len)
    mask[seq_len - n_padding:] = 0
    return hidden, mask


def test_aggregate_output_shape():
    hidden, mask = make_hidden()
    feat = aggregate(hidden, mask)
    assert feat.shape == (N_TRANSFORMER * HIDDEN_DIM,), \
        f"Expected ({N_TRANSFORMER * HIDDEN_DIM},), got {feat.shape}"


def test_aggregate_no_nan():
    hidden, mask = make_hidden()
    feat = aggregate(hidden, mask)
    assert not torch.isnan(feat).any(), "aggregate() returned NaN values"


def test_aggregate_last_token():
    """Result should equal last real token of all 24 transformer layers."""
    hidden = torch.randn(N_LAYERS, SEQ_LEN, HIDDEN_DIM)
    mask = torch.ones(SEQ_LEN)
    mask[SEQ_LEN - 3:] = 0  # 3 padding tokens, last real is SEQ_LEN-4
    feat = aggregate(hidden, mask)
    last_pos = SEQ_LEN - 4
    expected = hidden[1:, last_pos, :].reshape(-1)  # skip embedding (index 0)
    assert torch.allclose(feat, expected, atol=1e-5), "Mismatch: not last real token"


def test_extract_geometric_features_shape():
    hidden, mask = make_hidden()
    geo = extract_geometric_features(hidden, mask)
    assert geo.shape == (GEO_DIM,), \
        f"Expected ({GEO_DIM},), got {geo.shape}"


def test_extract_geometric_no_nan():
    hidden, mask = make_hidden()
    geo = extract_geometric_features(hidden, mask)
    assert not torch.isnan(geo).any(), "extract_geometric_features() returned NaN"


def test_combined_shape_with_geometric():
    hidden, mask = make_hidden()
    feat = aggregation_and_feature_extraction(hidden, mask, use_geometric=True)
    expected_dim = N_TRANSFORMER * HIDDEN_DIM + GEO_DIM
    assert feat.shape == (expected_dim,), \
        f"Expected ({expected_dim},), got {feat.shape}"


def test_combined_shape_without_geometric():
    hidden, mask = make_hidden()
    feat = aggregation_and_feature_extraction(hidden, mask, use_geometric=False)
    assert feat.shape == (N_TRANSFORMER * HIDDEN_DIM,), \
        f"Expected ({N_TRANSFORMER * HIDDEN_DIM},), got {feat.shape}"
