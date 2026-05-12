"""
aggregation.py — Hidden-state feature extraction for hallucination detection.

Extracts three complementary representations from Qwen2.5-0.5B hidden states:

  1. sequence_endpoint()       — hidden state at the last real token, all 24 layers (21504 dims)
  2. response_embedding()      — mean-pooled response tokens, deep layers only (12544 dims)
  3. cross_layer_statistics()  — norms and similarity profile across layers (70 dims)

Combined by aggregation_and_feature_extraction() → 34118-dim feature vector.

Key constant:
  SEMANTIC_LAYERS = 14  — number of deep transformer layers used for response embedding.
                          Tuned by cross-validation; early layers encode surface syntax
                          shared across all samples and add noise rather than signal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# How many of the deepest transformer layers to use when embedding response tokens.
# Layers 1-10 encode surface-level token patterns nearly identical across all samples
# (same system prompt, same RAG context format). Only layers 11-24 carry semantic
# content discriminative for hallucination detection.
SEMANTIC_LAYERS: int = 14


def _response_mask(
    attention_mask: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """Return a float mask restricted to response tokens, falling back to all real tokens."""
    mask = attention_mask.float()
    if response_start > 0 and response_start < mask.shape[0]:
        resp = mask.clone()
        resp[:response_start] = 0.0
        if resp.sum() > 0:
            return resp
    return mask


def sequence_endpoint(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Hidden state of the last real token concatenated across all 24 transformer layers.

    In a decoder-only model the last token's representation is informed by every
    preceding token via causal attention, making it a compact summary of the full
    sequence. Using all 24 layers preserves the full depth of the model's processing.

    Args:
        hidden_states:  (n_layers, seq_len, hidden_dim) — index 0 is the embedding layer.
        attention_mask: (seq_len,) — 1 for real tokens, 0 for padding.

    Returns:
        (24 * hidden_dim,) = (21504,) float tensor.
    """
    layers = hidden_states[1:]  # skip embedding layer → (24, seq_len, hidden_dim)
    mask = attention_mask.to(hidden_states.device)
    last_pos = int(mask.nonzero(as_tuple=False)[-1].item())
    return layers[:, last_pos, :].reshape(-1)


def response_embedding(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start: int = 0,
) -> torch.Tensor:
    """Mean-pool response tokens over the deepest SEMANTIC_LAYERS transformer layers.

    Restricting to response tokens removes shared prompt boilerplate that is nearly
    identical across all samples. Restricting to deep layers removes early-layer
    surface syntax representations that carry no hallucination signal.

    Args:
        hidden_states:   (n_layers, seq_len, hidden_dim).
        attention_mask:  (seq_len,) — 1 for real tokens.
        response_start:  First token index belonging to the assistant response.

    Returns:
        (SEMANTIC_LAYERS * hidden_dim,) = (12544,) float tensor.
    """
    layers = hidden_states[1:]  # (24, seq_len, hidden_dim)
    mask = _response_mask(attention_mask.to(hidden_states.device), response_start)

    n_tokens = mask.sum().clamp(min=1.0)
    pooled = (layers * mask.unsqueeze(-1).unsqueeze(0)).sum(dim=1) / n_tokens  # (24, hidden_dim)

    # Keep only the deepest SEMANTIC_LAYERS layers
    hidden_dim = hidden_states.shape[-1]
    pooled_deep = pooled[(24 - SEMANTIC_LAYERS):]  # (SEMANTIC_LAYERS, hidden_dim)
    return pooled_deep.reshape(-1)


def cross_layer_statistics(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start: int = 0,
) -> torch.Tensor:
    """Geometric statistics of the response representation trajectory across all 24 layers.

    Captures how the model's internal representation of the response evolves with depth:
    monotonically growing norms signal confident processing; high inter-layer similarity
    suggests stable representations; abrupt changes may indicate uncertainty.

    Uses all 24 layers (not just the deep SEMANTIC_LAYERS) because the compact 70-dim
    output means early-layer statistics contribute signal without inflating the feature space.

    Args:
        hidden_states:   (n_layers, seq_len, hidden_dim).
        attention_mask:  (seq_len,) — 1 for real tokens.
        response_start:  First token index belonging to the assistant response.

    Returns:
        (70,) float tensor:
          - 24 L2 norms, one per layer
          - 23 consecutive cosine similarities
          - 23 norm deltas (layer_{l+1} − layer_l)
    """
    transformer = hidden_states[1:]  # (24, seq_len, hidden_dim)
    mask = _response_mask(attention_mask.to(hidden_states.device), response_start)

    n_tokens = mask.sum().clamp(min=1.0)
    layer_means = (transformer * mask.unsqueeze(-1).unsqueeze(0)).sum(dim=1) / n_tokens  # (24, hidden_dim)

    norms = layer_means.norm(dim=-1)                                          # (24,)
    cos_sims = F.cosine_similarity(layer_means[:-1], layer_means[1:], dim=-1) # (23,)
    norm_deltas = norms[1:] - norms[:-1]                                      # (23,)

    return torch.cat([norms, cos_sims, norm_deltas])  # (70,)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
    response_start: int = 0,
) -> torch.Tensor:
    """Build the full feature vector for one sample.

    Concatenates sequence_endpoint + response_embedding (+ cross_layer_statistics
    if use_geometric is True).

    Args:
        hidden_states:   (n_layers, seq_len, hidden_dim).
        attention_mask:  (seq_len,) — 1 for real tokens, 0 for padding.
        use_geometric:   Append cross_layer_statistics (70 dims).
        response_start:  First response token index.

    Returns:
        (34048,) without geometric, (34118,) with geometric.
    """
    endpoint = sequence_endpoint(hidden_states, attention_mask)
    embedding = response_embedding(hidden_states, attention_mask, response_start)

    if use_geometric:
        geo = cross_layer_statistics(hidden_states, attention_mask, response_start)
        return torch.cat([endpoint, embedding, geo])

    return torch.cat([endpoint, embedding])
