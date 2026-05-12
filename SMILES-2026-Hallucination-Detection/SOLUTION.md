# Solution Report

## Reproducibility

**Environment:** Python 3.10+. Apple M5 (MPS, 16 GB RAM).

```bash
pip install -r requirements.txt
python solution.py
```

Outputs: `results.json` (5-fold CV metrics) and `predictions.csv` (test predictions).

Hidden-state extraction takes ~8 minutes on M4 and is cached to `features_cache.npz` / `features_cache_test.npz`. Delete either file to force re-extraction from scratch.

**Seeds:** Fixed to `42` everywhere — PyTorch, NumPy, MPS, KFold, PCA, LogisticRegression, SVC. The probe training and CV are fully deterministic when the cache is present. Re-extraction on MPS may vary by ±0.1% AUROC due to bfloat16 non-determinism in grouped-query attention; on CUDA or CPU it is deterministic.

---

## Results

5-fold stratified CV on the labelled dataset (689 samples, 70% positive):

| Split | Accuracy | F1 | AUROC |
|-------|----------|----|-------|
| Train | 78.80% | 86.84% | 94.88% |
| Val   | 75.92% | 84.89% | 76.22% |
| Test  | 75.91% | 84.82% | **78.31%** |

Majority-class baseline: 70.10% accuracy, 82.42% F1.

---

## Approach

### Intuition

A language model that hallucinates and one that answers correctly process the same prompt differently at the hidden-state level. The hallucination signal should be visible in how representations evolve across the transformer's depth during the generation of the response — not just at a single token or a single layer.

### Feature extraction (`aggregation.py`)

For each sample I run `prompt + response` through Qwen2.5-0.5B with `output_hidden_states=True` and build a 34118-dim vector from three components:

**1. Sequence endpoint — 21504 dims**
Hidden state at the last non-padding token, concatenated across all 24 transformer layers (24 × 896). In a decoder-only model every token attends to all previous ones, so the final token's representation aggregates the entire sequence. Using all 24 layers preserves the full depth of the model's processing chain.

**2. Response embedding — 12544 dims**
Mean of hidden states over response tokens (everything after `<|im_start|>assistant\n`), taken from only the deepest 14 of 24 transformer layers (14 × 896). Two design choices here:

- *Response-only:* the prompt is near-identical across all samples — same system prompt, same RAG context structure. Including prompt tokens in the pool averages in a constant that dilutes the hallucination signal.
- *Deep layers only:* Qwen2.5-0.5B's early layers encode surface syntax shared across all inputs. From a sweep over L ∈ {6, 8, 10, 12, 13, 14} I found that L=14 (layers 11–24) maximises CV AUROC. Earlier layers add noise.

**3. Layer trajectory statistics — 70 dims**
Computed from the response mean-pool across all 24 layers: L2 norms per layer (24), consecutive inter-layer cosine similarities (23), and norm deltas (23). These capture how the model's internal representation evolves with depth — a hallucinated response may show a different norm growth or similarity profile compared to a factual one. Using all 24 layers is fine here because 70 dims is small enough that early-layer statistics don't inflate the feature space.

Total: 21504 + 12544 + 70 = **34118 dims**.

### Probe classifier (`probe.py`)

Three independent `StandardScaler → PCA(n, whiten=True) → classifier` pipelines; final probability is their arithmetic mean:

| Pipeline | PCA dims | Classifier |
|----------|----------|------------|
| A | 128 | `LogisticRegression(C=0.1, class_weight='balanced')` |
| B | 192 | `LogisticRegression(C=0.3, class_weight='balanced')` |
| C | 192 | `SVC(rbf, C=0.3, class_weight='balanced', probability=True)` |

With ~440 training samples and 34118 input dimensions direct fitting would overfit badly. PCA whitening compresses the correlated feature space into orthogonal components, and three pipelines at different PCA scales and regularisation levels add diversity without adding overfitting risk.

Decision threshold is tuned per fold on the inner validation split (≈15% of the dataset) to maximise F1.

### Split strategy (`splitting.py`)

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` preserves the 70/30 class ratio in each fold. An inner split carved from the training portion provides a threshold-tuning set without leaking into the test fold.

### What helped

1. **Adding response mean-pool alongside last token** — the largest jump. Last-token tells you where the model ended up; mean-pool tells you the average trajectory. Together they are more informative than either alone.
2. **Response-only pooling** — excluding the prompt from the pool significantly cleaned up the signal. The prompt is nearly constant across samples, so including it just adds a shared constant to every feature vector.
3. **Restricting to deep transformer layers** — dropping layers 1–10 from the response embedding reduced noise and improved CV AUROC noticeably.
4. **Geometric trajectory features** — norms and cosine similarities across layers add a compact representation of how the model's processing evolves with depth.
5. **Diverse ensemble with SVM** — the RBF kernel captures non-linearities that logistic regression misses; using two LR members at different regularisation levels reduces variance.

---

## Experiments and Failed Attempts

**All-token pooling (first attempt).** Started by mean-pooling over all sequence tokens across the last 8 layers. Got ~60% Test AUROC — substantially worse than using the last token alone. The response tokens are a small fraction of the full sequence; averaging over the prompt introduces constant noise.

**Single-layer features.** Tried using only the last transformer layer (hidden_dim=896). Test AUROC around 62%. Using all 24 layers and PCA to compress them is far more informative.

**Gradient boosting (LightGBM, CatBoost).** With only ~440 training samples per fold these models overfit severely even after PCA: Train AUROC ~100%, Test AUROC ~58–62%. They need more data than we have.

**Adding token variance as extra features.** Computed per-dimension standard deviation over response tokens (another 12544 dims) and concatenated to the existing features. Total dims went to 46662. Test AUROC dropped by ~0.8% — the extra features are too noisy relative to the training set size, and PCA can't extract signal from them at 128–192 components.

**4-member ensemble.** Adding a linear SVM as a fourth pipeline gave a small CV improvement but slightly hurt the full evaluation (threshold tuning becomes less stable with four differently calibrated members on ~103-sample val splits). Reverted to three pipelines.

**Layer sweep details:**

| Deep layers L | Feature dim | CV AUROC |
|--------------|-------------|----------|
| 6 | 26950 | 0.7725 |
| 8 | 28742 | 0.7745 |
| 10 | 30534 | 0.7731 |
| 12 | 32326 | 0.7749 |
| 13 | 33222 | 0.7758 |
| **14** | **34118** | **0.7767** |
| 24 (all) | 43078 | 0.7698 |
