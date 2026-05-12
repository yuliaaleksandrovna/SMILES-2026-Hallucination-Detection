"""
Hallucination Detection in Small Language Models

# Files you can edit:
    - aggregation.py — layer selection and token pooling 
    - aggregation.py | extract_geometric_features — optional hand-crafted features 
    - probe.py | HallucinationProbe — probe classifier (nn.Module subclass) 
    - splitting.py | split_data — train / validation / test split strategy 

# Fixed infrastructure (do not edit)
    - model.py | LLM loader (get_model_and_tokenizer) 
    - evaluate.py | Evaluation loop, summary table, JSON output 

# Data Format — ChatML and Special Tokens
    The `prompt` column uses ChatML (Chat Markup Language), the conversation
    template built into Qwen models.  Each message is wrapped in role markers:

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    ... question and context ... <|im_end|>
    <|im_start|>assistant

    Special tokens and their roles:

    - `<|im_start|>` — opens a chat turn; the role (`system`, `user`, or `assistant`) immediately follows
    - `<|im_end|>` — closes the current chat turn
    - `<|endoftext|>` — end-of-sequence (EOS) token appended by the model at the end of its response

    The `prompt` ends right after `<|im_start|>assistant\n` — it provides the
    full context up to (but not including) the model's reply.  The `response`
    column holds the actual generated text, ending with `<|endoftext|>`.

    We feed the concatenation of `prompt + response` to the feature extractor
    so the hidden states capture both the question context and the model's
    specific answer — the hallucination signal lives in that joint representation.


"""

import os
import time

# Let PyTorch fall back to CPU for MPS ops that aren't fully supported
# (e.g. GQA attention broadcast with certain sequence lengths on M-series chips).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Fix all random seeds for reproducibility.
# Note: bfloat16 attention on MPS is not perfectly deterministic due to parallel
# floating-point accumulation; results may vary by ~0.1% AUROC across extractions.
# To guarantee exact reproduction, keep the feature caches (features_cache.npz /
# features_cache_test.npz) and do not delete them between runs.
torch.manual_seed(42)
np.random.seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)

from aggregation import aggregation_and_feature_extraction
from evaluate import print_summary, run_evaluation, save_predictions, save_results
from model import MAX_LENGTH, get_model_and_tokenizer
from probe import HallucinationProbe
from splitting import split_data

# ---------------------------------------------------------------------

DATA_FILE     = "./data/dataset.csv"   # path to the dataset CSV
OUTPUT_FILE   = "results.json"         # where to write the results summary
BATCH_SIZE    = 8
USE_GEOMETRIC = True                  # set True to enable geometric feature extraction
TEST_FILE        = "./data/test.csv"   # competition test set (labels are null)
PREDICTIONS_FILE = "predictions.csv"   # output file with predicted labels
CACHE_FILE       = "features_cache.npz"       # cache extracted features to skip re-extraction
CACHE_FILE_TEST  = "features_cache_test.npz"  # cache extracted test features

assert OUTPUT_FILE == "results.json"
assert PREDICTIONS_FILE == "predictions.csv"
# ---------------------------------------------------------------------
if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    extract_device = device

    print(f"Device       : {device}")
    print(f"Data         : {DATA_FILE}")
    print(f"Max length   : {MAX_LENGTH} tokens")
    print(f"Geometric feats: {USE_GEOMETRIC}")


    df = pd.read_csv(DATA_FILE)
    all_labels = np.array([int(float(h)) for h in df["label"]])

    model, tokenizer = None, None  # loaded lazily only when cache misses

    if os.path.exists(CACHE_FILE):
        # ── Fast path: load pre-extracted features ────────────────────────────
        print(f"Loading cached features from '{CACHE_FILE}' ...")
        cache = np.load(CACHE_FILE)
        X, y = cache["X"], cache["y"]
        extract_time = 0.0
        print(f"Feature matrix : {X.shape}  (feature_dim = {X.shape[1]})")
    else:
        # ── Slow path: extract features from the LLM ─────────────────────────
        # Build the text fed to the LLM: concatenation of prompt and response.
        all_texts  = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]

        n_total = len(all_labels)
        print(f"Loaded {n_total} samples  "
            f"({all_labels.sum()} hallucinated / {(all_labels == 0).sum()} truthful)")

        # Load the LLM
        model, tokenizer = get_model_and_tokenizer()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(extract_device)

        all_features: list = []
        all_prompts = df["prompt"].tolist()

        # Pad to a multiple of BATCH_SIZE to avoid single-sample last batches.
        # MPS GQA attention (Qwen2.5: 14 Q / 2 KV heads) crashes on batch_size=1
        # with certain sequence lengths. Padded entries are discarded after extraction.
        n_real = len(all_texts)
        pad = (-n_real) % BATCH_SIZE  # 0 if already divisible
        if pad:
            all_texts   = all_texts   + all_texts[:pad]
            all_prompts = all_prompts + all_prompts[:pad]

        t0 = time.time()

        for start in tqdm(range(0, len(all_texts), BATCH_SIZE),
                        desc="Extracting & aggregating", unit="batch"):

            # ── 1. Tokenise the current mini-batch ───────────────────────────────
            batch_texts   = all_texts[start : start + BATCH_SIZE]
            batch_prompts = all_prompts[start : start + BATCH_SIZE]

            # Prompt token lengths (no padding) — used to locate response start
            prompt_enc     = tokenizer(batch_prompts, padding=False, truncation=True,
                                       max_length=MAX_LENGTH)
            response_starts = [len(ids) for ids in prompt_enc["input_ids"]]

            encoding = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            input_ids      = encoding["input_ids"].to(extract_device)
            attention_mask = encoding["attention_mask"].to(extract_device)

            # ── 2. LLM forward pass ──────────────────────────────────────────────
            # outputs.hidden_states: tuple of (n_layers+1) tensors,
            # each with shape (batch, seq_len, hidden_dim).
            # Index 0 → token embeddings; index k → transformer layer k.
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # ── 3. Stack all layers into one tensor, move to CPU ─────────────────
            # Shape: (batch, n_layers, seq_len, hidden_dim)
            hidden = torch.stack(outputs.hidden_states, dim=1).float()
            mask   = attention_mask.cpu()

            # ── 4. Aggregate each sample and store the compact feature vector ─────
            # The raw `hidden` tensor is released at the end of this loop iteration.
            for i in range(hidden.size(0)):
                feat = aggregation_and_feature_extraction(
                    hidden[i],   # (n_layers, seq_len, hidden_dim)
                    mask[i],     # (seq_len,)
                    use_geometric=USE_GEOMETRIC,
                    response_start=response_starts[i],
                )
                all_features.append(feat.cpu())

        extract_time = time.time() - t0
        all_features = all_features[:n_real]   # drop padded entries
        print(f"Done in {extract_time:.1f} s  —  {len(all_features)} feature vectors extracted")

        # Stack into the (N, feature_dim) matrix used by the probe.
        X = np.vstack([f.numpy() for f in all_features])   # shape: (N, feature_dim)
        y = all_labels                                       # shape: (N,)

        print(f"Feature matrix : {X.shape}  (feature_dim = {X.shape[1]})")
        print(f"Geometric feats: {USE_GEOMETRIC}")
        np.savez(CACHE_FILE, X=X, y=y)
        print(f"Features cached to '{CACHE_FILE}'")

    splits = split_data(y, df)

    print(f"Splits : {len(splits)} fold(s)")
    for i, (tr, va, te) in enumerate(splits):
        print(f"  Fold {i + 1}: train={len(tr)}  "
            f"val={len(va) if va is not None else 'N/A'}  test={len(te)}")

    fold_results = run_evaluation(splits, X, y, HallucinationProbe)
    
    print_summary(fold_results, X.shape[1], len(X), extract_time)
    save_results(fold_results, X.shape[1], len(X), extract_time, OUTPUT_FILE)

    

    # ── Load test data ────────────────────────────────────────────────────────
    df_test  = pd.read_csv(TEST_FILE)
    test_ids = df_test.index
    print(f"Test set loaded: {len(df_test)} samples")

    if os.path.exists(CACHE_FILE_TEST):
        # ── Fast path: load pre-extracted test features ───────────────────────
        print(f"Loading cached test features from '{CACHE_FILE_TEST}' ...")
        X_test = np.load(CACHE_FILE_TEST)["X"]
        print(f"Test feature matrix: {X_test.shape}")
    else:
        # ── Slow path: extract test features from the LLM ────────────────────
        test_texts    = [f"{row['prompt']}{row['response']}" for _, row in df_test.iterrows()]
        test_prompts  = df_test["prompt"].tolist()

        # Load model/tokenizer if not already loaded (train fast path skipped it)
        if model is None:
            model, tokenizer = get_model_and_tokenizer()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.to(extract_device)

        test_features: list = []
        n_real_test = len(test_texts)
        pad_test = (-n_real_test) % BATCH_SIZE
        if pad_test:
            test_texts   = test_texts   + test_texts[:pad_test]
            test_prompts = test_prompts + test_prompts[:pad_test]

        for start in tqdm(range(0, len(test_texts), BATCH_SIZE),
                        desc="Test extraction & aggregation", unit="batch"):

            batch_texts   = test_texts[start : start + BATCH_SIZE]
            batch_prompts = test_prompts[start : start + BATCH_SIZE]

            prompt_enc      = tokenizer(batch_prompts, padding=False, truncation=True,
                                        max_length=MAX_LENGTH)
            response_starts = [len(ids) for ids in prompt_enc["input_ids"]]

            encoding = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            input_ids      = encoding["input_ids"].to(extract_device)
            attention_mask = encoding["attention_mask"].to(extract_device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            hidden = torch.stack(outputs.hidden_states, dim=1).float()
            mask   = attention_mask.cpu()

            for i in range(hidden.size(0)):
                feat = aggregation_and_feature_extraction(
                    hidden[i], mask[i],
                    use_geometric=USE_GEOMETRIC,
                    response_start=response_starts[i],
                )
                test_features.append(feat.cpu())

        test_features = test_features[:n_real_test]
        X_test = np.vstack([f.numpy() for f in test_features])  # (n_test, feature_dim)
        np.savez(CACHE_FILE_TEST, X=X_test)
        print(f"Test features cached to '{CACHE_FILE_TEST}'")

    # ── Fit final probe on training + validation data only ──────────────────
    # Collect the union of all train and validation indices across every split.
    # For a single split this excludes idx_test; for k-fold every sample appears
    # in a training fold, so all samples are used (same as fitting on X, y).
    idx_non_test = np.unique(np.concatenate([
        np.concatenate([idx_tr, idx_va]) if idx_va is not None else idx_tr
        for idx_tr, idx_va, _ in splits
    ]))
    final_probe = HallucinationProbe()
    final_probe.fit(X[idx_non_test], y[idx_non_test])

    # ── Predict and save ────────────────────────────────────────────────────
    save_predictions(final_probe, X_test, test_ids, PREDICTIONS_FILE)

