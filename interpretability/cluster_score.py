"""Rank 30-token windows using trunk logits and clusters of short continuations."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


KEY = ["task_id", "trunk_seed", "position"]
LOGIT_FEATURES = [
    "entropy_max", "entropy_sum", "varentropy_max", "varentropy_sum",
    "rank_max", "rank_sum", "rose_max", "rose_sum", "gap_max", "gap_sum",
    "surprise_max", "surprise_sum", "entropy_jump_max", "entropy_jump_sum",
    "position", "remaining", "open_tokens", "zero_pi",
]
CLUSTER_FEATURES = ["surprise", "entropy", "dispersion"]


def cluster_features(pieces, qwen_tokenizer, minilm_tokenizer, minilm_model):
    """Compute the three B1 features from the Qwen token IDs at one prefix.

    Pass the next trunk tokens first, then the tokens of its 100 continuations.
    Use the Qwen3.5-9B tokenizer and the all-MiniLM-L6-v2 tokenizer and model.
    """
    import torch

    if len(pieces) != 101:
        raise ValueError("Expected one trunk piece followed by its 100 continuations")
    texts = qwen_tokenizer.batch_decode([piece[:30] for piece in pieces], skip_special_tokens=True)
    inputs = minilm_tokenizer(texts, padding=True, truncation=True, max_length=256,
                              return_tensors="pt").to(minilm_model.device)
    minilm_model.eval()
    with torch.inference_mode():
        hidden = minilm_model(**inputs).last_hidden_state
        mask = inputs.attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
    vectors = pooled.float().cpu().numpy()
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    # The experiment stored normalized vectors in float16 before computing distances.
    vectors = vectors.astype(np.float16).astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    distances = np.clip(1 - vectors @ vectors.T, 0, 2)
    tree = linkage(distances[np.triu_indices(101, 1)], method="average")
    clusters = fcluster(tree, t=0.33, criterion="distance")
    counts = np.unique(clusters[1:], return_counts=True)[1]
    shares = counts / 100
    trunk_matches = np.count_nonzero(clusters[1:] == clusters[0])
    return {
        "surprise": -np.log((trunk_matches + 1) / 101),
        "entropy": -np.dot(shares, np.log(shares)),
        "dispersion": distances[1:, 1:][np.triu_indices(100, 1)].mean(),
    }


def held_trunk_scores(windows):
    """Fit on the other trunks and predict every window of the held trunk."""
    features = windows[LOGIT_FEATURES + CLUSTER_FEATURES].copy()
    features[LOGIT_FEATURES] = features[LOGIT_FEATURES].replace(np.inf, 1000.0)
    features = features.replace([np.inf, -np.inf], np.nan)
    scores = pd.Series(index=windows.index, dtype=float, name="score")
    for _, held in windows.groupby(KEY[:2]):
        train = windows.drop(held.index)
        labels = train["strong"]
        if labels.nunique() == 1:
            scores.loc[held.index] = labels.mean()
            continue
        train_x = features.loc[train.index]
        medians = train_x.median().fillna(0)
        weights = 1 / train.groupby(KEY[:2])["position"].transform("size")
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=0.2, max_iter=1000, class_weight="balanced", random_state=17011,
        ))
        model.fit(train_x.fillna(medians), labels,
                  logisticregression__sample_weight=weights / weights.mean())
        scores.loc[held.index] = model.predict_proba(features.loc[held.index].fillna(medians))[:, 1]
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("windows", type=Path, help="Parquet with window keys, features and strong labels")
    parser.add_argument("--output", required=True, type=Path, help="CSV with a held-trunk score per window")
    args = parser.parse_args()
    windows = pd.read_parquet(args.windows)
    windows[KEY].assign(score=held_trunk_scores(windows)).to_csv(args.output, index=False)
