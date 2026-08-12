"""Shared AUC and video helpers."""

from __future__ import annotations

import os
import re

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def fast_auc(labels, scores):
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = scores.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_s = scores[order]
    i = 0
    n = scores.size
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j + 1)
        i = j
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def category_of(video_key):
    base = os.path.basename(str(video_key)).split(".")[0]
    m = re.match(r"^[A-Za-z]+", base)
    return m.group(0) if m else base


def to_snippets(scores, labels, snippet=16):
    n = min(len(scores), len(labels))
    if n == 0:
        return np.array([]), np.array([])
    scores = np.asarray(scores[:n], dtype=float)
    labels = np.asarray(labels[:n], dtype=float)
    T = max(1, n // snippet)
    edges = np.linspace(0, n, T + 1, dtype=int)
    widths = np.maximum(edges[1:] - edges[:-1], 1)
    csum = np.concatenate([[0.0], np.cumsum(scores)])
    s = (csum[edges[1:]] - csum[edges[:-1]]) / widths
    cany = np.concatenate([[0.0], np.cumsum(labels > 0)])
    g = ((cany[edges[1:]] - cany[edges[:-1]]) > 0).astype(float)
    return s, g


def downsample_gt(gt_frames, T):
    gt_frames = np.asarray(gt_frames, dtype=float)
    if len(gt_frames) == T:
        return gt_frames
    edges = np.linspace(0, len(gt_frames), T + 1, dtype=int)
    return np.array(
        [gt_frames[edges[i] : edges[i + 1]].any() for i in range(T)],
        dtype=float,
    )


def cosine_rows(X, proto):
    """Cosine similarity of each row of X to proto. Both float arrays."""
    X = np.asarray(X, dtype=np.float64)
    proto = np.asarray(proto, dtype=np.float64).ravel()
    pn = np.linalg.norm(proto)
    if pn < 1e-12:
        return np.zeros(len(X), dtype=np.float64)
    xn = np.linalg.norm(X, axis=1)
    xn = np.maximum(xn, 1e-12)
    return (X @ proto) / (xn * pn)


def mann_whitney_auc(pos, neg):
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return None
    return auc(
        np.concatenate([np.ones(pos.size), np.zeros(neg.size)]),
        np.concatenate([pos, neg]),
    )
