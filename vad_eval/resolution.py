"""Paired video-bootstrap model comparison at one granularity."""

from __future__ import annotations

import numpy as np

from .granularity import metric_on_idx, point_metrics

GRANULARITIES = ("global", "per_category", "within_video")


def _ci(arr):
    if not arr:
        return None, None
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def paired_bootstrap(packs, n_boot=1000, seed=42, granularities=None):
    """
    Shared-index video bootstrap across models.

    Returns per-model CIs and pairwise difference CIs for each granularity.
    A pair is resolved when the 95% CI of A−B excludes zero.
    """
    granularities = granularities or GRANULARITIES
    tags = list(packs.keys())
    N = len(next(iter(packs.values()))["recs"])
    if any(len(p["recs"]) != N for p in packs.values()):
        raise ValueError("all packs must share the same video support")

    rng = np.random.default_rng(seed)
    streams = {t: {k: [] for k in granularities} for t in tags}
    diffs = {k: {} for k in granularities}
    for i, a in enumerate(tags):
        for b in tags[i + 1 :]:
            for k in granularities:
                diffs[k][(a, b)] = []

    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        vals = {t: metric_on_idx(packs[t], idx) for t in tags}
        for t in tags:
            for k in granularities:
                v = vals[t].get(k)
                if v is not None:
                    streams[t][k].append(v)
        for i, a in enumerate(tags):
            for b in tags[i + 1 :]:
                for k in granularities:
                    va, vb = vals[a].get(k), vals[b].get(k)
                    if va is not None and vb is not None:
                        diffs[k][(a, b)].append(va - vb)

    per_model = {t: {k: _ci(streams[t][k]) for k in granularities} for t in tags}
    paired = {}
    for k in granularities:
        paired[k] = {}
        for (a, b), arr in diffs[k].items():
            lo, hi = _ci(arr)
            mean = float(np.mean(arr)) if arr else None
            resolved = lo is not None and hi is not None and (hi < 0 or lo > 0)
            winner = None
            if resolved:
                winner = a if mean > 0 else b
            paired[k][f"{a}-{b}"] = {
                "model_a": a,
                "model_b": b,
                "diff_mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "resolved": resolved,
                "winner": winner,
                "n_boot_ok": len(arr),
            }
    return per_model, paired


def resolve_at_granularity(packs, granularity, n_boot=1000, seed=42):
    """
    Pairwise resolution for a single granularity.

    Returns point estimates, paired diffs, and summary counts.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}")

    points = {t: point_metrics(p) for t, p in packs.items()}
    _, paired_all = paired_bootstrap(
        packs, n_boot=n_boot, seed=seed, granularities=(granularity,)
    )
    paired = paired_all[granularity]
    n_resolved = sum(1 for r in paired.values() if r["resolved"])
    n_pairs = len(paired)
    vals = [points[t][granularity] for t in packs if points[t][granularity] is not None]
    spread = (max(vals) - min(vals)) if vals else None
    return {
        "granularity": granularity,
        "points": {t: points[t][granularity] for t in packs},
        "paired": paired,
        "n_resolved": n_resolved,
        "n_pairs": n_pairs,
        "spread": spread,
    }
