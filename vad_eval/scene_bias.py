"""Scene bias-AUC on normal frames (Section 4)."""

from __future__ import annotations

import numpy as np

from .io import is_normal_video
from .metrics import auc


def collect_normal_frames(ids, flat_scores, flat_gt):
    frame_scores, frame_strata = [], []
    vid_mean = []
    n_vid = 0
    for cid in ids:
        if cid not in flat_scores:
            continue
        cls, score = flat_scores[cid]
        gt = flat_gt.get(cid)
        if gt is None:
            if is_normal_video(cid, cls):
                gt = np.zeros(len(score), dtype=float)
            else:
                continue
        n = min(len(score), len(gt))
        if n == 0:
            continue
        score = np.asarray(score[:n], dtype=float)
        gt = np.asarray(gt[:n], dtype=float)
        stratum = "norm" if is_normal_video(cid, cls) else "anom"
        normal = score[gt == 0]
        if normal.size == 0:
            continue
        frame_scores.append(normal)
        frame_strata.append(np.full(normal.size, stratum))
        vid_mean.append(float(normal.mean()))
        n_vid += 1

    return {
        "frame_scores": np.concatenate(frame_scores) if frame_scores else np.array([]),
        "frame_strata": np.concatenate(frame_strata) if frame_strata else np.array([]),
        "vid_mean": np.asarray(vid_mean, dtype=float),
        "n_vid": n_vid,
    }


def bias_auc(scores_a, scores_b):
    """Pole A = positive class. AUC = 0.5 means no scene preference."""
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if scores_a.size == 0 or scores_b.size == 0:
        return None
    labels = np.concatenate([np.ones(scores_a.size), np.zeros(scores_b.size)])
    scores = np.concatenate([scores_a, scores_b])
    return auc(labels, scores)


def video_bias_bootstrap(means_a, means_b, n_boot=1000, seed=42, null_band=0.05):
    """
    Video-level bias-AUC with stratified pole bootstrap.

    Returns point estimate, 95% CI, and p_bias = fraction of replicates
    with |b - 0.5| > null_band.
    """
    sa = np.asarray(means_a, dtype=float)
    sb = np.asarray(means_b, dtype=float)
    point = bias_auc(sa, sb)
    if point is None or n_boot <= 0 or sa.size == 0 or sb.size == 0:
        return {
            "bias_auc": point,
            "ci_lo": None,
            "ci_hi": None,
            "p_bias": None,
            "boot_mean": None,
            "n_boot_ok": 0,
        }

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ia = rng.integers(0, sa.size, size=sa.size)
        ib = rng.integers(0, sb.size, size=sb.size)
        b = bias_auc(sa[ia], sb[ib])
        if b is not None:
            boots.append(b)
    if not boots:
        return {
            "bias_auc": point,
            "ci_lo": None,
            "ci_hi": None,
            "p_bias": None,
            "boot_mean": None,
            "n_boot_ok": 0,
        }

    boots = np.asarray(boots, dtype=float)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "bias_auc": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_bias": float(np.mean(np.abs(boots - 0.5) > null_band)),
        "boot_mean": float(boots.mean()),
        "n_boot_ok": int(boots.size),
    }


def factor_bias(ids_a, ids_b, flat_scores, flat_gt, n_boot=1000, seed=42):
    A = collect_normal_frames(ids_a, flat_scores, flat_gt)
    B = collect_normal_frames(ids_b, flat_scores, flat_gt)
    pooled = bias_auc(A["frame_scores"], B["frame_scores"])
    video = video_bias_bootstrap(A["vid_mean"], B["vid_mean"], n_boot=n_boot, seed=seed)

    strata = {}
    for name in ("anom", "norm"):
        sa = A["frame_scores"][A["frame_strata"] == name] if A["frame_scores"].size else np.array([])
        sb = B["frame_scores"][B["frame_strata"] == name] if B["frame_scores"].size else np.array([])
        strata[name] = bias_auc(sa, sb)

    return {
        "n_vid_a": A["n_vid"],
        "n_vid_b": B["n_vid"],
        "n_frames_a": int(A["frame_scores"].size),
        "n_frames_b": int(B["frame_scores"].size),
        "bias_auc_pooled": pooled,
        "bias_auc_anom": strata["anom"],
        "bias_auc_norm": strata["norm"],
        "bias_auc_video": video["bias_auc"],
        "bias_auc_video_ci_lo": video["ci_lo"],
        "bias_auc_video_ci_hi": video["ci_hi"],
        "p_bias": video["p_bias"],
        "bias_strength": None if pooled is None else abs(pooled - 0.5),
    }
