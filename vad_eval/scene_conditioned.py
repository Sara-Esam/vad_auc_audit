"""Scene-conditioned anomaly AUC and control gap (Supp. S1.2)."""

from __future__ import annotations

import numpy as np

from .metrics import auc, mann_whitney_auc


def _support(units):
    return {
        "n_videos": len(units),
        "n_anom_videos": sum(1 for u in units if u["n_anom_frames"] > 0),
        "n_norm_videos": sum(1 for u in units if u["n_norm_frames"] > 0),
        "n_anom_frames": sum(u["n_anom_frames"] for u in units),
        "n_norm_frames": sum(u["n_norm_frames"] for u in units),
    }


def _pooled_auc(units):
    if not units:
        return None
    s = np.concatenate([u["scores"] for u in units])
    y = np.concatenate([u["labels"] for u in units])
    return auc(y, s)


def scene_conditioned(units, factor):
    """
    ReferenceAUC, within-pole AUCs, SceneConditionedAUC, and Delta_scene.

    Delta_scene = ReferenceAUC - 0.5 * (AUC_A + AUC_B).
    """
    dj = [u for u in units if u["scenes"].get(factor) in ("A", "B")]
    ua = [u for u in dj if u["scenes"].get(factor) == "A"]
    ub = [u for u in dj if u["scenes"].get(factor) == "B"]

    ref = _pooled_auc(dj)
    auc_a = _pooled_auc(ua) if _support(ua)["n_anom_frames"] and _support(ua)["n_norm_frames"] else None
    auc_b = _pooled_auc(ub) if _support(ub)["n_anom_frames"] and _support(ub)["n_norm_frames"] else None
    can = auc_a is not None and auc_b is not None
    cond = 0.5 * (auc_a + auc_b) if can else None
    gap = (ref - cond) if (can and ref is not None) else None

    return {
        "factor": factor,
        "n_dj": len(dj),
        "ReferenceAUC": ref,
        "AUC_A": auc_a,
        "AUC_B": auc_b,
        "SceneConditionedAUC": cond,
        "Delta_scene": gap,
        "support_A": _support(ua),
        "support_B": _support(ub),
    }


def pair_matrix(units, factor):
    """Four pair-type AUCs and weights on D_j (Table S7)."""
    buckets = {"A": {"pos": [], "neg": []}, "B": {"pos": [], "neg": []}}
    for u in units:
        pole = u["scenes"].get(factor)
        if pole not in ("A", "B"):
            continue
        y, s = u["labels"], u["scores"]
        buckets[pole]["pos"].append(s[y == 1])
        buckets[pole]["neg"].append(s[y == 0])

    for p in ("A", "B"):
        buckets[p]["pos"] = (
            np.concatenate(buckets[p]["pos"]) if buckets[p]["pos"] else np.array([])
        )
        buckets[p]["neg"] = (
            np.concatenate(buckets[p]["neg"]) if buckets[p]["neg"] else np.array([])
        )

    n_pos = {p: int(buckets[p]["pos"].size) for p in ("A", "B")}
    n_neg = {p: int(buckets[p]["neg"].size) for p in ("A", "B")}
    n_pos_tot = n_pos["A"] + n_pos["B"]
    n_neg_tot = n_neg["A"] + n_neg["B"]
    if n_pos_tot == 0 or n_neg_tot == 0:
        return None

    out = {"factor": factor, "n_pos": n_pos, "n_neg": n_neg}
    recon = 0.0
    for sp in ("A", "B"):
        for sn in ("A", "B"):
            w = (n_pos[sp] * n_neg[sn]) / (n_pos_tot * n_neg_tot)
            a = mann_whitney_auc(buckets[sp]["pos"], buckets[sn]["neg"])
            out[f"w_{sp}{sn}"] = float(w)
            out[f"AUC_{sp}{sn}"] = a
            if a is not None:
                recon += w * a
    out["auc_reconstructed"] = float(recon)
    return out
