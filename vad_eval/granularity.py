"""Three-granularity AUC: global, per-category, within-video.

Supports two score sources:
  * model predictions (frame scores)
  * probe features (zero-shot prototype scores)
"""

from __future__ import annotations

import numpy as np

from .features import (
    score_category_zs,
    score_global_zs,
    within_video_loo_aucs,
)
from .metrics import auc, category_of, downsample_gt, fast_auc, to_snippets

MIN_ANOMALY_WV = 2
SNIPPET = 16


def build_score_records(units, snippet=SNIPPET, min_anomaly=MIN_ANOMALY_WV):
    """Build per-video records from prediction units."""
    recs = []
    for u in units:
        s_snip, g_snip = to_snippets(u["scores"], u["labels"], snippet=snippet)
        n_abn = int((g_snip > 0).sum())
        n_nor = int((g_snip == 0).sum())
        vid_auc = (
            auc(g_snip, s_snip)
            if (n_abn >= min_anomaly and n_nor >= 1)
            else None
        )
        recs.append(
            {
                "id": u["id"],
                "cat": u.get("cat") or category_of(u["id"]),
                "frame_scores": np.asarray(u["scores"], dtype=float),
                "frame_labels": np.asarray(u["labels"], dtype=float),
                "snip_scores": s_snip,
                "snip_gt": g_snip,
                "n_abn": n_abn,
                "n_nor": n_nor,
                "vid_auc": vid_auc,
            }
        )
    return recs


def build_probe_records(
    feats,
    gts,
    proto_feats=None,
    proto_gts=None,
    min_anomaly=MIN_ANOMALY_WV,
):
    """
    Build records for probe readout.

    Global / per-category use ZS scores with the matching prototype construction.
    Within-video uses leave-one-out per-video AUC (stored in vid_auc).
    """
    global_scores = score_global_zs(feats, gts, proto_feats, proto_gts)
    cat_scores = score_category_zs(feats, gts, proto_feats, proto_gts)
    loo = within_video_loo_aucs(feats, gts, min_anomaly=min_anomaly)

    recs = []
    for cid, X in feats.items():
        if cid not in gts:
            continue
        frame_gt = np.asarray(gts[cid], dtype=float)
        snip_gt = downsample_gt(frame_gt, len(X))
        # expand snippet scores to frame length for pooled frame AUC
        g_score = global_scores[cid]
        c_score = cat_scores[cid]
        # upsample snippet score to frames by repeating
        def _to_frames(snip_s, n_frames):
            if len(snip_s) == n_frames:
                return snip_s
            edges = np.linspace(0, n_frames, len(snip_s) + 1, dtype=int)
            out = np.empty(n_frames, dtype=float)
            for i in range(len(snip_s)):
                out[edges[i] : edges[i + 1]] = snip_s[i]
            return out

        n_abn = int((snip_gt > 0).sum())
        n_nor = int((snip_gt == 0).sum())
        recs.append(
            {
                "id": cid,
                "cat": category_of(cid),
                "frame_scores": _to_frames(g_score, len(frame_gt)),
                "frame_labels": frame_gt,
                "snip_scores": g_score,
                "snip_gt": snip_gt,
                "cat_frame_scores": _to_frames(c_score, len(frame_gt)),
                "cat_snip_scores": c_score,
                "n_abn": n_abn,
                "n_nor": n_nor,
                "vid_auc": loo.get(cid),
            }
        )
    return recs


def build_pack(recs, min_anomaly=MIN_ANOMALY_WV, source="scores"):
    cats = sorted({r["cat"] for r in recs if r["cat"] != "Normal"})
    cat_index = {c: i for i, c in enumerate(cats)}
    n = len(recs)
    vauc = np.full(n, np.nan)
    cat_codes = np.full(n, -1, dtype=int)
    mask_wv = np.zeros(n, dtype=bool)
    for i, r in enumerate(recs):
        if r["vid_auc"] is not None:
            vauc[i] = r["vid_auc"]
        if r["cat"] in cat_index:
            cat_codes[i] = cat_index[r["cat"]]
        if r["vid_auc"] is not None and r["n_nor"] >= 1 and r["n_abn"] >= min_anomaly:
            mask_wv[i] = True
    return {
        "source": source,
        "recs": recs,
        "cats": cats,
        "vauc": vauc,
        "cat_codes": cat_codes,
        "mask_wv": mask_wv,
    }


def _pool(pack, idx, level="frame", cat_code=None, score_key=None):
    sel = idx if cat_code is None else idx[pack["cat_codes"][idx] == cat_code]
    if sel.size == 0:
        return None
    use_cat = score_key == "category" and pack["source"] == "probe"
    if level == "frame":
        ys = [pack["recs"][i]["frame_labels"] for i in sel]
        if use_cat:
            ss = [pack["recs"][i]["cat_frame_scores"] for i in sel]
        else:
            ss = [pack["recs"][i]["frame_scores"] for i in sel]
        fn = auc
    else:
        ys = [pack["recs"][i]["snip_gt"] for i in sel]
        if use_cat:
            ss = [pack["recs"][i]["cat_snip_scores"] for i in sel]
        else:
            ss = [pack["recs"][i]["snip_scores"] for i in sel]
        fn = fast_auc
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    if y.size == 0 or not (y.min() < y.max()):
        return None
    return fn(y, s)


def _per_category(pack, idx, level="frame"):
    vals = []
    key = "category" if pack["source"] == "probe" else None
    for c_idx in range(len(pack["cats"])):
        a = _pool(pack, idx, level=level, cat_code=c_idx, score_key=key)
        if a is not None:
            vals.append(a)
    if not vals:
        return None
    return float(np.mean(vals))


def _within_video(pack, idx):
    m = pack["mask_wv"][idx]
    if not m.any():
        return None
    return float(pack["vauc"][idx][m].mean())


def point_metrics(pack):
    idx = np.arange(len(pack["recs"]))
    g = _pool(pack, idx, level="frame", score_key=None)
    pc = _per_category(pack, idx, level="frame")
    wv = _within_video(pack, idx)
    delta = (g - wv) if (g is not None and wv is not None) else None
    return {
        "global": g,
        "per_category": pc,
        "within_video": wv,
        "delta_wv": delta,
        "n_wv_videos": int(pack["mask_wv"].sum()),
        "n_categories": len(pack["cats"]),
    }


def metric_on_idx(pack, idx):
    """Bootstrap-time metrics (snippet-pooled for speed)."""
    g = _pool(pack, idx, level="snippet")
    pc = _per_category(pack, idx, level="snippet")
    wv = _within_video(pack, idx)
    return {"global": g, "per_category": pc, "within_video": wv}


def align_packs(model_units_or_recs, source="scores"):
    """Intersect video ids; return packs with shared order."""
    if source == "scores":
        id_sets = [{u["id"] for u in units} for units in model_units_or_recs.values()]
    else:
        id_sets = [{r["id"] for r in recs} for recs in model_units_or_recs.values()]
    shared = sorted(set.intersection(*id_sets)) if id_sets else []
    packs = {}
    for tag, data in model_units_or_recs.items():
        if source == "scores":
            by_id = {u["id"]: u for u in data}
            recs = build_score_records([by_id[i] for i in shared])
        else:
            by_id = {r["id"]: r for r in data}
            recs = [by_id[i] for i in shared]
        packs[tag] = build_pack(recs, source=source)
    return packs, shared
