"""Three-granularity AUC readout and paired video bootstrap."""

from __future__ import annotations

import numpy as np

from .metrics import auc, category_of, fast_auc, to_snippets


def build_records(units, snippet=16):
    recs = []
    for u in units:
        s_snip, g_snip = to_snippets(u["scores"], u["labels"], snippet=snippet)
        n_abn = int((g_snip > 0).sum())
        n_nor = int((g_snip == 0).sum())
        vid_auc = auc(g_snip, s_snip) if (n_abn > 0 and n_nor > 0) else None
        recs.append(
            {
                "id": u["id"],
                "cat": category_of(u["id"]),
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


def build_pack(recs, min_anomaly_wv=2):
    cats = sorted({r["cat"] for r in recs if r["cat"] != "Normal"})
    cat_index = {c: i for i, c in enumerate(cats)}
    n = len(recs)
    vauc = np.full(n, np.nan)
    cat_codes = np.full(n, -1, dtype=int)
    mask_wv = np.zeros(n, dtype=bool)
    snip_scores, snip_gt = [], []
    for i, r in enumerate(recs):
        snip_scores.append(r["snip_scores"])
        snip_gt.append(r["snip_gt"])
        if r["vid_auc"] is not None:
            vauc[i] = r["vid_auc"]
        if r["cat"] in cat_index:
            cat_codes[i] = cat_index[r["cat"]]
        if r["vid_auc"] is not None and r["n_nor"] >= 1 and r["n_abn"] >= min_anomaly_wv:
            mask_wv[i] = True
    return {
        "recs": recs,
        "cats": cats,
        "vauc": vauc,
        "cat_codes": cat_codes,
        "mask_wv": mask_wv,
        "snip_scores": snip_scores,
        "snip_gt": snip_gt,
    }


def _pooled_cat(pack, idx, cat_code, level="snippet"):
    sel = idx[pack["cat_codes"][idx] == cat_code]
    if sel.size == 0:
        return None
    if level == "frame":
        ys = [pack["recs"][i]["frame_labels"] for i in sel]
        ss = [pack["recs"][i]["frame_scores"] for i in sel]
        fn = auc
    else:
        ys = [pack["snip_gt"][i] for i in sel]
        ss = [pack["snip_scores"][i] for i in sel]
        fn = fast_auc
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    if y.size == 0 or not (y.min() < y.max()):
        return None
    return fn(y, s)


def _global(pack, idx, level="snippet"):
    if level == "frame":
        ys = [pack["recs"][i]["frame_labels"] for i in idx]
        ss = [pack["recs"][i]["frame_scores"] for i in idx]
        fn = auc
    else:
        ys = [pack["snip_gt"][i] for i in idx]
        ss = [pack["snip_scores"][i] for i in idx]
        fn = fast_auc
    if not ys:
        return None
    return fn(np.concatenate(ys), np.concatenate(ss))


def _per_category(pack, idx, level="snippet"):
    vals = {}
    for c_idx, c in enumerate(pack["cats"]):
        a = _pooled_cat(pack, idx, c_idx, level=level)
        if a is not None:
            vals[c] = a
    if not vals:
        return None, {}
    return float(np.mean(list(vals.values()))), vals


def _within_video(pack, idx):
    m = pack["mask_wv"][idx]
    if not m.any():
        return None, 0
    return float(pack["vauc"][idx][m].mean()), int(m.sum())


def point_metrics(pack):
    idx = np.arange(len(pack["recs"]))
    g = _global(pack, idx, level="frame")
    pc, cat_auc = _per_category(pack, idx, level="frame")
    wv, n_wv = _within_video(pack, idx)
    delta = (g - wv) if (g is not None and wv is not None) else None
    return {
        "global": g,
        "per_category": pc,
        "within_video": wv,
        "delta_wv": delta,
        "n_wv_videos": n_wv,
        "n_categories": len(cat_auc),
        "cat_auc": cat_auc,
    }


def _metric_snip(pack, idx):
    g = _global(pack, idx, level="snippet")
    pc, _ = _per_category(pack, idx, level="snippet")
    wv, _ = _within_video(pack, idx)
    dl = (g - wv) if (g is not None and wv is not None) else None
    return {"global": g, "per_category": pc, "within_video": wv, "delta_wv": dl}


def _ci(arr):
    if not arr:
        return None, None
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def paired_bootstrap(packs, n_boot=1000, seed=42):
    """Shared-index video bootstrap across models; CIs on values and differences."""
    tags = list(packs.keys())
    N = len(next(iter(packs.values()))["recs"])
    if any(len(p["recs"]) != N for p in packs.values()):
        raise ValueError("all packs must share the same video support / order")

    rng = np.random.default_rng(seed)
    keys = ("global", "per_category", "within_video", "delta_wv")
    streams = {t: {k: [] for k in keys} for t in tags}
    diffs = {k: {} for k in keys}
    for i, a in enumerate(tags):
        for b in tags[i + 1 :]:
            for k in keys:
                diffs[k][(a, b)] = []

    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        vals = {t: _metric_snip(packs[t], idx) for t in tags}
        for t in tags:
            for k, v in vals[t].items():
                if v is not None:
                    streams[t][k].append(v)
        for i, a in enumerate(tags):
            for b in tags[i + 1 :]:
                for k in keys:
                    va, vb = vals[a][k], vals[b][k]
                    if va is not None and vb is not None:
                        diffs[k][(a, b)].append(va - vb)

    per_model = {t: {k: _ci(streams[t][k]) for k in keys} for t in tags}
    paired = {}
    for k in keys:
        paired[k] = {}
        for (a, b), arr in diffs[k].items():
            lo, hi = _ci(arr)
            mean = float(np.mean(arr)) if arr else None
            resolved = lo is not None and hi is not None and (hi < 0 or lo > 0)
            paired[k][f"{a}-{b}"] = {
                "model_a": a,
                "model_b": b,
                "diff_mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "resolved": resolved,
                "n_boot_ok": len(arr),
            }
    return per_model, paired


def count_resolved(paired, metric="global"):
    """Number of model pairs whose paired CI excludes zero."""
    rows = paired.get(metric, {})
    return sum(1 for r in rows.values() if r["resolved"]), len(rows)


def align_units_by_ids(model_units):
    """Intersect video ids across models; return packs keyed by tag with shared order."""
    id_sets = [{u["id"] for u in units} for units in model_units.values()]
    shared = set.intersection(*id_sets) if id_sets else set()
    ordered = sorted(shared)
    packs = {}
    for tag, units in model_units.items():
        by_id = {u["id"]: u for u in units}
        packs[tag] = build_pack(build_records([by_id[i] for i in ordered]))
    return packs, ordered
