"""Probe-feature loading and zero-shot prototype scores."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import ann_video_frame_counts, canonical_id, sort_by_ann_file
from .metrics import category_of, cosine_rows, downsample_gt


def _load_feature_file(fpath, multi_crop_reduce="mean"):
    if fpath.endswith(".pt"):
        import torch

        obj = torch.load(fpath, map_location="cpu")
        if isinstance(obj, dict):
            if "frame_feat" in obj:
                feat = obj["frame_feat"]
            elif "embeddings" in obj:
                feat = obj["embeddings"]
            else:
                feat = next(iter(obj.values()))
        else:
            feat = obj
        if hasattr(feat, "detach"):
            feat = feat.detach().cpu().numpy()
    else:
        feat = np.load(fpath)
    feat = np.asarray(feat)
    if feat.ndim == 3:
        if multi_crop_reduce == "mean":
            feat = feat.mean(axis=0)
        elif multi_crop_reduce == "first":
            feat = feat[0]
        else:
            feat = feat[np.random.randint(0, feat.shape[0])]
    elif feat.ndim > 3:
        feat = feat.reshape(-1, feat.shape[-1])
    return feat.astype(np.float32)


def load_probe_features(
    feature_path,
    gt_array,
    ann_path,
    snippet_length=16,
    file_suffix=None,
    multi_crop_reduce="mean",
    frame_level=False,
    video_allowlist=None,
):
    """
    Load per-video probe features aligned to a flat GT array ordered by ann_path.

    Returns
    -------
    feats : {canonical_id: (T, D)}
    gts   : {canonical_id: frame-level labels}
    """
    feature_path = Path(feature_path)
    counts = ann_video_frame_counts(ann_path)
    feats, gts = {}, {}
    frame_count = 0

    for item in sort_by_ann_file(ann_path):
        if video_allowlist is not None and item not in video_allowlist:
            frame_count += counts[item]
            continue

        candidate = item.replace(".npy", file_suffix) if file_suffix else item
        fpath = feature_path / candidate
        if not fpath.exists():
            alt = feature_path / item
            if alt.exists():
                fpath = alt
            else:
                continue

        try:
            feat = _load_feature_file(str(fpath), multi_crop_reduce)
        except Exception:
            continue

        T = len(feat)
        video_frame_len = counts.get(
            item, T if frame_level else T * snippet_length
        )
        gt_slice = np.asarray(gt_array[frame_count : frame_count + video_frame_len], dtype=float)
        frame_count += video_frame_len

        cid = canonical_id(item)
        feats[cid] = feat
        gts[cid] = gt_slice

    return feats, gts


def load_flat_gt(gt_path, ann_path):
    """Load flat or nested GT as a flat array in annotation order."""
    if str(gt_path).endswith(".json"):
        with open(gt_path) as fh:
            import json

            nested = json.load(fh)
        counts = ann_video_frame_counts(ann_path)
        chunks = []
        for item in sort_by_ann_file(ann_path):
            cid = canonical_id(item)
            n = counts[item]
            found = None
            for _cls, vids in nested.items():
                for vid, arr in vids.items():
                    if canonical_id(vid) == cid:
                        found = np.asarray(arr, dtype=float).ravel()
                        break
                if found is not None:
                    break
            if found is None:
                found = np.zeros(n, dtype=float)
            elif len(found) != n:
                found = found[:n] if len(found) > n else np.pad(found, (0, n - len(found)))
            chunks.append(found)
        return np.concatenate(chunks)

    arr = np.load(gt_path, allow_pickle=True)
    if arr.dtype == object and arr.ndim == 0:
        raise ValueError("nested GT dict given; pass a flat array or JSON")
    return np.asarray(arr, dtype=float).ravel()


def mean_prototypes(feats, gts):
    """Global anomaly / normal means from snippet-level GT."""
    abn_sum = nor_sum = None
    n_abn = n_nor = 0
    for cid, X in feats.items():
        if cid not in gts:
            continue
        row_gt = downsample_gt(gts[cid], len(X))
        if abn_sum is None:
            d = X.shape[1]
            abn_sum = np.zeros(d, dtype=np.float64)
            nor_sum = np.zeros(d, dtype=np.float64)
        abn = X[row_gt > 0]
        nor = X[row_gt == 0]
        if len(abn):
            abn_sum += abn.astype(np.float64).sum(0)
            n_abn += len(abn)
        if len(nor):
            nor_sum += nor.astype(np.float64).sum(0)
            n_nor += len(nor)
    if n_abn == 0 or n_nor == 0:
        raise ValueError("need both anomaly and normal snippets for prototypes")
    return abn_sum / n_abn, nor_sum / n_nor, n_abn, n_nor


def category_prototypes(feats, gts):
    """Per-category anomaly means + one global normal mean."""
    cat_sum, cat_count = {}, defaultdict(int)
    nor_sum = None
    n_nor = 0
    for cid, X in feats.items():
        if cid not in gts:
            continue
        cat = category_of(cid)
        row_gt = downsample_gt(gts[cid], len(X))
        if nor_sum is None:
            nor_sum = np.zeros(X.shape[1], dtype=np.float64)
        abn_idx = np.where(row_gt > 0)[0]
        nor_idx = np.where(row_gt == 0)[0]
        if len(abn_idx) and cat != "Normal":
            if cat not in cat_sum:
                cat_sum[cat] = np.zeros(X.shape[1], dtype=np.float64)
            cat_sum[cat] += X[abn_idx].astype(np.float64).sum(0)
            cat_count[cat] += len(abn_idx)
        if len(nor_idx):
            nor_sum += X[nor_idx].astype(np.float64).sum(0)
            n_nor += len(nor_idx)
    if n_nor == 0 or not cat_sum:
        raise ValueError("could not build category prototypes")
    cat_proto = {c: cat_sum[c] / cat_count[c] for c in cat_sum}
    return cat_proto, nor_sum / n_nor


def score_global_zs(feats, gts, proto_feats=None, proto_gts=None):
    """Contrastive cosine scores with global prototypes."""
    src_f = proto_feats if proto_feats is not None else feats
    src_g = proto_gts if proto_gts is not None else gts
    mu_abn, mu_nor, _, _ = mean_prototypes(src_f, src_g)
    scores = {}
    for cid, X in feats.items():
        scores[cid] = cosine_rows(X, mu_abn) - cosine_rows(X, mu_nor)
    return scores


def score_category_zs(feats, gts, proto_feats=None, proto_gts=None):
    """Contrastive cosine scores with per-category anomaly prototypes."""
    src_f = proto_feats if proto_feats is not None else feats
    src_g = proto_gts if proto_gts is not None else gts
    cat_proto, mu_nor = category_prototypes(src_f, src_g)
    scores = {}
    for cid, X in feats.items():
        cat = category_of(cid)
        if cat == "Normal" or cat not in cat_proto:
            # normals / missing cats: score vs global normal only (negated)
            scores[cid] = -cosine_rows(X, mu_nor)
        else:
            scores[cid] = cosine_rows(X, cat_proto[cat]) - cosine_rows(X, mu_nor)
    return scores


def _loo_proto_matrix(X, idx, total_sum):
    """(T, D) prototypes; leave-one-out on rows in idx."""
    T, D = X.shape
    n = len(idx)
    full = total_sum / n
    proto = np.broadcast_to(full, (T, D)).copy()
    if n >= 2:
        proto[idx] = (total_sum - X[idx]) / (n - 1)
    return proto


def within_video_loo_aucs(feats, gts, min_anomaly=2):
    """Per-video leave-one-out ZS AUC. Returns {cid: auc}."""
    out = {}
    for cid, X in feats.items():
        if cid not in gts:
            continue
        X = np.asarray(X, dtype=np.float64)
        row_gt = downsample_gt(gts[cid], len(X))
        abn_idx = np.where(row_gt > 0)[0]
        nor_idx = np.where(row_gt == 0)[0]
        if len(abn_idx) < min_anomaly or len(nor_idx) < 1:
            continue
        abn_sum = X[abn_idx].sum(0)
        nor_sum = X[nor_idx].sum(0)
        ap = _loo_proto_matrix(X, abn_idx, abn_sum)
        np_ = _loo_proto_matrix(X, nor_idx, nor_sum)
        # row-wise cosine
        score = np.array(
            [
                (np.dot(X[t], ap[t]) / (max(np.linalg.norm(X[t]), 1e-12) * max(np.linalg.norm(ap[t]), 1e-12)))
                - (np.dot(X[t], np_[t]) / (max(np.linalg.norm(X[t]), 1e-12) * max(np.linalg.norm(np_[t]), 1e-12)))
                for t in range(len(X))
            ],
            dtype=np.float64,
        )
        from .metrics import auc

        a = auc(row_gt, score)
        if a is not None:
            out[cid] = a
    return out
