"""I/O helpers for nested prediction/GT dicts and scene-factor lists."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def canonical_id(name: str) -> str:
    name = str(name).strip()
    if name.endswith(".mp4"):
        name = name[:-4]
    if name.endswith("_x264"):
        name = name[: -len("_x264")]
    return name


def is_normal_video(cid: str, class_name: str) -> bool:
    return class_name == "Normal" or cid.lower().startswith("normal")


def load_nested_dict(path):
    path = str(path)
    if path.endswith(".json"):
        with open(path) as fh:
            return json.load(fh)
    return np.load(path, allow_pickle=True).item()


def flatten_scores(pred):
    """Map canonical id -> (class_name, score_array)."""
    out = {}
    for cls, vids in pred.items():
        for vid, arr in vids.items():
            cid = canonical_id(vid)
            if cid not in out:
                out[cid] = (cls, np.asarray(arr, dtype=float))
    return out


def flatten_gt(*dicts):
    out = {}
    for d in dicts:
        if not d:
            continue
        for cls, vids in d.items():
            for vid, arr in vids.items():
                cid = canonical_id(vid)
                g = np.asarray(arr, dtype=float)
                if g.size == 0:
                    continue
                if cid not in out:
                    out[cid] = g
    return out


def parse_annotation(annotation_path):
    """Ordered (canonical_id, frame_count) from UCF_Annotation.txt."""
    entries = []
    with open(annotation_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            vid = canonical_id(parts[0].split("/")[-1])
            entries.append((vid, int(parts[1])))
    return entries


def flat_gt_from_array(arr, annotation_path):
    """Split a flat test-GT array ordered by UCF_Annotation.txt."""
    arr = np.asarray(arr, dtype=float).ravel()
    entries = parse_annotation(annotation_path)
    snippet = None
    for s in (16, 1):
        if sum((c // s) * s for _, c in entries) == arr.size:
            snippet = s
            break
    if snippet is None:
        snippet = 1

    out = {}
    off = 0
    for vid, cnt in entries:
        L = (cnt // snippet) * snippet
        out[vid] = arr[off : off + L]
        off += L
    return out


def load_test_gt(path, annotation_path):
    if str(path).endswith(".json"):
        with open(path) as fh:
            return flatten_gt(json.load(fh))
    obj = np.load(path, allow_pickle=True)
    if obj.dtype == object and obj.ndim == 0:
        return flatten_gt(obj.item())
    if annotation_path is None:
        raise ValueError("flat GT array requires --annotation")
    return flat_gt_from_array(obj, annotation_path)


def merge_gt(test_map, train_nested):
    out = {k: v for k, v in test_map.items() if np.asarray(v).size > 0}
    for k, v in flatten_gt(train_nested).items():
        if k not in out:
            out[k] = v
    return out


def load_model_arrays(pred_path, gt_path=None, train_gt_path=None, annotation_path=None):
    pred = load_nested_dict(pred_path)
    if gt_path is None:
        gt_path = str(pred_path).replace("_pre_dict", "_gt_dict")
    test_gt = load_test_gt(gt_path, annotation_path)
    train_gt = {}
    if train_gt_path and Path(train_gt_path).is_file():
        train_gt = load_nested_dict(train_gt_path)
    return flatten_scores(pred), merge_gt(test_gt, train_gt)


def collect_video_units(flat_scores, flat_gt):
    units = []
    for cid, (cls, score) in flat_scores.items():
        gt = flat_gt.get(cid)
        if gt is None:
            if is_normal_video(cid, cls):
                gt = np.zeros(len(score), dtype=float)
            else:
                continue
        n = min(len(score), len(gt))
        if n == 0:
            continue
        y = np.asarray(gt[:n], dtype=float)
        s = np.asarray(score[:n], dtype=float)
        units.append(
            {
                "id": cid,
                "class": cls,
                "is_normal_video": is_normal_video(cid, cls),
                "scores": s,
                "labels": y,
                "n_anom_frames": int((y == 1).sum()),
                "n_norm_frames": int((y == 0).sum()),
            }
        )
    return units


def load_video_list(path):
    ids = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.add(canonical_id(line))
    return ids


def load_scene_pairs(pairs_json, lists_dir):
    """Load factor panel; return list of dicts with ids_a / ids_b."""
    lists_dir = Path(lists_dir)
    with open(pairs_json) as fh:
        specs = json.load(fh)

    pairs = []
    for sp in specs:
        path_a = lists_dir / sp["list_a"]
        path_b = lists_dir / sp["list_b"]
        ids_a = load_video_list(path_a) if path_a.is_file() else set()
        ids_b = load_video_list(path_b) if path_b.is_file() else set()
        both = ids_a & ids_b
        pairs.append(
            {
                "factor": sp["factor"],
                "pole_a": sp.get("pole_a", "A"),
                "pole_b": sp.get("pole_b", "B"),
                "ids_a": ids_a - both,
                "ids_b": ids_b - both,
                "n_ambiguous": len(both),
            }
        )
    return pairs


def official_test_ids(annotation_path):
    return {vid for vid, _ in parse_annotation(annotation_path)}


def attach_scene_membership(units, pairs):
    for u in units:
        scenes = {}
        for sp in pairs:
            cid = u["id"]
            if cid in sp["ids_a"]:
                scenes[sp["factor"]] = "A"
            elif cid in sp["ids_b"]:
                scenes[sp["factor"]] = "B"
            else:
                scenes[sp["factor"]] = None
        u["scenes"] = scenes
    return units
