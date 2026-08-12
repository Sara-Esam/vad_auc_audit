# Auditing Frame-Level AUC in WSVAD

Public evaluation protocol and UCF-Crime scene-factor annotations for:

> *Auditing Frame-Level AUC in Weakly Supervised Video Anomaly Detection:
> Granularity, Resolution, and Scene Bias*

The code reads **existing frame-level predictions**. No model training is required.

## Contents

```
annotations/          scene-factor lists, pairs, and per-video tags
vad_eval/             protocol library
scripts/              command-line entry points
configs/              example model path config
```

### Annotations

Twelve binary scene factors on UCF-Crime (recording properties and scene content).
See [`annotations/FACTORS.md`](annotations/FACTORS.md).

- `annotations/lists/*.list` — pole membership used in the paper
- `annotations/scene_pairs.json` — factor panel
- `annotations/scene_factors.json` — per-video tag lists

### Protocol

1. **Granularity-aware AUC** — global, per-category, and within-video readout of the same scores, with a paired video bootstrap for confidence intervals and resolved model comparisons.
2. **Scene bias-AUC** — on normal frames only, treat scene pole A vs B as a binary label and measure how much the anomaly score separates recordings by scene (`0.5` = unbiased).
3. **Scene-conditioned AUC** — anomaly-vs-normal AUC after removing cross-pole pairs; `Delta_scene = ReferenceAUC - SceneConditionedAUC`.

## Setup

```bash
pip install -r requirements.txt
```

Prediction format: nested dict `class_name -> video_id -> 1-D score array`, saved as `.npy` (`np.save(..., dict)`). GT uses the same nesting, or a flat test-GT array ordered by `UCF_Annotation.txt`.

## Usage

Copy and edit `configs/models.example.json`, then:

```bash
# three-granularity AUC + paired bootstrap (official 290-video test)
python scripts/evaluate_granularity.py \
  --models configs/models.json \
  --annotation /path/to/UCF_Annotation.txt \
  --scope test \
  --n-boot 1000 \
  --out outputs/granularity

# scene bias-AUC (full prediction pool)
python scripts/evaluate_scene_bias.py \
  --pred /path/to/model_pre_dict.npy \
  --gt /path/to/model_gt_dict.npy \
  --train-gt /path/to/ucf_train_gt.json \
  --tag my_model \
  --out outputs/scene_bias

# scene-conditioned AUC / Delta_scene
python scripts/evaluate_scene_conditioned.py \
  --pred /path/to/model_pre_dict.npy \
  --gt /path/to/model_gt_dict.npy \
  --train-gt /path/to/ucf_train_gt.json \
  --tag my_model \
  --write-pair-matrix \
  --out outputs/scene_conditioned
```

## Prediction / GT conventions

Video ids are normalized by stripping `.mp4` and a trailing `_x264`. Scene lists may use either form. For the full-pool scene audit, provide train-frame GT so normal frames inside training videos are labelled.

## Citation

If you use the annotations or protocol, please cite the paper (bibtex to be added upon publication).

## License

MIT — see `LICENSE`. UCF-Crime videos remain under their original terms; this repository redistributes only scene-factor labels and evaluation code.
