# Auditing Frame-Level AUC in WSVAD

Evaluation protocol and UCF-Crime scene-factor annotations for:

> *Auditing Frame-Level AUC in Weakly Supervised Video Anomaly Detection:
> Granularity, Resolution, and Scene Bias*

Computes metrics from **existing predictions** or **pre-classifier probe features**. No training code.

## Layout

```
annotations/     scene-factor lists, pairs, per-video tags
vad_eval/        protocol library
scripts/         CLIs
configs/         example model configs
```

## What each script does

| Script | Role |
|---|---|
| `evaluate_granularity.py` | Global / per-category / within-video AUC (`--source scores` or `probe`) |
| `evaluate_resolution.py` | Pairwise model comparison at one granularity (paired bootstrap) |
| `evaluate_scene_bias.py` | Bias-AUC on normal frames (pooled + per-video bootstrap) |
| `evaluate_scene_conditioned.py` | Scene-conditioned AUC and Δ_scene (Supp. S1.2) |
| `plot_granularity.py` | Forest plot from `results.json` |
| `plot_scene_bias.py` | Dot plot + heatmap from bias CSV |

### Scores vs probe

- **scores** — read model frame predictions; apply three pairing rules.
- **probe** — zero-shot prototype readout on pre-classifier embeddings:
  - global prototypes → global AUC
  - per-category prototypes → per-category AUC
  - leave-one-out within-video prototypes → within-video AUC

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# (1) three-granularity AUC on predictions
python scripts/evaluate_granularity.py \
  --source scores \
  --models configs/models.json \
  --annotation /path/to/UCF_Annotation.txt \
  --scope test --n-boot 1000 \
  --out outputs/granularity

# (1b) same protocol on probe features
python scripts/evaluate_granularity.py \
  --source probe \
  --models configs/models_probe.json \
  --out outputs/granularity_probe

# (2) pairwise resolution at one granularity
python scripts/evaluate_resolution.py \
  --source scores \
  --models configs/models.json \
  --annotation /path/to/UCF_Annotation.txt \
  --granularity within_video \
  --n-boot 1000 \
  --out outputs/resolution

# (3) scene bias-AUC
python scripts/evaluate_scene_bias.py \
  --pred /path/to/model_pre_dict.npy \
  --gt /path/to/model_gt_dict.npy \
  --train-gt /path/to/ucf_train_gt.json \
  --tag my_model \
  --out outputs/scene_bias

# (4) figures
python scripts/plot_granularity.py \
  --results outputs/granularity/results.json \
  --out outputs/figures

python scripts/plot_scene_bias.py \
  --csv outputs/scene_bias/scene_bias_my_model.csv \
  --out outputs/figures

# (5) scene-conditioned AUC
python scripts/evaluate_scene_conditioned.py \
  --pred /path/to/model_pre_dict.npy \
  --gt /path/to/model_gt_dict.npy \
  --train-gt /path/to/ucf_train_gt.json \
  --tag my_model --write-pair-matrix \
  --out outputs/scene_conditioned
```

See `annotations/FACTORS.md` for scene-factor definitions.

## License

MIT. UCF-Crime videos remain under their original terms; this repo redistributes scene-factor labels and evaluation code only.
