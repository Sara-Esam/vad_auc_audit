# UCF-Crime scene-factor annotations

Twelve binary scene factors on UCF-Crime. Each video is assigned to one pole
of a factor, or left unmarked (ambiguous) for that factor only. Ambiguous
videos are excluded from that factor's comparison and retained for others.

Manual labels cover recording properties and scene content. Resolution and
motion energy are derived from frame statistics (Laplacian variance and
frame-difference magnitude on normal segments). A label is assigned only when
the criterion holds for most of the video.

## Factors (Table S1)

| Factor | Poles (A / B) | Criterion | Support |
|---|---|---|---|
| Color encoding | colored / grayscale | RGB vs IR/nighttime grayscale | 757 / 142 |
| Spatial resolution | low-res / high-res | Laplacian variance, median split | 720 / 606 |
| Camera continuity | singleview / multiview | one continuous camera vs cuts/zooms | 373 / 163 |
| Camera elevation | highway / street-sidewalk | elevated mount vs eye-level | 124 / 112 |
| Field of view | close-view / wide-view | tight framing vs wide area | 535 / 652 |
| Motion energy | low-motion / high-motion | frame-diff magnitude, terciles | 700 / 626 |
| Number of people | 1-4persons / 4+persons | ≤4 vs 5+ visible people | 764 / 384 |
| Setting | outdoor / indoor | exterior vs building interiors | 302 / 403 |
| Viewpoint | shop-shelves / shop-cashier | in-store camera orientation | 104 / 128 |
| Location (residential / commercial) | private-house-facade / shop-shelves | house facade vs shops | 121 / 104 |
| Location (residential / fuel) | private-house-facade / gas-station | shared residential pole | 121 / 26 |
| Location (fuel / commercial) | shop-shelves / gas-station | shared poles | 104 / 26 |
| Concurrent activity | singleactivity / multiactivity | one vs several simultaneous activities | 579 / 584 |

Concurrent activity is a composition control: its poles correlate with anomaly
class by construction. Location factors that share poles are not independent.

## Files

- `scene_factors.json` — per-video tag lists for the annotated pool
- `lists/*.list` — pole membership used by the audit (one video id per line)
- `scene_pairs.json` — factor panel with pole A/B list filenames
