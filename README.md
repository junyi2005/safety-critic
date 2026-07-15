# navdp-safety-critic

Reference implementation of the context-conditioned safety critic for
diffusion-based visual navigation: a diffusion generator proposes $K$ candidate
trajectories from RGB-D observations, and a learnable critic with a
trajectory-dependent clearance budget selects among them. ESDF is used only
offline to supervise the teacher; the deployed selector runs from RGB-D alone,
with no map building.

## Status

This repository implements the method as described in the paper. **It has not
yet reproduced the numbers in the paper's result tables.** Those tables were
produced by an earlier training pipeline that did not implement the critic
described here, and the corresponding runs are not reproducible from this code.
Re-running the benchmarks against this implementation is in progress; the
tables will be updated from those runs. Please treat the code — not the current
tables — as the description of what this critic does.

The components are unit- and integration-tested on synthetic scenes (see
`tests/`), and both training stages run end to end. Training on real collected
data requires the extra dependencies below.

## Method

The critic scores a trajectory $\tau$ with waypoints $\mathbf{p}_0 \dots \mathbf{p}_T$:

$$V_\phi(\tau) = V_\text{safe}(\tau) + V_\text{efficient}(\tau) + V_\text{balance}(\tau)$$

| Term | Implementation |
|---|---|
| Margin head $q_\eta$ — $d_{\min,j} = d_\text{safe} + \operatorname{softplus}(q_\eta(\mathbf{f}_j))$ | `models/scorer.py: MarginHead` |
| CBF residual — $r^\text{cbf}_j = (1-\rho)h(\mathbf{p}_j) - h(\mathbf{p}_{j+1})$ | `models/scorer.py: _cbf_loss` |
| Gated detour ratio — $\bar{w}(\tau)\,[L_\text{path}/(D_\text{chord}+\varepsilon) - 1]_+$ | `models/scorer.py: _gated_detour` |
| Softplus-reparameterized weights $\mathbf{w} = \operatorname{softplus}(\tilde{\mathbf{w}})$ | `models/scorer.py: weights` |
| Discriminator $D_\phi(\tau) = \sigma(-a\,C_\phi(\tau) + b)$ | `models/scorer.py: discriminator_logit` |
| Adversarial loss $\mathcal{L}_\text{scr}$ | `engine/train_teacher.py` |
| Selector loss $\mathcal{L}_\text{sel}$ | `engine/train_student.py` |
| A* non-expert proposal $q(\tau \mid \mathbf{p}^R, \mathbf{p}^G, \text{ESDF})$ | `data/negatives.py` |

The margin head is fed a geometry-only context feature — local clearance,
finite-difference ESDF gradient magnitude, and normalized step index — so the
budget adapts to the scene rather than to the time step alone.

### Two implementation notes

- **Unsafe count.** $V_\text{safe}$ contains $\sum_j \mathbb{I}(d_j < d_{\min,j})$,
  whose gradient w.r.t. $d_{\min}$ is zero almost everywhere. Training uses a
  sigmoid surrogate (`soft_unsafe=True`, active in `train()` mode only) so
  $q_\eta$ receives gradient; `eval()` scores with the exact hard count.
- **Margin initialization.** With a default-initialized output layer,
  $\operatorname{softplus}(0) \approx 0.69$ m — larger than the clearance
  available in ordinary indoor corridors, which would make every waypoint
  unsafe from step 0. The output bias is instead set so the budget starts at
  $d_\text{safe} + \texttt{init\_margin}$ (default 0.25 m total).

## Install

```bash
pip install -r requirements.txt
```

Two dependencies are not on PyPI and must be installed separately:

- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) —
  importable as `depth_anything.depth_anything_v2.dpt`. Needed by
  `models/backbone.py` only; the critic itself imports without it.
- [habitat-sim](https://github.com/facebookresearch/habitat-sim) — needed by
  `scripts/build_esdf.py` only, for navmesh queries.

Collect training data with the companion repository,
[habitat-lab-navdp](../habitat-lab-navdp).

## Usage

```bash
# 1. Build the scene ESDF (offline, teacher supervision only)
python scripts/build_esdf.py --scene <scene.glb> --out scene_esdf.npz

# 2. Train: diffusion generator -> teacher critic -> student selector
python scripts/train.py --esdf_npz scene_esdf.npz --data_root <dir> --num_scenes 5

# Stages are selectable; the student stage requires a trained teacher.
python scripts/train.py --esdf_npz scene_esdf.npz --stages teacher
python scripts/train.py --esdf_npz scene_esdf.npz --stages student
```

Key options: `--d_safe` (physical floor, default 0.1 m), `--rho` (CBF
conservativeness, 0.1), `--kappa` (safety-gate sharpness, 10.0),
`--k_candidates` (candidates per observation in $\mathcal{D}_\text{mix}$, 16).

## Tests

```bash
python -m pytest tests/ -v
```

Covers the ESDF query, each critic term against hand-computed values, gradient
flow to $q_\eta$ and $\tilde{\mathbf{w}}$, the A* proposal's start/goal matching
and $d_\text{safe}$ rejection, the RGB-D tensor-layout dispatch, and both
training stages end to end.

## Layout

```
navdp_safety/
  data/     dataset, ESDF build/query, A* non-expert proposals
  models/   RGB-D backbone, diffusion policy, safety critic
  engine/   train_diffusion / train_teacher / train_student, evaluation
scripts/    build_esdf.py, train.py
tests/      synthetic-scene unit and integration tests
```

## License

MIT — see [LICENSE](LICENSE).
