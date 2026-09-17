# ISAR RadarGS: Sequential Encounter Reconstruction

This repository contains the RadarGS implementation used to study 3D reconstruction as chronological radar encounters accumulate. Each `K` model uses a physically ordered encounter prefix; the newer continual-learning experiments can also warm-start from the preceding stage and control historical replay.

## Current experiment

- Scenarios: `CPA50` (59 encounters) and `CPA80` (36 encounters)
- Outputs: `K=1..10`, followed by scenario milestones
- Training: independent 12,000-iteration models from each chronological prefix
- Geometry frame: native CAD coordinates
- Angular-capacity ablation: spatial Fourier encoding stays at level 4 while direction encoding is reduced from 4 to 1
- Controls: legacy direction level 4 at `K=4` and `K=10` for both scenarios

The direction-capacity experiment follows one main observation from *When 3D Gaussian Splatting Recovers Real Surfaces*: excessive view-dependent appearance capacity can compensate for incorrect geometry. The paper studies optical 3DGS with a finite spherical-harmonic head. This code uses a nonlinear radar-scattering MLP, so reduced direction encoding is an experimental analogue rather than the paper's formal guarantee.

## `skynet5d-4` continual experiments

The current code additionally supports the eight-encounter `skynet5d-4` study:

- independent cumulative-prefix models for `K=1..8`;
- sequential warm starts where stage `K` inherits stage `K-1`;
- fixed 25% historical replay;
- fixed 50% replay with optional protection of inherited high-support Gaussians;
- natural proportional sampling from all accumulated frames;
- current-encounter and cumulative evaluation after every stage.

Prepare and run these protocols with `prepare_skynet5d4.py`, `prepare_incremental.py`, `prepare_replay.py`, and their matching queue scripts. Dataset paths and output roots are explicit in those experiment drivers and should be adjusted for a new host.

## Live dashboard

The read-only dashboard scans the newest checkpoint, displays the matching training preview, and plots Gaussian centres in an interactive 3D view. The point-cloud panel includes a live support-threshold slider and 0.1/0.3/0.5/0.7 presets. Filtering occurs in the browser and does not reserve CUDA memory or modify checkpoints.

On the server:

```bash
./start_live_dashboard.sh
```

On a local computer, open an SSH tunnel:

```bash
ssh -N -L 6006:127.0.0.1:6006 -p 47049 root@connect.nmb2.seetacloud.com
```

Then open <http://127.0.0.1:6006>. The browser refreshes run status every 15 seconds and reloads point-cloud data only when a new checkpoint appears.

## Dataset

The two complete simulator archives are stored with Git LFS under `datasets/skynet5d-3`:

```text
CPA50.zip  sha256: 3d27bd531b2680ee73effef6930e356428e2a61842c88aa90f0124d850f91358
CPA80.zip  sha256: 96f250f8b84cc19c1c3ee385a52fc446f585e9d415ae1927a9de52db8aa08e27
```

Install Git LFS before cloning or run `git lfs pull` after installation. The archives contain per-frame amplitude-only ISAR PNGs, recorded LOS and cross-range vectors, propagated encounter metadata, pulse tables, and chronological batch manifests. The simulator does not provide complex I/Q phase.

## Run the queued experiment

```bash
python prepare_sequential.py
python run_sequential_queue.py
```

`prepare_sequential.py` audits all folders, verifies frame counts and view-vector conventions, creates chronological K prefixes, and recovers the native CAD GT frame using the documented inverse rigid transform. The queue executes at most six models concurrently and evaluates/renders each completed model.

## Repository layout

```text
radar_gs/                  model, renderer, losses, data loader and trainer
configs/                   base experiment configuration
live_dashboard/            browser interface
prepare_sequential.py      acquisition audit and cumulative-K job builder
run_sequential_queue.py    bounded persistent training/evaluation queue
prepare_incremental.py     sequential warm-start K=1..8 job builder
prepare_replay.py          fixed replay/protection job builder
prepare_proportional.py    natural accumulated-frame sampling job builder
run_replay.py              bounded replay chains and evaluation
run_proportional.py        proportional replay chain and evaluation
render_sequential_result.py final native-CAD point-cloud and LOS render export
datasets/skynet5d-3/       complete original dataset archives through Git LFS
```

Generated checkpoints and experiment outputs are intentionally stored outside the repository under `/root/autodl-fs/radar_gs_isar_enc/outputs`.
