# ISAR RadarGS: Sequential Encounter Reconstruction

This repository contains the RadarGS implementation and the `skynet5d-3` acquisition used to study 3D reconstruction as chronological radar encounters accumulate. Each `K` model uses the first `K` encounters from a propagated orbit sequence. No view is selected to optimise reconstruction quality.

## Current experiment

- Scenarios: `CPA50` (59 encounters) and `CPA80` (36 encounters)
- Outputs: `K=1..10`, followed by scenario milestones
- Training: independent 12,000-iteration models from each chronological prefix
- Geometry frame: native CAD coordinates
- Angular-capacity ablation: spatial Fourier encoding stays at level 4 while direction encoding is reduced from 4 to 1
- Controls: legacy direction level 4 at `K=4` and `K=10` for both scenarios

The direction-capacity experiment follows one main observation from *When 3D Gaussian Splatting Recovers Real Surfaces*: excessive view-dependent appearance capacity can compensate for incorrect geometry. The paper studies optical 3DGS with a finite spherical-harmonic head. This code uses a nonlinear radar-scattering MLP, so reduced direction encoding is an experimental analogue rather than the paper's formal guarantee.

## Live dashboard

The read-only dashboard scans the newest checkpoint, displays the matching training preview, and plots Gaussian centres in an interactive 3D view. It does not reserve CUDA memory.

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
render_sequential_result.py final native-CAD point-cloud and LOS render export
datasets/skynet5d-3/       complete original dataset archives through Git LFS
```

Generated checkpoints and experiment outputs are intentionally stored outside the repository under `/root/autodl-fs/radar_gs_isar_enc/outputs`.
