# SC-GS: Support-Constraint Gaussian Splatting for ISAR 3-D Reconstruction

Code behind the IEEE TAES manuscript *Support-Constraint Gaussian Splatting (SC-GS)*. It reconstructs the 3-D structure of a space object from a sequence of ISAR images, for space domain awareness (SDA). Targets: Skynet-5D (main) and GOES-R (second target).

## Method in one page

Each Gaussian primitive i has:
- a centre μ_i and a covariance Σ_i = R S Sᵀ Rᵀ;
- a view-shared support q_i = σ(ℓ_i / T), the probability that the primitive belongs to the structure;
- an order-3 spherical-harmonic amplitude a_i(k_r) = σ(c_iᵀ Y(k_r)), with 16 coefficients per primitive.

Forward model, per view:
1. **Projection.** Each primitive is projected onto the range / cross-range image plane through the LOS geometry (k_r, k_d) and the image Jacobian. This gives a 2-D footprint.
2. **LOS visibility.** Primitives are binned in depth on a plane normal to the LOS. Transmittance is T = exp(−Σρ), and V_i is the kernel-weighted mean of T. Nearer primitives therefore attenuate farther ones.
3. **Image formation.** S = Σ_i q_i V_i a_i G_i(x), and Î = 1 − exp(−(S + b)).
4. **Loss.**
   - The image loss is evaluated only inside the echo mask: pixels with detected returns, dilated by a 5×5 max-pool.
   - A binary-entropy prior on q pushes each primitive to a clear keep/drop decision. `lambda_support_binary = 1.0` is the adopted setting.
   - The total loss is minimised with Adam.

Structural output: the centres with q ≥ 0.5. No GT is used to select them.

Training settings:
- 12,000 iterations in four stages; q is held fixed at 0.2 in stage A.
- Batch of 8 views.
- 10,000 initial centres, capped at 20,000.
- Adam with lr 1e-3, multiplied by 0.8 every 5,000 iterations; gradients clipped at norm 5.
- The optimiser is rebuilt after each density-control step.

## Where the code is

| Path | Content |
|---|---|
| `radar_gs/` | Core renderer, SH3 amplitude, LOS visibility and Gaussian model (sequential RadarGS code base) |
| `outputs/echo_mask_robustness_20261009/source_snapshot/` | Frozen source used for the paper runs: `mask_sh3_worker.py` (SC-GS), `hybrid_worker.py`, the baselines (`adapted_papers.py`, `pei_*`) and `evaluate_suite.py` (CD-L1/L2, HD95, coverage) |
| `outputs/sh3_echo_daily_recovery_20261011/source_snapshot/` | Same snapshot plus `daily_worker.py` and `multicpi_tracking.py`, used for the daily and acquisition-time runs |
| `outputs/q_ent1_20261027/` | **Final Skynet-5D suite (λ_ent = 1).** `main_suite.py` defines the runs and launches `gen_worker.py`; `ent1_suite.py` re-runs every paper configuration; `ent1_matched.py` evaluates at matched K; `los_big.py` runs the 5°/10° LOS-error tests |
| `outputs/goes_ent1_20261027/` | Final GOES-R suite (`goes_suite.py`, `collect_goes.py`) |
| `outputs/q_main_20261017/` | Earlier main suite (λ_ent = 0.3) and the figure scripts (`figs_*.py`, `acq_*.py`) |
| `outputs/q_support_20261016/`, `q_robust*_20261018/`, `q_ctrl_20261018/`, `q_speckle_20261026/`, `baseline_robust_20261026/`, `goes_main_20261025/`, `ranerf_bsat_20261018/` | Ablations on q, robustness (noise, LOS error, seeds), the control runs, coherent speckle, baseline robustness, the first GOES-R suite and the RaNeRF baseline |

Only scripts were added from `outputs/`. Run outputs, checkpoints, PLYs and data stay ignored by `.gitignore`.

## Running

Run commands from the repository root. The suites claim runs through an on-disk queue, so several GPU jobs can share one suite:

```bash
python outputs/q_ent1_20261027/main_suite.py   # launches gen_worker.py per run, then evaluate_suite.py
```

`main_suite.py` sets `PYTHONPATH=<source_snapshot>:.:outputs`. Each run writes its `config.json`, its log and its metrics into its own run folder. `collect_robust.py` / `collect_goes.py` gather these into `summary.csv`. Noise and LOS errors are injected into the training views only, through `GEN_PERTURB` in `gen_worker.py`. A LOS error is a Rodrigues rotation by θ ~ N(0, σ²) applied to (k_r, k_d, n_v).

## Results (λ_ent = 1, matched point count)

| Target | K | CD-L1 (m) | CD-L2 (m²) | HD95 (m) | Coverage @0.1 m |
|---|---|---|---|---|---|
| Skynet-5D | 13,126 | 0.0494 | 0.00465 | 0.162 | 99.0 % |
| GOES-R | 12,115 | 0.0375 | 0.00205 | — | 98.4 % |

Matched K: each Gaussian baseline keeps its K highest-weight (α) centres. K is set by SC-GS's own q ≥ 0.5 output, and RaNeRF is given 20,000 voxels. On Skynet-5D, SC-GS lowers CD-L1 by 38 % relative to the strongest baseline. Retraining from two further random seeds gives CD-L1 = 0.0495 ± 0.0003 m.
