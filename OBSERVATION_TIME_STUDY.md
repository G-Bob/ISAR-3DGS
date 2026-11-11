# BSAT Cumulative Observation-Time Study
Execution update (2026-10-04): D01/D03/D07/D13 have now all been submitted. The design-only statements below describe the original plan; see EXPERIMENT_LOG.md for current execution records.

Design date: 2026-10-04 (Europe/London). Status: DESIGNED; four new training jobs have NOT been launched. Existing full-data baseline remains untouched.

## Scientific question

How much 3D reconstruction quality can be obtained after a limited elapsed observation time along the actual BSAT trajectory? The independent variable is available elapsed acquisition time, not an optimized encounter subset or CPI count.

## Frozen source and time definition

Project: /rds/homes/g/gey/Yao/programs/ISAR-3DGS
Dataset: /rds/homes/g/gey/Yao/programs/GEIST-6/datasets/bsat
Frozen source run: outputs/bsat_restart_20261004_102638/
Use the union of that run's train_cpis.txt and heldout_cpis.txt (2,378 CPIs), never a changing live directory scan, for execution manifests.

UTC t0 = 2026-09-13T22:57:26.139504Z, earliest CPI start in this snapshot.
Latest CPI end = 2026-09-30T00:50:15.601435Z; elapsed span = 16.07835 days.
A day means exactly 24 elapsed hours. Include a CPI only when its entire acquisition has ended by t0 + D*24h. Read cpi_start_utc/cpi_end_utc from _COMPLETE.json and cross-check static_los_metadata.json. Sort by acquisition time, not filesystem modification time.
UTC calendar-day prefixes are NOT equivalent: the first calendar date has only one CPI. Calendar-day results, if requested later, need separate labels and manifests.

## Five nested groups

All five groups use the original globally fixed train/held-out membership. No held-out CPI may enter initialization, training, evidence estimation or parameter selection.

| Group | Elapsed window | Available CPIs/images | Train images | Held-out images within window |
| --- | --- | ---: | ---: | ---: |
| D01 | First 24 h | 73 | 59 | 14 |
| D03 | First 72 h | 248 | 199 | 49 |
| D07 | First 168 h | 695 | 556 | 139 |
| D13 | First 312 h | 1790 | 1432 | 358 |
| ALL | Entire frozen snapshot, 16.07835 days | 2378 | 1903 | 475 |

The early windows are nested within later windows. Do not choose high-quality LOS or optimize coverage by changing the acquisition subset. One CPI contributes one image; CPI count is not physical encounter count.

## Controlled training

- Independently initialize D01/D03/D07/D13 from only their own training images. Do not warm-start from a larger-window model; that leaks later observations.
- Reuse the existing ALL baseline if it completes with the exact frozen split/config; do not duplicate it unnecessarily.
- Match the ALL run's effective configuration and code revision: seed=42, image_size=256, 10,000 initial Gaussians, backprojection_surface initialization, known CAD radar axes, same preprocessing/calibration, same regularization and adaptive policies, lambda_surface_gap=0.
- Each run: 12,000 iterations, four views per batch, four stages with fractions 0.15/0.5/0.8, checkpoints/previews every 1,000 iterations.
- Preserve ALL effective run_config.json as the reference. Validate count-dependent initialization behavior and small-data constraints before launching D01.
- This main study holds optimization budget fixed (48,000 view draws/run), NOT per-view exposure. Expected mean draws per training view are approximately 814/241/86/34/25 for D01/D03/D07/D13/ALL. Report this explicitly.
- Equal per-view exposure is a distinct compute-scaling study and must not be mixed into the main trend.
- Record actual loading, initialization, optimization and total elapsed time separately where measurable. Parallel wall-clock duration is resource-contended and not a clean algorithm-speed comparison.

## Evaluation protocol

1. Primary common test: the same 475 globally held-out CPIs for every model, unchanged input processing, metric implementation and display scale. This provides comparable PSNR/SSIM/MAE curves.
2. For D01 etc., this common pool includes observations after their training window: label it mixed unseen-CPI interpolation plus later-time generalization, NOT purely within-window interpolation or guaranteed unseen LOS directions.
3. Secondary time-resolved report: per-day held-out metrics and in-window vs later-window subsets. Do not compare variable within-window averages as if they were the same test set.
4. No hyperparameter tuning on the shared test set. This is one predeclared method across all durations.
5. Geometry: locate and verify the BSAT GT mesh, metre units and CAD coordinate frame first. Use one fixed evaluation protocol, GT sample set/seed and q>=0.3 threshold for Chamfer L1, HD95, normal consistency and voxel IoU. GT path/alignment are currently unverified; report geometry metrics as pending until resolved.
6. Also report support volume, overlap, largest component, Gaussian count, and matched point-cloud/render previews. Use identical LOS, camera, axes limits, q threshold and colour scale across durations.
7. Fixed unseen-CPI holdout contains neighbouring aspects; it does not prove generalization to a wholly new trajectory. A blocked day/aspect holdout would be an additional experiment, not silently substituted here.

## Intended artifacts and execution safeguards

Planned new study root: outputs/bsat_observation_days_20261004/ (ensure unused before creation).
Each new group D01/D03/D07/D13 should have train_cpis.txt, launch.json, train.log, training/run_config.json, training/selected_views.json, training/checkpoints/, training/render_preview/, and evaluation/.
Store one shared evaluation-view manifest plus a table of CPI UTC, duration membership, day and train/test split in the study root. ALL should link to the existing run rather than copy model files.

Before launching: validate strict nesting and train/test disjointness, confirm every selected CPI ends before its cutoff, compare manifest counts against the table, test D01 initialization, and check allocation time/memory. Do not stop or modify the healthy ALL run. Use only safe concurrency after measuring actual memory; 40GB capacity alone does not establish an efficient parallel count.

No new training/evaluation jobs were submitted when writing this design. The existing ALL run is already running. No automatic follow-up schedule was created.

## Presentation

Main x-axis: elapsed acquisition days, with ALL at 16.07835 (not an arbitrary fifth category).
Plot held-out PSNR/SSIM and geometry errors/IoU against time, annotated with actual training-image counts. Show matched D01/D03/D07/D13/ALL point clouds and the same representative held-out render frames.
The conclusion must follow observed results: estimate when useful structure emerges and when improvement saturates. Do not presuppose monotonic improvement; later views may expose inconsistencies or redistribute support.
