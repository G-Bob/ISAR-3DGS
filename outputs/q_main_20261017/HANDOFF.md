# q_main_20261017 — main-method re-run with lambda_support_binary=0.3, tau=0.5

34 runs (main_suite.py RUNS): loss factorial F1-F9, representation R1-R2, seeds S1-S4, topology T1-T3, daily prefixes D01-D16.
Proposed seeds 42-44 already exist in outputs/q_support_20261016/ent300_seed4{2,3,4}.

Queue: shared on-disk queue; a run is claimed by creating its directory (claim.json records the Slurm job).
- Jupyter job 57798395 (ends ~14:44 BST 17 Oct): 3 workers, log queue_jupyter.log
- Batch jobs 57802990-92 (qmain.sbatch, bbgpu, A100, 10 h): logs slurm_<job>.log
- No new claim if the job has < 110 min left; runs abandoned by an ended job are renamed *.staleNNN and re-claimed.
- When every run has finished, the last worker runs collect.py -> collected.json, summary.csv, figs/ and writes ALL_DONE.

Check progress:
  squeue -u $USER
  for d in outputs/q_main_20261017/[FSRTD]*_*; do echo $(basename $d) $(grep '^iter' $d/train.log | tail -n1 | cut -c1-12) $(cat $d/exit.json 2>/dev/null); done
Manual collection at any time: python outputs/q_main_20261017/collect.py
If all batch jobs end before the queue is empty: sbatch outputs/q_main_20261017/qmain.sbatch
