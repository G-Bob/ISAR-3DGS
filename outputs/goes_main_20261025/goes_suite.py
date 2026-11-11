"""GOES-R second target (2026-10-25): Table I/II method comparison + D01-D16 acquisition-time study.

Same protocol as the Skynet/BSAT runs; only the data root, the ground truth and the (rebuilt) initializers change.
- Split: Skynet 1903 train / 475 test CPI names (identical trajectory and CPI list).
- Initializers: rebuilt on GOES images with the daily suite's initialize() recipe
  (backprojection_surface 10k seed42 bbox6 + tracks from <=96 training views, radial normals).
- Proposed (P_sh3_mask) and D01-D16: SH3 + echo mask, lambda_support_binary=0.3, run through q_main's gen_worker on the
  sh3_echo_daily_recovery source snapshot (the pipeline of q_main D01-D16).
- Baselines: Efficient-3DGS (Pei, uniform init, pei_bsat.py), ISAR-GS (isargs) and RNSplat (rns_base) via adapted_papers.py.
- Evaluation: frozen common475 test views, GOES GT (goes_gt_blender_5000.npy), no ICP.
Queue: a run is claimed by creating its directory; several GPU jobs cooperate. Usage (from ISAR-3DGS root):
  python outputs/goes_main_20261025/goes_suite.py --workers 3      # prepare (once, locked) + work the queue
  python outputs/goes_main_20261025/goes_suite.py --smoke           # 2-iteration smoke on available data
"""
import argparse, csv, inspect, json, os, shutil, socket, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path('outputs/goes_main_20261025')
DATA = '/rds/homes/g/gey/Yao/programs/GEIST-6/datasets/goes'
GT = ROOT / 'goes_gt_blender_5000.npy'
PY = '/rds/homes/g/gey/miniconda3/bin/python'
DAILY = Path('outputs/sh3_echo_daily_recovery_20261011'); DSRC = DAILY / 'source_snapshot'
RSRC = Path('outputs/echo_mask_robustness_20261009/source_snapshot')
PSRC = Path('outputs/paper_reproduction/suite_source_snapshot')
PEISRC = Path('outputs/paper_reproduction/P30_Pei_uniform_wide_restart_source_snapshot')
GEN = Path('outputs/q_main_20261017/gen_worker.py')
STUDY = Path('outputs/bsat_observation_days_20261004')
TRAIN = Path('outputs/echo_mask_robustness_20261009/sh3_echo_seed43/train_cpis.txt')
HELD = Path('outputs/bsat_methods8_20261004/heldout_cpis.txt')

GS = ['P_sh3_mask'] + [f'D{d:02d}_sh3_mask' for d in range(1, 17)] + ['A_fullimage_sh3', 'A_novis_sh3']
ABL = {'A_fullimage_sh3': 'full-image supervision (R6_MASK=0)', 'A_novis_sh3': 'no LOS visibility (disable_visibility)'}  # appended 2026-10-25, run after the main queue
PAPER = {'B_Pei_uniform': 'pei', 'B_ISARGS': 'isargs', 'B_RNSplat_base': 'rns_base'}
RUNS = ['P_sh3_mask', 'B_Pei_uniform', 'B_ISARGS', 'B_RNSplat_base'] + GS[1:]


def write(p, v): Path(p).write_text(json.dumps(v, indent=2))


def needed_names():
    return TRAIN.read_text().split() + HELD.read_text().split()


def data_ready():
    import glob
    miss = [n for n in needed_names() if not (Path(DATA) / n / '_COMPLETE.json').exists() or not glob.glob(f'{DATA}/{n}/isar_images_noaxis/*.png')]
    return miss


# ---------------------------------------------------------------- preparation
def patched_copy(src, dst, old, new):
    t = Path(src).read_text(); assert t.count(old) == 1, (src, old); Path(dst).write_text(t.replace(old, new))


def prepare_sources():
    s = ROOT / 'src'; s.mkdir(exist_ok=True)
    # evaluate_suite (GS + adapted papers): only the GT file changes; preset/test views stay the frozen common475.
    patched_copy(RSRC / 'evaluate_suite.py', s / 'evaluate_suite_goes.py', "ref=np.load(old/'skynet_gt_blender_5000.npy')", "ref=np.load(os.environ['GOES_GT'])")
    patched_copy(PSRC / 'evaluate_suite.py', s / 'evaluate_suite_goes_paper.py', "ref=np.load(old/'skynet_gt_blender_5000.npy')", "ref=np.load(os.environ['GOES_GT'])")
    # Pei: copy the frozen snapshot, patch only GT in its evaluator.
    pei = s / 'pei'
    if not pei.exists():
        shutil.copytree(PEISRC, pei, ignore=shutil.ignore_patterns('__pycache__'))
        patched_copy(PEISRC / 'evaluate_pei.py', pei / 'evaluate_pei.py', "ref=np.load(old/'skynet_gt_blender_5000.npy')", "import os;ref=np.load(os.environ['GOES_GT'])")
    q = Path('outputs/q_support_20261016/q_analysis.py').read_text()
    for a, b in [("DATA=Path('/rds/homes/g/gey/Yao/programs/GEIST-6/datasets/bsat')", f"DATA=Path('{DATA}')"),
                 ("GT=np.load('outputs/bsat_observation_days_20261004/skynet_gt_blender_5000.npy')", f"GT=np.load('{GT}')")]:
        assert q.count(a) == 1; q = q.replace(a, b)
    (s / 'q_analysis_goes.py').write_text(q)


def initialize(run, names):
    """Exact copy of sh3_echo_daily_recovery_20261011/suite.py initialize(), with GOES data."""
    sys.path.insert(0, str(DSRC.resolve()))
    import numpy as np, torch
    from radar_gs.data import load_dataset
    from radar_gs.initialization import backprojection_surface
    from multicpi_tracking import track_keypoints_multi_cpi, self_check
    torch.set_num_threads(4); self_check(); start = time.perf_counter()
    cfg = json.loads((run / 'config.json').read_text())
    kwargs = {k: cfg[k] for k in inspect.signature(load_dataset).parameters if k in cfg}
    kwargs.update(data_root=DATA, encounter_names=names, encounter_count=None, frames_per_encounter=1, images_per_encounter=1)
    views, _ = load_dataset(**kwargs)
    assert len(views) == len(names) and {v.encounter for v in views} == set(names)
    points, _ = backprojection_surface(10000, cfg['bbox'], 42, views); points = points.numpy().copy()
    selected = np.linspace(0, len(views) - 1, min(96, len(views))).round().astype(int); sampled = [views[i] for i in selected]
    geo = [dict(kr=v.kr.numpy(), kd=v.kd.numpy(), dr=float(v.dr), dd=float(v.dd)) for v in sampled]
    detections, tracks = track_keypoints_multi_cpi([v.image.numpy() for v in sampled], geo, bbox=cfg['bbox'])
    used = set()
    for t in tracks[:2000]:
        point = np.asarray(t['point']); order = np.argsort(np.linalg.norm(points - point, axis=1))
        idx = next(int(i) for i in order if int(i) not in used); used.add(idx); points[idx] = point
    normals = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-8)
    values = np.c_[points, normals].astype(np.float32)
    assert values.shape == (10000, 6) and np.isfinite(values).all() and np.unique(values[:, :3], axis=0).shape[0] == 10000
    np.savez(run / 'initializer.npz', points=values)
    write(run / 'tracks.json', dict(selected_cpis=[v.encounter for v in sampled], tracks=tracks))
    write(run / 'initialization_verified.json', dict(seconds=time.perf_counter() - start, training_views=len(views), selected_track_views=len(sampled), tracks=len(tracks), inserted=len(used), data_root=DATA))
    print('INIT', run.name, len(names), 'tracks', len(tracks), flush=True)


def gs_config(run, names_file, init_path):
    cfg = json.loads((DAILY / 'D16' / 'config.json').read_text())
    cfg.update(data_root=DATA, config=str(run / 'config.json'), out_dir=str(run / 'training'), encounter_names_file=str(Path(names_file).resolve()),
               init_path=str(Path(init_path).resolve()), initialization='pointcloud_surface', seed=42, lambda_support_binary=0.3)
    return cfg


def prepare():
    ROOT.mkdir(exist_ok=True); prepare_sources()
    I = ROOT / 'init'; I.mkdir(exist_ok=True)
    # Initializers: ALL (1903) and each day prefix (Skynet prefix lists, identical CPI names and end times).
    jobs = [('ALL', TRAIN)] + [(f'D{d:02d}', DAILY / f'D{d:02d}' / 'train_cpis.txt') for d in range(1, 17)]
    for name, lst in jobs:
        r = I / name; r.mkdir(exist_ok=True)
        if (r / 'initializer.npz').exists(): continue
        names = Path(lst).read_text().split(); (r / 'train_cpis.txt').write_text('\n'.join(names) + '\n')
        write(r / 'config.json', gs_config(r, r / 'train_cpis.txt', r / 'initializer.npz')); initialize(r, names)
    # Run plans (configs are written when a run is claimed).
    write(ROOT / 'PLAN.json', dict(target='GOES-R-0.4.glb', data_root=DATA, gt=str(GT), split='Skynet 1903/475 names', runs=RUNS,
          proposed='SH3+echo mask, lambda_support_binary 0.3, q>=0.5, gen_worker+sh3_echo_daily_recovery snapshot',
          baselines=PAPER, budget='seed42 12k batch8 init10k cap20k', evaluation='common475 frozen views, GOES GT, no ICP'))
    (ROOT / 'PREPARED').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))


# ---------------------------------------------------------------- execution
def env_gs():
    return dict(os.environ, PYTHONPATH=str(DSRC.resolve()) + ':.:outputs', GEN_SRC=str(DSRC), GEN_HYBRID='sh3', R6_MASK='1', R6_TRACK='0', R6_GUARD='0',
                ROBUST_MODE='sh3_echo', GOES_GT=str(GT.resolve()))


def run_logged(cmd, env, log):
    with open(log, 'w') as f: return subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)


def execute(name, r, iters=12000, names_file=None):
    started = dict(host=socket.gethostname(), job=os.environ.get('SLURM_JOB_ID'), time=time.time(), name=name)
    if name in GS:
        tag = 'ALL' if (name == 'P_sh3_mask' or name in ABL) else name[:3]
        names_file = names_file or ROOT / 'init' / tag / 'train_cpis.txt'
        write(r / 'config.json', gs_config(r, names_file, ROOT / 'init' / tag / 'initializer.npz'))
        if name == 'A_novis_sh3':
            c = json.loads((r / 'config.json').read_text()); c['disable_visibility'] = True; write(r / 'config.json', c)
        write(r / 'PLAN.json', dict(name=name, init=tag, lambda_support_binary=0.3, target='GOES', ablation=ABL.get(name, '')))
        env = env_gs()
        if name == 'A_fullimage_sh3': env['R6_MASK'] = '0'
        cmd = [PY, '-u', str(GEN), '--config', str(r / 'config.json'), '--data_root', DATA, '--out_dir', str(r / 'training'), '--encounter_names_file', str(names_file),
               '--frames_per_encounter', '1', '--images_per_encounter', '1', '--iters', str(iters), '--views_per_batch', '8', '--checkpoint_interval', '500',
               '--preview_interval', '500', '--seed', '42', '--device', 'cuda']
        write(r / 'started.json', dict(started, command=cmd)); code = run_logged(cmd, env, r / 'train.log'); write(r / 'exit.json', dict(code=code))
        if code or iters != 12000: return code
        env['PYTHONPATH'] = str(RSRC.resolve()) + ':.:outputs'
        ec = run_logged([PY, '-u', str(ROOT / 'src' / 'evaluate_suite_goes.py'), str(r), '--hybrid', 'sh3'], env, r / 'evaluation.log')
        write(r / 'evaluation_exit.json', dict(code=ec)); return ec
    mode = PAPER[name]
    write(r / 'PLAN.json', dict(name=name, mode=mode, target='GOES', note='disclosed BSAT-protocol adaptation, same settings as Skynet run'))
    env = dict(os.environ, GOES_GT=str(GT.resolve()), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    if mode == 'pei':
        cfg = json.loads((PEISRC / 'm19_config.json').read_text()); cfg.update(data_root=DATA)
        if names_file: cfg['encounter_names'] = Path(names_file).read_text().split()
        write(r / 'config.json', cfg); env['PYTHONPATH'] = '.'
        out = r / 'run'
        cmd = [PY, '-u', str(ROOT / 'src' / 'pei' / 'pei_bsat.py'), '--config', str(r / 'config.json'), '--out', str(out), '--init', 'uniform', '--iters', str(iters)]
        write(r / 'started.json', dict(started, command=cmd)); code = run_logged(cmd, env, r / 'train.log'); write(r / 'exit.json', dict(code=code))
        if code or iters != 12000: return code
        ec = run_logged([PY, '-u', str(ROOT / 'src' / 'pei' / 'evaluate_pei.py'), str(out)], env, r / 'evaluation.log')
        write(r / 'evaluation_exit.json', dict(code=ec)); return ec
    src = {'isargs': 'P32_ISARGS_complex_continuum', 'rns_base': 'P33_RNS_alpha_base'}[mode]
    cfg = json.loads((Path('outputs/paper_reproduction') / src / 'config.json').read_text())
    tf = r / 'train_cpis.txt'; shutil.copy2(names_file or TRAIN, tf)
    cfg.update(data_root=DATA, out_dir=str(r), config=str(r / 'config.json'), encounter_names_file=str(tf.resolve()), init_path=str((ROOT / 'init' / 'ALL' / 'initializer.npz').resolve()))
    if 'encounter_names' in cfg: cfg['encounter_names'] = tf.read_text().split()
    write(r / 'config.json', cfg)
    env.update(PYTHONPATH='.:outputs:' + str(PSRC.resolve()), PAPER_MODE=mode, R6_MASK='1', R6_TRACK='1', R6_GUARD='1')
    cmd = [PY, '-u', str(PSRC / 'adapted_papers.py'), '--config', str(r / 'config.json'), '--out', str(r / 'training_stage'), '--init', 'train_points', '--iters', str(iters)]
    write(r / 'started.json', dict(started, command=cmd)); code = run_logged(cmd, env, r / 'train.log'); write(r / 'exit.json', dict(code=code))
    if code or iters != 12000: return code
    for f in (r / 'training_stage').iterdir():
        d = r / f.name
        if not d.exists(): d.symlink_to(f.resolve(), target_is_directory=f.is_dir())
    env['PYTHONPATH'] = str(PSRC.resolve()) + ':.:outputs'
    ec = run_logged([PY, '-u', str(ROOT / 'src' / 'evaluate_suite_goes_paper.py'), str(r), '--paper', mode], env, r / 'evaluation.log')
    write(r / 'evaluation_exit.json', dict(code=ec)); return ec


# ---------------------------------------------------------------- queue (same rules as q_main main_suite)
MIN_LEFT = 110 * 60


def _secs(raw):
    raw = raw.strip()
    if not raw or raw in ('INVALID', 'UNLIMITED'): return 10 ** 9
    days = 0
    if '-' in raw: d, raw = raw.split('-'); days = int(d)
    x = [int(v) for v in raw.split(':')]
    while len(x) < 3: x.insert(0, 0)
    return days * 86400 + x[0] * 3600 + x[1] * 60 + x[2]


def remaining():
    j = os.environ.get('SLURM_JOB_ID')
    if not j: return 10 ** 9
    try: return _secs(subprocess.check_output(['squeue', '-j', j, '-h', '-o', '%L'], text=True))
    except Exception: return 10 ** 9


def alive(job):
    if not job: return False
    try: return bool(subprocess.check_output(['squeue', '-j', str(job), '-h', '-o', '%T'], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception: return False


def stale(r):
    if (r / 'exit.json').exists(): return False
    c = r / 'claim.json'
    if not c.exists(): return time.time() - r.stat().st_mtime > 600
    return not alive(json.loads(c.read_text()).get('job'))


LOCK = threading.Lock()


def claim():
    if remaining() < MIN_LEFT: print(time.strftime('%H:%M:%S'), 'deadline: no new claims', flush=True); return None, None
    with LOCK:
        for name in RUNS:
            r = ROOT / name
            if r.exists():
                if not stale(r): continue
                try: os.rename(r, r.with_name(r.name + '.stale%d' % int(time.time())))
                except OSError: continue
            try: r.mkdir(exist_ok=False)
            except FileExistsError: continue
            write(r / 'claim.json', dict(job=os.environ.get('SLURM_JOB_ID'), host=socket.gethostname(), time=time.time()))
            return name, r
    return None, None


def loop(i):
    time.sleep(30 * i)
    while True:
        name, r = claim()
        if name is None: return
        print(time.strftime('%H:%M:%S'), 'worker', i, 'start', name, flush=True)
        try: execute(name, r)
        except Exception as e: (r / 'crash.txt').write_text(repr(e))
        print(time.strftime('%H:%M:%S'), 'worker', i, 'done', name, flush=True)


def wait_prepared():
    try: os.mkdir(ROOT / 'prepare.lock'); mine = True
    except FileExistsError: mine = False
    if mine:
        try: prepare()
        except Exception as e: (ROOT / 'prepare_crash.txt').write_text(repr(e)); raise
    while not (ROOT / 'PREPARED').exists():
        if (ROOT / 'prepare_crash.txt').exists(): raise SystemExit('prepare crashed')
        time.sleep(30)


def smoke():
    """2-iteration finite smoke for every pipeline on available CPIs (no evaluation)."""
    S = ROOT / 'smoke'; S.mkdir(exist_ok=True); prepare_sources()
    missing = set(data_ready()); avail = [n for n in TRAIN.read_text().split() if n not in missing]
    pick = [avail[i] for i in range(0, len(avail), max(1, len(avail) // 16))][:16]
    lst = S / 'smoke16.txt'; lst.write_text('\n'.join(pick) + '\n')
    r = S / 'init_smoke'; r.mkdir(exist_ok=True)
    write(r / 'config.json', gs_config(r, lst, r / 'initializer.npz'))
    if not (r / 'initializer.npz').exists(): initialize(r, pick)
    for d in ['ALL', 'D01']:
        p = ROOT / 'init' / d; p.mkdir(parents=True, exist_ok=True)
        if not (p / 'initializer.npz').exists(): shutil.copy2(r / 'initializer.npz', p / 'initializer.npz'); (p / 'SMOKE_ONLY').write_text('replace in prepare')
        (p / 'train_cpis.txt').exists() or (p / 'train_cpis.txt').write_text(lst.read_text())
    res = {}
    for name in ['P_sh3_mask', 'B_Pei_uniform', 'B_ISARGS', 'B_RNSplat_base']:
        d = S / name
        if d.exists(): d.rename(d.with_name(d.name + '.old%d' % int(time.time())))  # never delete
        d.mkdir(); (d / 'SMOKE').write_text('1')
        res[name] = execute(name, d, iters=2, names_file=lst); print('SMOKE', name, res[name], flush=True)
    for d in ['ALL', 'D01']:
        p = ROOT / 'init' / d
        if (p / 'SMOKE_ONLY').exists(): (p / 'initializer.npz').rename(p / 'initializer_smoke.npz')
    write(S / 'smoke_result.json', res); print('SMOKE_DONE', res, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=3); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    if a.check: m = data_ready(); print('MISSING', len(m), m[:10]); sys.exit(1 if m else 0)
    if a.smoke: smoke(); sys.exit(0)
    while True:  # wait on the GPU for the last CPIs (1143-1200), then start immediately
        m = data_ready()
        if not m: break
        print(time.strftime('%H:%M:%S'), 'waiting for data,', len(m), 'missing, e.g.', m[:3], flush=True)
        if remaining() < MIN_LEFT: raise SystemExit('job too short after waiting for data')
        time.sleep(300)
    wait_prepared()
    ts = [threading.Thread(target=loop, args=(i,)) for i in range(a.workers)]; [t.start() for t in ts]; [t.join() for t in ts]
    print('QUEUE_EMPTY', flush=True)
    if all((ROOT / n / 'evaluation_exit.json').exists() for n in RUNS):
        try:
            os.mkdir(ROOT / 'collect.lock')
            subprocess.call([PY, '-u', str(ROOT / 'collect_goes.py')], stdout=open(ROOT / 'collect.log', 'w'), stderr=subprocess.STDOUT)
            (ROOT / 'ALL_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
        except FileExistsError: pass
