"""Baseline robustness suite (2026-10-26): Efficient-3DGS (P30), ISAR-GS (P32) and RNSplat (P33) retrained on Skynet-5D with the
same training-image perturbations as the SC-GS robustness study (GEN_PERTURB: rician / speckleclip / los), via perturb_wrap.py.
Original Table I commands and configs; evaluation on the clean test views. Claim a run by mkdir of its directory."""
import json, os, shutil, socket, subprocess, sys, threading, time
from pathlib import Path
ROOT = Path('outputs/baseline_robust_20261026'); PY = '/rds/homes/g/gey/miniconda3/bin/python'
PR = Path('outputs/paper_reproduction'); PSRC = PR / 'suite_source_snapshot'; PEI = PR / 'P30_Pei_uniform_wide_restart_source_snapshot'
CONDS = [('snr20', 'rician:20'), ('spk1', 'speckleclip:1'), ('los1p0', 'los:1.0'), ('snr30', 'rician:30'), ('spk3', 'speckleclip:3'),
         ('los0p25', 'los:0.25'), ('snr10', 'rician:10'), ('spk10', 'speckleclip:10'), ('los0p5', 'los:0.5'), ('los2p0', 'los:2.0')]
METHODS = {'pei': None, 'isargs': 'P32_ISARGS_complex_continuum', 'rns_base': 'P33_RNS_alpha_base'}
RUNS = [(f'{m}_{c}', m, spec) for c, spec in CONDS for m in METHODS]
MIN_LEFT = 150 * 60  # ponytail: fixed budget guess per run; raise if baselines turn out slower


def write(p, d): Path(p).write_text(json.dumps(d, indent=2, default=str))


def logged(cmd, env, log):
    with open(log, 'w') as f: return subprocess.call([str(c) for c in cmd], env=env, stdout=f, stderr=subprocess.STDOUT)


def remaining():
    jid = os.environ.get('SLURM_JOB_ID')
    if not jid: return 10 ** 9
    raw = subprocess.run(['squeue', '-h', '-j', jid, '-o', '%L'], capture_output=True, text=True).stdout.strip()
    if not raw or raw in ('INVALID', 'UNLIMITED'): return 10 ** 9
    days = 0
    if '-' in raw: d, raw = raw.split('-'); days = int(d)
    x = [int(v) for v in raw.split(':')]
    while len(x) < 3: x.insert(0, 0)
    return days * 86400 + x[0] * 3600 + x[1] * 60 + x[2]


def execute(name, m, spec, r):
    write(r / 'PLAN.json', dict(name=name, method=m, perturb=spec, host=socket.gethostname(), job=os.environ.get('SLURM_JOB_ID'), time=time.ctime()))
    env = dict(os.environ, GEN_PERTURB=spec, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    wrap = ROOT / 'perturb_wrap.py'
    if m == 'pei':
        env['PYTHONPATH'] = '.'
        cmd = [PY, '-u', wrap, PEI / 'pei_bsat.py', '--config', PEI / 'm19_config.json', '--out', r / 'run', '--init', 'uniform', '--iters', '12000']
        code = logged(cmd, env, r / 'train.log'); write(r / 'exit.json', dict(code=code))
        if code: return code
        env.pop('GEN_PERTURB')
        ec = logged([PY, '-u', PEI / 'evaluate_pei.py', r / 'run'], env, r / 'evaluation.log')
    else:
        cfg = json.loads((PR / METHODS[m] / 'config.json').read_text())
        cfg.update(out_dir=str(r), config=str(r / 'config.json')); write(r / 'config.json', cfg)
        env.update(PYTHONPATH='.:outputs:' + str(PSRC.resolve()), PAPER_MODE=m, R6_MASK='1', R6_TRACK='1', R6_GUARD='1')
        cmd = [PY, '-u', wrap, PSRC / 'adapted_papers.py', '--config', r / 'config.json', '--out', r / 'training_stage', '--init', 'train_points', '--iters', '12000']
        code = logged(cmd, env, r / 'train.log'); write(r / 'exit.json', dict(code=code))
        if code: return code
        for f in (r / 'training_stage').iterdir():
            d = r / f.name
            if not d.exists(): d.symlink_to(f.resolve(), target_is_directory=f.is_dir())
        env.pop('GEN_PERTURB'); env['PYTHONPATH'] = str(PSRC.resolve()) + ':.:outputs'
        ec = logged([PY, '-u', PSRC / 'evaluate_suite.py', r, '--paper', m], env, r / 'evaluation.log')
    write(r / 'evaluation_exit.json', dict(code=ec)); return ec


def loop(i):
    time.sleep(30 * i)
    for name, m, spec in RUNS:
        if remaining() < MIN_LEFT: print(time.strftime('%H:%M:%S'), 'worker', i, 'stop: not enough time left', flush=True); return
        r = ROOT / name
        try: r.mkdir()
        except FileExistsError: continue
        print(time.strftime('%H:%M:%S'), 'worker', i, 'start', name, flush=True)
        code = execute(name, m, spec, r)
        print(time.strftime('%H:%M:%S'), 'worker', i, 'done', name, code, flush=True)


if __name__ == '__main__':
    ts = [threading.Thread(target=loop, args=(i,)) for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3)]
    [t.start() for t in ts]; [t.join() for t in ts]; print('QUEUE_EMPTY', flush=True)
