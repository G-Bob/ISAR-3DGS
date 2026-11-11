"""Collect finished q_main runs: q_analysis metrics (q>=0.5 main, q>=0.3, Otsu, all) + PSNR/SSIM. Skips runs already collected."""
import json, subprocess, sys
from pathlib import Path
M = Path('outputs/q_robust_20261018'); Q = Path('outputs/q_support_20261016')
out = M / 'collected.json'; res = json.loads(out.read_text()) if out.exists() else {}
extra = {}
dirs = {p.name: p for p in sorted(M.iterdir()) if p.is_dir() and p.name[0] in 'NLD' and p.name != 'smoke'}
dirs.update(extra)
todo = [n for n, d in dirs.items() if n not in res and (d / 'evaluation_exit.json').exists() and (d / 'training/checkpoints/final.pt').exists()]
for n in todo:
    d = dirs[n]; tmp = M / f'_qa_{n}.json'; lab = ('mlp' if '_mlp_' in n else 'sh3') + '=' + str(d / 'training/checkpoints/final.pt')
    subprocess.check_call([sys.executable, str(Q / 'q_analysis.py'), str(tmp), lab], stdout=subprocess.DEVNULL)
    a = list(json.loads(tmp.read_text()).values())[0]; tmp.unlink()
    s = json.loads((d / 'evaluation_common475/summary.json').read_text())
    pick = lambda m: {k: m[k] for k in ('N', 'cd_l1', 'cd_l2', 'hd95', 'acc', 'comp', 'cov01')}
    res[n] = dict(q050=pick(a['sweep']['0.50']), q030=pick(a['sweep']['0.30']), all=pick(a['all']), otsu=dict(threshold=a['otsu']['threshold'], **pick(a['otsu'])),
                  mid=a['q_mid_fraction'], hist=a['q_hist'], psnr=s['image']['mean_psnr_db'], ssim=s['image']['mean_ssim'],
                  sweep_l1={t: v['cd_l1'] for t, v in a['sweep'].items()}, auroc01=a['auroc_d0.1']['q'])
    print('collected', n, round(res[n]['q050']['cd_l1'], 5), flush=True)
out.write_text(json.dumps(res, indent=1)); print('total', len(res), 'todo_done', len(todo))
rows = ['run,N50,cdl1_50,cdl2_50,hd95_50,cov_50,otsu,cdl1_otsu,cdl1_30,all_cdl1,all_cdl2,all_hd95,mid,psnr,ssim']
for n, r in sorted(res.items()):
    a, b = r['q050'], r['all']
    rows.append(','.join([n, str(a['N'])] + ['%.6f' % v for v in (a['cd_l1'], a['cd_l2'], a['hd95'], a['cov01'], r['otsu']['threshold'], r['otsu']['cd_l1'], r['q030']['cd_l1'], b['cd_l1'], b['cd_l2'], b['hd95'], r['mid'], r['psnr'], r['ssim'])]))
(M / 'summary.csv').write_text(chr(10).join(rows) + chr(10))
