import json,os,subprocess,sys
from pathlib import Path
SRC=Path('outputs/echo_mask_robustness_20261009/source_snapshot')
env=dict(os.environ,PYTHONPATH=str(SRC.resolve())+':.:outputs',ROBUST_MODE='sh3_echo')
for r in map(Path,sys.argv[1:]):
    with (r/'evaluation.log').open('w') as f: ec=subprocess.call([sys.executable,'-u',str(SRC/'evaluate_suite.py'),str(r),'--hybrid','sh3'],env=env,stdout=f,stderr=subprocess.STDOUT)
    (r/'evaluation_exit.json').write_text(json.dumps(dict(code=ec)))
