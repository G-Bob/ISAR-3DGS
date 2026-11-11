import os,sys,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
mode=os.environ['ROBUST_MODE']
if mode=='sh3_echo':
 from hybrid_worker import install
 install('sh3')
os.environ.update(R6_MASK='0' if mode=='mlp_full' else '1',R6_TRACK='0',R6_GUARD='0')
runpy.run_path(str(Path(__file__).resolve().parent/'round6_fast_worker.py'),run_name='__main__')
