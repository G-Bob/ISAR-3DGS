import os, runpy, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from hybrid_worker import install
install('sh3')
os.environ.update(R6_MASK='1',R6_TRACK='0',R6_GUARD='0')
runpy.run_path(str(HERE/'round6_fast_worker.py'),run_name='__main__')
