import json
from pathlib import Path
from run_replay import ROOT, chain

jobs=json.loads((ROOT/'proportional/jobs.json').read_text())
ok=chain('proportional',jobs)
(ROOT/('PROPORTIONAL_COMPLETE' if ok else 'PROPORTIONAL_FAILED')).touch()
