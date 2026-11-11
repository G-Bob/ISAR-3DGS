"""Generic worker: frozen round6 pipeline + optional hybrid installs. Env: GEN_SRC, GEN_HYBRID (comma list), R6_MASK/R6_TRACK/R6_GUARD,
GEN_PERTURB (optional, training views only): 'rician:SNRdB' adds Rician noise with sigma = peak * 10^(-SNR/20);
'los:SIGMAdeg' rotates (kr, kd, normal) of each view by a fixed random rotation with angle ~ N(0, SIGMA) about a random axis."""
import math, os, runpy, sys
from pathlib import Path
import torch
SRC = Path(os.environ['GEN_SRC']).resolve()
sys.path.insert(0, str(SRC))
from hybrid_worker import install
for mode in [m for m in os.environ.get('GEN_HYBRID', '').split(',') if m]:
    install(mode)
SPEC = os.environ.get('GEN_PERTURB', '')
if SPEC:
    import radar_gs.data as rd
    _orig = rd.load_dataset
    def _rot(axis, ang):
        axis = axis / axis.norm(); K = torch.tensor([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        return torch.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
    def load_dataset(*a, **k):
        views, sel = _orig(*a, **k)
        kind, val = SPEC.split(':'); val = float(val); stats = []
        for i, v in enumerate(views):
            g = torch.Generator().manual_seed(90000 + i)
            if kind == 'rician':
                img = v.image; sig = float(img.max()) * 10 ** (-val / 20)
                n1 = torch.randn(img.shape, generator=g) * sig; n2 = torch.randn(img.shape, generator=g) * sig
                v.image = torch.sqrt((img + n1.to(img)) ** 2 + n2.to(img) ** 2).clamp(0, 1); stats.append(sig)
            elif kind == 'los':
                ang = float(torch.randn(1, generator=g)) * math.radians(val); R = _rot(torch.randn(3, generator=g), ang)
                v.kr = (R.to(v.kr) @ v.kr); v.kd = (R.to(v.kd) @ v.kd); v.normal = (R.to(v.normal) @ v.normal); stats.append(abs(math.degrees(ang)))
            else:
                raise ValueError(SPEC)
        print('GEN_PERTURB', SPEC, 'views', len(views), 'mean_stat', sum(stats) / max(len(stats), 1), flush=True)
        return views, sel
    rd.load_dataset = load_dataset
runpy.run_path(str(SRC / 'round6_fast_worker.py'), run_name='__main__')
