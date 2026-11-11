"""perturb_wrap.py TARGET.py [args]: apply GEN_PERTURB to radar_gs.data.load_dataset (same code as the SC-GS robustness worker), then run TARGET as __main__.
GEN_PERTURB (optional, training views only): 'rician:SNRdB' adds Rician noise with sigma = peak * 10^(-SNR/20);
'speckle:L' multiplies amplitudes by sqrt(Gamma(L, 1/L)) intensity speckle (L looks) and renormalises to unit peak; 'speckleclip:L' applies the same speckle but keeps the image level and clips to [0, 1] like 'rician'; 'los:SIGMAdeg' rotates (kr, kd, normal) of each view by a fixed random rotation with angle ~ N(0, SIGMA) about a random axis."""
import math, os, runpy, sys
from pathlib import Path
import torch
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
            elif kind == 'speckle':
                import numpy as np
                rs = np.random.RandomState(90000 + i)
                f = torch.from_numpy(np.sqrt(rs.gamma(val, 1.0 / val, size=tuple(v.image.shape))).astype('float32')).to(v.image)
                img = v.image * f; v.image = img / img.max().clamp_min(1e-12); stats.append(float(f.std()))
            elif kind == 'speckleclip':
                import numpy as np
                rs = np.random.RandomState(90000 + i)
                f = torch.from_numpy(np.sqrt(rs.gamma(val, 1.0 / val, size=tuple(v.image.shape))).astype('float32')).to(v.image)
                v.image = (v.image * f).clamp(0, 1); stats.append(float(f.std()))
            elif kind == 'los':
                ang = float(torch.randn(1, generator=g)) * math.radians(val); R = _rot(torch.randn(3, generator=g), ang)
                v.kr = (R.to(v.kr) @ v.kr); v.kd = (R.to(v.kd) @ v.kd); v.normal = (R.to(v.normal) @ v.normal); stats.append(abs(math.degrees(ang)))
            else:
                raise ValueError(SPEC)
        print('GEN_PERTURB', SPEC, 'views', len(views), 'mean_stat', sum(stats) / max(len(stats), 1), flush=True)
        return views, sel
    rd.load_dataset = load_dataset
T = Path(sys.argv[1]).resolve(); sys.argv = [str(T)] + sys.argv[2:]; sys.path.insert(0, str(T.parent))
runpy.run_path(str(T), run_name='__main__')
