import json
from pathlib import Path
R = Path('outputs/baseline_robust_20261026'); out = json.loads((R / 'collected.json').read_text())
for n, e in out.items():
    f = next(iter(sorted((R / n).rglob('summary.json'))), None)
    if f is None: print(n, 'no summary'); continue
    img = json.loads(f.read_text()).get('image', {})
    e['render'] = {k: v for k, v in img.items() if isinstance(v, (int, float))} if isinstance(img, dict) else {}
    e['render_file'] = str(f)
    print(n, {k: round(v, 3) for k, v in e['render'].items() if 'psnr' in k.lower() or 'ssim' in k.lower()})
(R / 'collected.json').write_text(json.dumps(out, indent=1))