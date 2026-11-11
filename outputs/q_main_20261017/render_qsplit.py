import json, math, sys, inspect
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import maximum_filter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hybrid_worker import install
install('sh3')
from radar_gs.io import load_scene
from radar_gs.data import load_dataset
from radar_gs.presets import load_encounter_batch_preset
from radar_gs.losses import compress_amplitude
import evaluate_ablation as ev
torch.set_num_threads(2); dev = torch.device('cuda'); torch.cuda.set_per_process_memory_fraction(0.2)
training = Path('outputs/q_support_20261016/ent300_seed42/training')
scene, ck = load_scene(training / 'checkpoints/final.pt', dev); cfg = ck['arguments']
old = Path('outputs/bsat_observation_days_20261004')
preset = load_encounter_batch_preset(old / 'evaluation_preset.json', allow_incomplete=True)
views, _ = load_dataset(cfg['data_root'], image_size=256, native_dr=cfg['native_dr'], native_dd=cfg['native_dd'], encounter_names=preset['encounter_names'], frames_per_encounter=1, background_threshold=cfg['background_threshold'], normalization_percentile=cfg['normalization_percentile'])
print('views', len(views))
R = ev.RadarVisibilityRenderer
print('renderer sig', str(inspect.signature(R.__init__))[:200])
renderer = R(256, 256, cfg['depth_bins'], cfg['kernel_radius'], cfg['minimum_pixel_variance'], cfg['opacity_gain'], cfg.get('visibility_pixel_size', 0.0), cfg.get('support_composition', 'noisy_or')).to(dev)
T = cfg.get('final_support_temperature', 0.35); gamma = float(cfg.get('render_gamma', 1.0))
with torch.no_grad():
    q = scene.support_probability(T)
qs = {'full': q, 'struct': torch.where(q >= 0.5, q, torch.zeros_like(q)), 'off': torch.where(q < 0.5, q, torch.zeros_like(q))}
print('N struct', int((q >= 0.5).sum()), 'N off', int((q < 0.5).sum()))
def echo_mask(img):
    a = img / max(img.max(), 1e-6); med = np.median(a); mad = np.median(np.abs(a - med))
    return maximum_filter(a > max(0.02, med + 4 * mad), size=5)
stats = {k: dict(psnr=[], energy=[], inside=[]) for k in qs}
keep = {}
pick = set(np.linspace(0, len(views) - 1, 3).round().astype(int).tolist())
with torch.no_grad():
    for i, v in enumerate(views):
        tgt = v.image.cpu().numpy(); m = echo_mask(tgt)
        preds = {}
        for k, s in qs.items():
            out = renderer(scene, v.kr, v.kd, v.dr, v.dd, support_probability=s, support_temperature=T)
            pred = compress_amplitude(out.image, gamma).cpu().numpy(); preds[k] = pred
            mse = float(((pred - tgt) ** 2).mean()); stats[k]['psnr'].append(-10 * math.log10(max(mse, 1e-12)))
            stats[k]['energy'].append(float(out.image.sum())); stats[k]['inside'].append(float(pred[m].sum() / max(pred.sum(), 1e-9)))
        if i in pick: keep[i] = (tgt, preds, v.encounter)
summ = {}
for k in qs:
    summ[k] = dict(psnr=float(np.mean(stats[k]['psnr'])), energy_share=float(np.mean(np.array(stats[k]['energy']) / np.maximum(np.array(stats['full']['energy']), 1e-9))), inside_mask=float(np.mean(stats[k]['inside'])))
print(json.dumps(summ, indent=1))
json.dump(summ, open('outputs/q_main_20261017/render_qsplit.json', 'w'), indent=1)
fig, axs = plt.subplots(3, 4, figsize=(7.16, 5.5)); L = 'abcdefghijkl'; k = 0
for r, i in enumerate(sorted(keep)):
    tgt, preds, name = keep[i]
    ys, xs = np.where(tgt > 0.05); cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    h = min(128, int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.65) + 8)
    sl = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h))
    for c, img in enumerate([tgt, preds['full'], preds['struct'], preds['off']]):
        ax = axs[r, c]; ax.imshow(np.sqrt(np.clip(img[sl], 0, 1)), cmap='magma', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title('(%s)' % L[k], fontsize=8, loc='left'); k += 1
fig.tight_layout(h_pad=0.6, w_pad=0.4)
fig.savefig('outputs/q_main_20261017/figs_labelled/render_qsplit.png', dpi=300)
print('picked', [keep[i][2] for i in sorted(keep)])
