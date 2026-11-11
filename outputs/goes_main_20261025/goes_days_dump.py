# GOES-R day-prefix models: render one test CPI and keep the q>=0.5 centres, for the acquisition-time visual.
# run from ISAR-3DGS with PYTHONPATH=outputs/sh3_echo_daily_recovery_20261011/source_snapshot
import json, sys
from pathlib import Path
import numpy as np, torch
from hybrid_worker import install
install('sh3')
from radar_gs.io import load_scene
from radar_gs.data import load_dataset
from radar_gs.presets import load_encounter_batch_preset
from radar_gs.losses import compress_amplitude
import evaluate_ablation as ev
dev = torch.device('cuda'); torch.cuda.set_per_process_memory_fraction(0.2)
G = Path('outputs/goes_main_20261025'); DAYS = [1, 3, 5, 7, 9, 12, 16]
preset = load_encounter_batch_preset(Path('outputs/bsat_observation_days_20261004/evaluation_preset.json'), allow_incomplete=True)
out = {'days': np.array(DAYS), 'gt_pts': np.load(G / 'goes_gt_blender_5000.npy')}
views = None
for d in DAYS:
    scene, ck = load_scene(G / f'D{d:02d}_sh3_mask/training/checkpoints/final.pt', dev); cfg = ck['arguments']
    if views is None:
        views, _ = load_dataset(cfg['data_root'], image_size=256, native_dr=cfg['native_dr'], native_dd=cfg['native_dd'],
                                encounter_names=preset['encounter_names'], frames_per_encounter=1,
                                background_threshold=cfg['background_threshold'], normalization_percentile=cfg['normalization_percentile'])
        idx = [int(i) for i in sys.argv[1:]] or [len(views) // 2]
        out['view_idx'] = np.array(idx); out['view_names'] = np.array([str(views[i].encounter) for i in idx])
        out['gt_img'] = np.stack([views[i].image.cpu().numpy() for i in idx])
    renderer = ev.RadarVisibilityRenderer(256, 256, cfg['depth_bins'], cfg['kernel_radius'], cfg['minimum_pixel_variance'], cfg['opacity_gain'],
                                          cfg.get('visibility_pixel_size', 0.0), cfg.get('support_composition', 'noisy_or')).to(dev)
    T = cfg.get('final_support_temperature', 0.35); gamma = float(cfg.get('render_gamma', 1.0))
    with torch.no_grad():
        q = scene.support_probability(T)
        imgs = []
        for i in idx:
            v = views[i]
            o = renderer(scene, v.kr, v.kd, v.dr, v.dd, support_probability=q, support_temperature=T)
            imgs.append(compress_amplitude(o.image, gamma).cpu().numpy())
        mu = scene.means.detach() if hasattr(scene, 'means') else scene.mu.detach()
        out[f'img_D{d:02d}'] = np.stack(imgs)
        out[f'pts_D{d:02d}'] = mu[q >= 0.5].cpu().numpy().astype(np.float32)
    print('day', d, 'struct', int((q >= 0.5).sum()), flush=True)
np.savez_compressed(G / 'figs/goes_days_visual.npz', **out)
print('saved', out['view_names'])
