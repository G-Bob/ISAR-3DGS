"""Audit native CAD acquisition and build chronological, nested training jobs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh

BASE = Path('/root/autodl-fs/radar_gs_isar_enc')
WORK = BASE / 'workspaces/v21_sequential_drift_20260907'
ROOT = BASE / 'outputs/radar_gs/skynet5d3_sequential_v21_20260907'
DATA = Path('/root/Yao/datasets/skynet5d-3')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'jobs.json').exists():
        raise RuntimeError('Existing job manifest: refuse to regenerate')
    legacy_path = BASE / 'outputs/radar_gs/skynet5d2_k1_k10_optimal_v19_20260902/training/k4/run_config.json'
    baseline = json.loads(legacy_path.read_text())
    transform_path = Path('/root/Yao/3dmodel/skynet5d_half_scale_prediction_frame.transform.json')
    transform = json.loads(transform_path.read_text())
    gt = trimesh.load(transform['output_glb'], force='scene', process=False)
    gt.apply_transform(np.linalg.inv(np.asarray(transform['homogeneous_transform'])))
    assert np.allclose(gt.bounds, transform['original_bounds'], atol=1e-5)
    gt_path = ROOT / 'gt_native_cad_recovered.glb'
    gt.export(gt_path)
    write(ROOT / 'gt_provenance.json', {'source_transform': transform, 'operation': 'inverse historical rigid transform, no fitted alignment', 'output': str(gt_path), 'bounds': gt.bounds.tolist()})
    jobs, audits = [], {}
    for scenario, maximum, milestones in [('CPA50',59,[15,20,30,40,50,59]), ('CPA80',36,[15,20,30,36])]:
        entries = {}
        for folder in sorted((DATA / scenario).glob('skynet*')):
            meta = json.loads((folder / 'static_los_metadata.json').read_text())
            p = meta['preset_encounter']
            number = p['encounter_number']
            assert number not in entries, (scenario,number,'duplicate')
            assert meta['canonical_object_frame']['cad_sha256'] == transform['input_sha256_before']
            views = np.load(folder / 'view_angle.npy')
            anchors = np.load(folder / 'frame_anchor_indices.npy')
            images = sorted((folder/'isar_images_noaxis').glob('*.png'))
            assert len(images) == len(views) == len(anchors) == p['frame_count'], (scenario,number,'incomplete')
            assert np.isfinite(views).all()
            assert np.allclose(views[:,:3], np.load(folder/'range_axis_cad.npy'), atol=1e-6)
            assert np.allclose(views[:,:3], -np.load(folder/'physical_incident_cad.npy'), atol=1e-6)
            assert np.allclose(np.linalg.norm(views[:,:3],axis=1),1,atol=1e-5)
            assert np.allclose(np.linalg.norm(views[:,3:6],axis=1),1,atol=1e-5)
            assert np.max(np.abs(np.sum(views[:,:3]*views[:,3:6],axis=1))) < 1e-5
            entries[number] = {'encounter_number':number,'ca_utc':p['ca_utc'], 'folder_name':folder.name,'frame_count':len(views), 'arc_length_deg':p['total_aperture_deg'], 'range_axis_center_cad':p['range_axis_center_cad'], 'omega_axis_cad':p['omega_axis_cad'], 'pass_side':p['pass_side'], 'los':views[:,:3].tolist(), 'metadata_sha256':hashlib.sha256((folder/'static_los_metadata.json').read_bytes()).hexdigest()}
        assert sorted(entries) == list(range(1,maximum+1))
        ordered = [entries[i] for i in range(1,maximum+1)]
        assert [x['ca_utc'] for x in ordered] == sorted(x['ca_utc'] for x in ordered)
        audits[scenario] = ordered
        for k in list(range(1,11))+milestones:
            names = [e['folder_name'] for e in ordered[:k]]
            names_path = ROOT/'presets'/f'{scenario}_k{k:02d}.txt'
            names_path.parent.mkdir(exist_ok=True)
            names_path.write_text('\n'.join(names)+'\n')
            write(names_path.with_suffix('.json'), {'schema':'audited_chronological_prefix_v1','K':k,'encounters':ordered[:k], 'total_frames':sum(e['frame_count'] for e in ordered[:k])})
            for variant, levels in [('angular_low',1)] + ([('legacy_control',4)] if k in (4,10) else []):
                name = f'{scenario}_k{k:02d}_{variant}'
                config = dict(baseline)
                config.update(data_root=str(DATA/scenario),out_dir=str(ROOT/'training'/name), config=None,
                    encounter_preset_json=None,encounter_preset_metadata=None,encounter_names=None,
                    encounter_names_file=str(names_path),encounter_count=k,
                    frames_per_encounter=None,images_per_encounter=None,frame_start=0,los_angle_deg=0,
                    direction_encoding_levels=levels,encoding_levels=4,
                    split_min_positive_encounters=min(k,2),validate_only=False)
                config_path=ROOT/'configs'/f'{name}.json'
                write(config_path,config)
                jobs.append({'name':name,'scenario':scenario,'k':k,'variant':variant,'config':str(config_path),'run_dir':config['out_dir'],'gt':str(gt_path),'total_frames':sum(e['frame_count'] for e in ordered[:k])})
    # Pair the smallest controlled runs first; remaining K retain their exact chronological prefix.
    jobs.sort(key=lambda j:(0 if j['k']==4 else 1 if j['k']<=10 else 2,j['k'],j['scenario'],j['variant']))
    write(ROOT/'acquisition_audit.json',audits)
    write(ROOT/'jobs.json',jobs)
    write(ROOT/'protocol.json',{'seed':42,'iterations':12000,'training':'independent from-scratch models of nested chronological prefixes; all recorded frames', 'spatial_encoding_levels':4,'direction_encoding_levels':{'angular_low':1,'legacy_control':4},'paper':'https://arxiv.org/abs/2608.30054','interpretation':'Reduce direction input frequencies; nonlinear MLP is not a strictly bandlimited SH model. Optical first-hit theorem is not assumed valid for radar.', 'evaluation':'prefix in-sample amplitude reconstruction and full native-CAD geometry; no held-out claim','native_dr':baseline['native_dr'],'native_dd':baseline['native_dd'],'pixel_calibration_provenance':'inherited previous simulator calibration; new metadata has identical pulse/CPI shape but does not record bandwidth','geometry':'GT used only for final evaluation','motion':'propagated CSV angular replay; fixed range and equivalent constant angular rate as supplied'})
    print(json.dumps({'jobs':len(jobs),'scenario_frames':{s:sum(e['frame_count'] for e in es) for s,es in audits.items()}, 'root':str(ROOT)},indent=2))


if __name__=='__main__':
    main()
