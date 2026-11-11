"""One common475 image evaluation and all frozen geometry selections, no ICP."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from radar_gs.data import load_dataset
from radar_gs.presets import load_encounter_batch_preset
from radar_gs.ply_evaluation import ground_truth_metrics
import evaluate_ablation as ev


def main():
    a=argparse.ArgumentParser();a.add_argument('run');a.add_argument('--hybrid');a.add_argument('--paper');args=a.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    run=Path(args.run);training=run/'training' if args.hybrid else run;out=run/'evaluation_common475'
    assert not (out/'summary.json').exists(),'Completed evaluation must not repeat'
    out.mkdir(exist_ok=True);dev=torch.device('cuda')
    if args.hybrid:
        from hybrid_worker import install
        install(args.hybrid)
        from radar_gs.io import load_scene
        scene,ck=load_scene(training/'checkpoints/final.pt',dev)
        cfg=ck['arguments'];renderer_type=ev.RadarVisibilityRenderer
    else:
        os.environ['PAPER_MODE']=args.paper
        import adapted_papers as ap
        ap.install();scene,ck=ap.base.load_scene(training/'checkpoints/final.pt',dev)
        renderer_type=ap.base.LinearRenderer;cfg=ck['arguments']
    assert ck['iteration']==12000 and all(torch.isfinite(v).all() for v in ck['model'].values())
    old=Path('outputs/bsat_observation_days_20261004');preset=load_encounter_batch_preset(old/'evaluation_preset.json',allow_incomplete=True)
    views,_=load_dataset(cfg['data_root'],image_size=256,native_dr=cfg['native_dr'],native_dd=cfg['native_dd'],encounter_names=preset['encounter_names'],frames_per_encounter=1,background_threshold=cfg['background_threshold'],normalization_percentile=cfg['normalization_percentile'])
    keys=ev._training_view_keys(training/'selected_views.json');views=[v for v in views if (v.encounter,v.frame_index) not in keys];views=ev._fixed_evaluation_views(views,old/'evaluation_views_common475.json',475,42);assert len(views)==475
    ev.RadarVisibilityRenderer=renderer_type
    image=ev._evaluate_images(scene,ck,views,dev,out,0,'zero')
    xyz=scene.means.detach().cpu().numpy();q=scene.support_probability(cfg.get('final_support_temperature',1.)).detach().cpu().numpy();ref=np.load(old/'skynet_gt_blender_5000.npy')
    selections=[('all',np.ones(len(xyz),bool)),('q030',q>=.3)]
    if args.paper:selections.insert(1,('alpha004',q>=.004))
    geometry={n:{'points':int(m.sum()),'metrics':ground_truth_metrics(xyz[m],ref,[.1,.2,.3,.5]) if m.any() else {'available':False}} for n,m in selections}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,m in selections:
        fig,axs=plt.subplots(1,3,figsize=(15,5))
        for ax,(u,v) in zip(axs,[(0,1),(0,2),(1,2)]):
            ax.scatter(ref[:,u],ref[:,v],s=1,c='gray',alpha=.35,label='Skynet GT');ax.scatter(xyz[m,u],xyz[m,v],s=1,c='tab:blue',alpha=.4,label=name);ax.set_aspect('equal');ax.set_xlabel('xyz'[u]+' (m)');ax.set_ylabel('xyz'[v]+' (m)')
        axs[0].legend();fig.suptitle(run.name+' / '+name+' / fixed coordinates');fig.tight_layout();fig.savefig(out/('geometry_'+name+'.png'),dpi=160);plt.close(fig)
    renderer=renderer_type(256,256,depth_bins=cfg.get('depth_bins',32),kernel_radius=cfg.get('kernel_radius',4),minimum_pixel_variance=cfg.get('minimum_pixel_variance',.35),opacity_gain=cfg.get('opacity_gain',1.),support_composition=cfg.get('support_composition','optical_density')).cuda()
    for j in [0,68,203,339,474]:
        v=views[j]
        with torch.no_grad():
            rr=renderer(scene,v.kr,v.kd,v.dr,v.dd,support_temperature=cfg.get('final_support_temperature',1.));pred=(rr.raw_image if cfg.get('disable_visibility') else rr.image).clamp(0,1).cpu().numpy()
        fig,axs=plt.subplots(1,3,figsize=(10,3.5))
        for ax,im,title in zip(axs,[v.image.numpy(),pred,abs(pred-v.image.numpy())],['Target','Prediction','Absolute residual']):ax.imshow(im,cmap='gray',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
        fig.tight_layout();fig.savefig(out/f'preview_{j:03d}.png',dpi=130);plt.close(fig)
    summary={'method':run.name,'views':475,'image':image,'geometry':geometry,'alignment':'fixed Skynet GT, no ICP','selection_warning':'support probability and energy opacity differ; report all predefined selections','scalar_image_protocol':'RNS inverse embedding channel0; RGB training reported separately','hypothesis_limits':json.loads((run/'PLAN.json').read_text())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('SUITE_EVAL_DONE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
