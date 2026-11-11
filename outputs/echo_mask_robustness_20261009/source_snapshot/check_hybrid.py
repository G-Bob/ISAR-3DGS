import json,os,sys,time
from pathlib import Path
import torch
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
sys.path.insert(0,str(Path('outputs').resolve()))
from hybrid_worker import install
from radar_gs.io import load_scene
from radar_gs.data import load_dataset
from radar_gs.renderer import RadarVisibilityRenderer
import round6_fast_worker as r6
from radar_gs.losses import compress_amplitude

mode=os.environ['HYBRID_MODE'];install(mode);r6.GUARD=True;r6.TRACK=True
run=Path('outputs/bsat_methods8_20261004/M19_track_mask_guard_fast_restart')
scene,ck=load_scene(run/'training/checkpoints/final.pt',torch.device('cuda'));cfg=ck['arguments']
views,_=load_dataset(cfg['data_root'],image_size=256,native_dr=cfg['native_dr'],native_dd=cfg['native_dd'],encounter_names=cfg['encounter_names'][:1],frames_per_encounter=1,images_per_encounter=1,background_threshold=cfg['background_threshold'],normalization_percentile=cfg['normalization_percentile']);v=views[0]
renderer=RadarVisibilityRenderer(256,256,depth_bins=cfg['depth_bins'],kernel_radius=cfg['kernel_radius'],minimum_pixel_variance=cfg['minimum_pixel_variance'],opacity_gain=cfg['opacity_gain'],support_composition=cfg['support_composition']).cuda()
t=time.perf_counter();out=renderer(scene,v.kr,v.kd,v.dr,v.dd,support_temperature=.35);signal=out.raw_image if mode=='no_visibility' else out.image
target=v.image.cuda();pred=compress_amplitude(signal,1.);loss,_=r6.masked_loss(pred,target,.1,.2);track=r6.TrackConstraint();track.shape=target.shape;loss=loss+track(scene,1000);loss.backward();torch.cuda.synchronize()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in scene.parameters())
assert scene.means.grad.abs().sum()>0
if mode=='sh3':assert scene.features.grad.abs().sum()>0
if mode=='full_covariance':
    assert scene.pei_quaternions.grad.abs().sum()>0
    before=scene.count
    result=scene.adaptive_control(torch.ones(before,device='cuda'),torch.ones(before,device='cuda'),torch.ones(before,device='cuda'),.35,.075,.0001,.3,.5,0.,before+5,2000)
    assert len(scene.pei_quaternions)==scene.count
    assert scene.count!=before,'Topology gate must actually replace parameters'
    opt=torch.optim.Adam(scene.parameter_groups(.001));out=renderer(scene,v.kr,v.kd,v.dr,v.dd,support_temperature=.35);l=out.image.mean();l.backward();assert torch.isfinite(scene.pei_quaternions.grad).all();opt.step()
d={'mode':mode,'count':scene.count,'loss':float(loss.detach()),'seconds':time.perf_counter()-t,'means_gradient':float(scene.means.grad.norm()),'feature_gradient':float(scene.features.grad.norm()),'passed':True}
Path('outputs/paper_reproduction/check_hybrid_'+mode+'.json').write_text(json.dumps(d,indent=2));print('HYBRID_CHECK_PASSED',json.dumps(d),flush=True)
