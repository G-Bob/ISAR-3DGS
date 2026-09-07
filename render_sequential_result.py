"""Save native radar predictions at recorded LOS and fixed-axis point-cloud views."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radar_gs.ply_evaluation import read_ascii_ply
import torch
from radar_gs.io import load_scene
from radar_gs.renderer import RadarVisibilityRenderer
from radar_gs.losses import compress_amplitude, display_amplitude
from radar_gs.geometry import orthonormalize_view

p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
run=Path(a.run);out=Path(a.out)
table=read_ascii_ply(run/'gaussians_final.ply')
q=table.column('target_support_probability');xyz=np.column_stack([table.column(x) for x in ('x','y','z')]);xyz=xyz[q>=.3]
fig=plt.figure(figsize=(7,5));ax=fig.add_subplot(111,projection='3d')
ax.scatter(*xyz.T,s=.6,c='royalblue',alpha=.7,rasterized=True)
ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',xlim=(-5,5),ylim=(-2,2),zlim=(-2,2))
ax.set_box_aspect((10,4,4));ax.view_init(elev=22,azim=-65);ax.set_title(run.name+' | q >= 0.3')
fig.tight_layout();fig.savefig(out/'pointcloud_native_cad.png',dpi=220);plt.close(fig)
scene,checkpoint=load_scene(run/'checkpoints/final.pt',torch.device('cuda'))
c=checkpoint['arguments'];scene.eval()
renderer=RadarVisibilityRenderer(c['image_size'],c['image_size'],c['depth_bins'],c['kernel_radius'],c['minimum_pixel_variance'],c['opacity_gain'],c.get('visibility_pixel_size',0.),c.get('support_composition','noisy_or')).cuda()
folder=Path(c['data_root'])/c['encounter_names'][0]
geometry=np.load(folder/'view_angle.npy')
image=Image.open(next((folder/'isar_images_noaxis').glob('*.png')))
dr=c['native_dr']*image.height/c['image_size'];dd=c['native_dd']*image.width/c['image_size']
records=[]
for index in [0,len(geometry)//2,len(geometry)-1]:
    kr,kd,_=orthonormalize_view(torch.tensor(geometry[index,:3],dtype=torch.float32),torch.tensor(geometry[index,3:6],dtype=torch.float32))
    with torch.no_grad():
        result=renderer(scene,kr,kd,dr,dd,support_temperature=c.get('final_support_temperature',1.))
        values=display_amplitude(compress_amplitude(result.raw_image,c.get('render_gamma',1.))).cpu().numpy()
    Image.fromarray(np.uint8(np.clip(values,0,1)*255)).save(out/f'prediction_raw_E001_frame{index:05d}.png')
    records.append({'frame_index':index,'kr':kr.tolist(),'kd':kd.tolist(),'dr':dr,'dd':dd,'mode':'native raw renderer, no visibility gating, no background'})
(out/'matched_render_los.json').write_text(json.dumps(records,indent=2))
