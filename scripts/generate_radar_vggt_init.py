import argparse
from pathlib import Path
import numpy as np
from radar_gs.data import load_dataset
from radar_gs.radar_vggt import radar_vggt_surface
p=argparse.ArgumentParser()
p.add_argument('--data_root',required=True); p.add_argument('--out',required=True)
p.add_argument('--encounter_count',type=int,required=True); p.add_argument('--images_per_encounter',type=int,default=50)
p.add_argument('--count',type=int,default=10000); p.add_argument('--bbox',type=float,default=6.0); p.add_argument('--seed',type=int,default=42)
p.add_argument('--view_count',type=int,default=16); p.add_argument('--patch_grid',type=int,default=16); p.add_argument('--steps',type=int,default=120)
a=p.parse_args()
views,_=load_dataset(a.data_root,image_size=256,native_dr=.0149896,native_dd=.0148244,los_angle_deg=0,encounter_count=a.encounter_count,encounter_names=None,frames_per_encounter=None,images_per_encounter=a.images_per_encounter,frame_start=0,background_threshold=0,normalization_percentile=99.5,amplitude_processing_mode='control',psf_beta_range=3,psf_beta_doppler=5,psf_mask_minimum_weight=.25,clean_gain=.1,clean_maximum_removed_energy_fraction=.1,legacy_background_threshold_for_decode=.1)
points,normals,info=radar_vggt_surface(a.count,a.bbox,a.seed,views,view_count=a.view_count,patch_grid=a.patch_grid,steps=a.steps,device='cuda')
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,points=np.concatenate((points,normals),axis=1)); print(info,flush=True)
