"""Prepare a warm-start chain with natural proportional cumulative sampling."""
import json
from pathlib import Path

BASE=Path('/root/autodl-fs/radar_gs_isar_enc')
CONTROL=BASE/'outputs/radar_gs/skynet5d4_incremental_v23_20260909'
ROOT=BASE/'outputs/radar_gs/skynet5d4_replay_v24_20260910'
VARIANT='proportional'
DATA=Path('/root/Yao/datasets/skynet5d-4')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2))

def main():
    vr=ROOT/VARIANT
    if (vr/'jobs.json').exists(): raise RuntimeError('Existing proportional manifest: refuse to overwrite')
    audit=json.loads((CONTROL/'acquisition_audit.json').read_text())
    baseline=json.loads((CONTROL/'configs/stage_k02_encounter_02.json').read_text())
    gt=json.loads((CONTROL/'jobs.json').read_text())[0]['gt'];jobs=[]
    for k in range(2,9):
        name=f'stage_k{k:02d}_{VARIANT}';run=vr/'training'/name
        names=vr/'presets'/f'stage_k{k:02d}_training.txt';names.parent.mkdir(parents=True,exist_ok=True);names.write_text('\n'.join(e['folder_name'] for e in audit[:k])+'\n')
        current=vr/'presets'/f'stage_k{k:02d}_current.json';cumulative=vr/'presets'/f'stage_k{k:02d}_cumulative.json'
        write(current,{'selected_k':1,'encounters':[{'folder_name':audit[k-1]['folder_name']}]})
        write(cumulative,{'selected_k':k,'encounters':[{'folder_name':e['folder_name']} for e in audit[:k]]})
        previous=(CONTROL/'training/stage_k01_encounter_01/checkpoints/final.pt' if k==2 else vr/'training'/f'stage_k{k-1:02d}_{VARIANT}'/'checkpoints/final.pt')
        config=dict(baseline);config.update(data_root=str(DATA),out_dir=str(run),config=None,encounter_preset_json=None,encounter_preset_metadata=None,encounter_names=None,encounter_names_file=str(names),encounter_count=k,frames_per_encounter=None,images_per_encounter=None,frame_start=0,los_angle_deg=0,iters=6000,checkpoint_interval=100,preview_interval=100,seed=42,warm_start_checkpoint=str(previous),continual_current_encounter=None,continual_replay_fraction=0.0,protect_warm_start_support_threshold=0.0,validate_only=False)
        cp=vr/'configs'/f'{name}.json';write(cp,config);jobs.append({'name':name,'variant':VARIANT,'k':k,'config':str(cp),'run_dir':str(run),'current_preset':str(current),'cumulative_preset':str(cumulative),'gt':gt})
    write(vr/'jobs.json',jobs);write(vr/'protocol.json',{'sampling':'uniform over all frames in encounters 1..K','expected_new_fraction':'frames(E_K) / frames(E_1..E_K)','iterations_per_stage':6000,'shared_k1_control':str(CONTROL)})
    print(json.dumps({'jobs':len(jobs),'variant':VARIANT},indent=2))

if __name__=='__main__':main()
