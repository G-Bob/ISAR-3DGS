import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/root/autodl-fs/radar_gs_isar_enc/outputs/radar_gs/skynet5d3_sequential_v21_20260907')
audit=json.loads((root/'acquisition_audit.json').read_text())
fig=plt.figure(figsize=(12,5))
for i,(scenario,entries) in enumerate(audit.items(),1):
    ax=fig.add_subplot(1,2,i,projection='3d')
    for e in entries:
        los=np.asarray(e['los'])
        color=plt.cm.viridis((e['encounter_number']-1)/(len(entries)-1))
        ax.plot(*los.T,color=color,lw=.6,alpha=.7)
        ax.scatter(*los.T,s=1,color=color)
    ax.set(xlabel='CAD X',ylabel='CAD Y',zlabel='CAD Z',xlim=(-1,1),ylim=(-1,1),zlim=(-1,1),title=f'{scenario}: all recorded LOS, E1-E{len(entries)}')
    ax.set_box_aspect((1,1,1));ax.view_init(23,-60)
    sm=plt.cm.ScalarMappable(norm=plt.Normalize(1,len(entries)),cmap='viridis')
    fig.colorbar(sm,ax=ax,shrink=.65,pad=.1,label='Chronological encounter number')
fig.tight_layout();fig.savefig(root/'recorded_los_drift.png',dpi=180)
print('Saved',root/'recorded_los_drift.png')
