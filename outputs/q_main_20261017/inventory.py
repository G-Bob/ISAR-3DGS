import json,sys
from pathlib import Path
ref=json.loads(Path('outputs/paper_reproduction/M46_EchoMask_SH3/config.json').read_text())
B=Path('outputs/bsat_methods8_20261004'); P=Path('outputs/paper_reproduction'); Q=Path('outputs/support_causality_20261008'); D=Path('outputs/sh3_echo_daily_recovery_20261011')
runs=[P/'M46_EchoMask_SH3',B/'M15_multicpi_tracks',B/'M16_track_constraint_fast_restart',B/'M16_track_constraint_parallel_restart',B/'M17_track_mask_fast_restart',B/'M17_track_constraint_mask',B/'M18_echo_mask_only',B/'M19_track_mask_guard_fast_restart',B/'M19_track_mask_background_guard',B/'M20_mask_background_guard',P/'M47_guard_only',P/'M48_tracking_guard_no_mask',P/'M30_M19_SH3',P/'M31_M19_no_visibility',P/'M32_M19_full_covariance']+sorted(Q.glob('Q4*'))+[D/'D01',D/'D16']
out={}
for r in runs:
    e=dict(exists=r.exists())
    if r.exists():
        c=r/'config.json'
        if c.exists():
            cfg=json.loads(c.read_text()); e['diff']={k:v for k,v in cfg.items() if ref.get(k)!=v and k not in('out_dir','config','encounter_names_file','seed')}
        s=r/'started.json'
        if s.exists():
            st=json.loads(s.read_text()); e['cmd']=' '.join(str(x) for x in st.get('command',[]))[:400]; e['env']={k:v for k,v in st.items() if k not in('command','pid','time','job')}
        e['files']=sorted(p.name for p in r.iterdir())[:30]
        e['summary']=(r/'evaluation_common475'/'summary.json').exists()
    out[str(r)]=e
Path('outputs/q_main_20261017/inventory.json').write_text(json.dumps(out,indent=1))
print('ok',len(out))
