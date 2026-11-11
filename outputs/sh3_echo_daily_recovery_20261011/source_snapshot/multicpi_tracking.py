"""Geometry gated static scatterer tracks; no assumption of constant brightness."""
import numpy as np
from scipy.ndimage import maximum_filter

def track_keypoints_multi_cpi(images, geometry, bbox=6., max_peaks=120, gate_px=2., min_views=4):
    """geometry contains kr,kd,dr,dd. Returns detections and conservative tracks.
    Track IDs are hypotheses, not verified physical correspondences.
    Uses centered renderer convention: row=center-range/dr.
    """
    detections=[]; matrices=[]; observations=[]; descriptors=[]
    for im,g in zip(images,geometry):
        im=np.asarray(im,float); h,w=im.shape
        rc=np.argwhere((im==maximum_filter(im,size=5))&(im>max(.08*im.max(),np.median(im)+4*np.median(abs(im-np.median(im))))))
        rc=rc[np.argsort(im[rc[:,0],rc[:,1]])[::-1]][:max_peaks]
        pad=np.pad(im,3); patches=np.array([pad[r:r+7,c:c+7].ravel() for r,c in rc]).reshape(-1,49)
        patches/=np.maximum(np.linalg.norm(patches,axis=1,keepdims=True),1e-9)
        A=np.stack([-np.asarray(g['kr'])/g['dr'],np.asarray(g['kd'])/g['dd']])
        detections.append(rc); matrices.append(A); observations.append(rc-np.array([(h-1)/2,(w-1)/2]));descriptors.append(patches)
    candidates=[]
    def fit(obs):
        A=np.concatenate([matrices[v] for v,k in obs]); y=np.concatenate([observations[v][k] for v,k in obs]);p=np.linalg.lstsq(A,y,rcond=None)[0]
        err=np.linalg.norm((A@p-y).reshape(-1,2),axis=1);sv=np.linalg.svd(A,compute_uv=False)
        return p,err,sv
    # ponytail: pair hypotheses scale quadratically; capped peaks and seed frames.
    for i in range(0,len(images)-1,4):
        j=next((j for j in range(i+1,min(i+17,len(images))) if np.linalg.cond(np.concatenate([matrices[i],matrices[j]]))<30),None)
        if j is None:continue
        for ki in range(len(detections[i])):
            options=[]
            for kj in np.where(descriptors[j]@descriptors[i][ki]>.85)[0]:
                p,e,sv=fit([(i,ki),(j,int(kj))])
                if e.max()<=gate_px and abs(p).max()<=bbox:options.append((float(e.max()),int(kj),p))
            options.sort(key=lambda x:x[0])
            if not options or (len(options)>1 and options[1][0]-options[0][0]<.25):continue
            _,kj,p=options[0];obs=[(i,ki),(j,kj)]
            for v in range(i+1,len(images)):
                if v==j or len(detections[v])==0:continue
                dist=np.linalg.norm(observations[v]-matrices[v]@p,axis=1)
                sim=descriptors[v]@descriptors[i][ki]
                ids=np.where((dist<=gate_px)&(sim>.75))[0]
                if not len(ids):continue
                ids=ids[np.argsort(dist[ids])]
                if len(ids)>1 and dist[ids[1]]-dist[ids[0]]<.5:continue
                trial=obs+[(v,int(ids[0]))];p2,e,sv=fit(trial)
                if e.max()<=gate_px and abs(p2).max()<=bbox:obs=trial;p=p2
            if len(obs)<min_views:continue
            p,e,sv=fit(obs);uncertainty=gate_px/sv[-1]
            if uncertainty>.15:continue
            candidates.append(dict(point=p.tolist(),observations=[list(o) for o in obs],rms_px=float(np.sqrt(np.mean(e**2))),uncertainty_m=float(uncertainty)))
    tracks=[];used=set()
    for t in sorted(candidates,key=lambda t:(-len(t['observations']),t['rms_px'])):
        keys=set(map(tuple,t['observations']))
        if keys&used:continue
        used|=keys;t['id']=len(tracks);tracks.append(t)
    return detections,tracks

def self_check():
    # Exact projected point with permuted distractors; full rank fit and empty input.
    imgs=[];gs=[];p=np.array([.1,.2,.3])
    for a in np.linspace(0,1.2,8):
        kr=np.array([np.cos(a),0,np.sin(a)]);kd=np.array([0,1,0]);im=np.zeros((101,101));rc=np.round([50-kr@p/.02,50+kd@p/.02]).astype(int);im[tuple(rc)]=1
        imgs.append(im);gs.append(dict(kr=kr,kd=kd,dr=.02,dd=.02))
    _,ts=track_keypoints_multi_cpi(imgs,gs)
    assert ts and np.linalg.norm(np.array(ts[0]['point'])-p)<.03
    assert not track_keypoints_multi_cpi([np.zeros((10,10))]*4,gs[:4])[1]
    print('TRACK_SYNTHETIC_CHECK_OK',flush=True)
