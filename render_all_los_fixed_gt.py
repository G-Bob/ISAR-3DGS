#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def unit(x):
    x=np.asarray(x,float)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def equal_fixed(ax, center, half):
    ax.set_xlim(center[0]-half,center[0]+half)
    ax.set_ylim(center[1]-half,center[1]+half)
    ax.set_zlim(center[2]-half,center[2]+half)
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=22,azim=-58)
    ax.set_axis_off()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--gt",type=Path,required=True)
    p.add_argument("--coverage-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    mesh=o3d.io.read_triangle_mesh(str(a.gt),enable_post_processing=True)
    verts=np.asarray(mesh.vertices); tri=np.asarray(mesh.triangles)
    centers=(verts[tri[:,0]]+verts[tri[:,1]]+verts[tri[:,2]])/3
    box=mesh.get_axis_aligned_bounding_box(); c=np.asarray(box.get_center())
    model_half=float(np.max(box.get_extent())/2)*1.08
    sphere_radius=model_half*1.55
    palette=plt.get_cmap("tab10")

    fig=plt.figure(figsize=(20,10))
    sphere_fig=plt.figure(figsize=(20,10))
    manifest={}
    for panel,k in enumerate(range(2,10),1):
        preset=json.loads((a.data_root/f"preset_batch_20260822_{k}.json").read_text())
        ax=fig.add_subplot(2,4,panel,projection="3d")
        ax.scatter(centers[:,0],centers[:,1],centers[:,2],c="#777777",s=.12,alpha=.28)
        sax=sphere_fig.add_subplot(2,4,panel,projection="3d")
        u=np.linspace(0,2*np.pi,30); v=np.linspace(0,np.pi,16)
        sx=np.outer(np.cos(u),np.sin(v)); sy=np.outer(np.sin(u),np.sin(v)); sz=np.outer(np.ones_like(u),np.cos(v))
        sax.plot_wireframe(sx,sy,sz,color="#bbbbbb",linewidth=.25,alpha=.35)
        records=[]
        for ei,enc in enumerate(preset["encounters"]):
            los=unit(np.load(a.data_root/enc["folder_name"]/"range_axis_cad.npy"))
            pos=c+los*sphere_radius; color=palette(ei%10)
            # Every one of the 60 actually trained LOS samples is shown.
            ax.scatter(pos[:,0],pos[:,1],pos[:,2],color=color,s=5,alpha=.82)
            ax.plot(pos[:,0],pos[:,1],pos[:,2],color=color,linewidth=.8,alpha=.65)
            sax.scatter(los[:,0],los[:,1],los[:,2],color=color,s=7,alpha=.9)
            sax.plot(los[:,0],los[:,1],los[:,2],color=color,linewidth=.9,alpha=.75)
            records.append({"encounter":enc["folder_name"],"color_index":ei,"view_count":len(los),"los":los.tolist()})
        equal_fixed(ax,c,sphere_radius*1.08); ax.set_title(f"K={k}: fixed GT + all {60*k} LOS",fontsize=11)
        equal_fixed(sax,np.zeros(3),1.08); sax.set_title(f"K={k}: LOS unit sphere ({k} arcs)",fontsize=11)
        manifest[str(k)]=records
    fig.suptitle("Fixed GT geometry and every LOS used in each preset (same camera and scale)",fontsize=16)
    fig.tight_layout(); fig.savefig(a.out/"K02_K09_fixed_gt_all_los.png",dpi=240,bbox_inches="tight"); plt.close(fig)
    sphere_fig.suptitle("Every training LOS on the unit sphere; color identifies encounter",fontsize=16)
    sphere_fig.tight_layout(); sphere_fig.savefig(a.out/"K02_K09_all_los_unit_sphere.png",dpi=240,bbox_inches="tight"); plt.close(sphere_fig)
    (a.out/"all_los_manifest.json").write_text(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
