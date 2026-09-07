import json
import open3d as o3d

mesh=o3d.io.read_triangle_mesh('/root/Yao/3dmodel/skynet5d_half_scale_prediction_frame.glb',enable_post_processing=True)
pcd=mesh.sample_points_uniformly(number_of_points=5000)
pts=[[round(float(v),4) for v in p] for p in pcd.points]
with open('/root/Yao/radar_gs_isar_enc/outputs/preset_gt_coverage_open3d_20260825/gt_points_5000.json','w') as f:
    json.dump(pts,f,separators=(',',':'))
