#!/usr/bin/env python3
"""Overlay q_i points on a transparent original GLB without remeshing it.

The original GLB scene and textures are retained. Gaussian PLY points are
appended as opaque coloured spheres. ``--alignment auto`` estimates a
translation, rotation and uniform scale with PCA plus trimmed similarity ICP.

Dependencies: numpy, scipy
"""

from __future__ import annotations

import argparse
import itertools
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
SCRIPT_VERSION = "2.1-distance"


@dataclass(frozen=True)
class Cloud:
    xyz: np.ndarray
    qi: np.ndarray


@dataclass
class GLB:
    meta: dict
    binary: bytearray


def read_ply(path: Path) -> Cloud:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{path}: not a PLY file")
    try:
        end = next(i for i, line in enumerate(lines) if line.strip() == "end_header")
    except StopIteration as exc:
        raise ValueError(f"{path}: missing end_header") from exc
    if not any(line.strip() == "format ascii 1.0" for line in lines[: end + 1]):
        raise ValueError("only ASCII PLY is supported")
    count = None
    element = None
    names: list[str] = []
    for line in lines[: end + 1]:
        fields = line.split()
        if fields[:1] == ["element"] and len(fields) == 3:
            element = fields[1]
            if element == "vertex":
                count = int(fields[2])
        elif fields[:1] == ["property"] and element == "vertex":
            if len(fields) != 3 or fields[1] == "list":
                raise ValueError("list vertex properties are unsupported")
            names.append(fields[2])
    if count is None:
        raise ValueError("missing PLY vertex element")
    rows = lines[end + 1 : end + 1 + count]
    values = np.asarray([[float(x) for x in row.split()] for row in rows])
    if values.shape != (count, len(names)):
        raise ValueError("PLY vertex rows do not match its header")
    if not all(name in names for name in ("x", "y", "z")):
        raise ValueError("PLY requires x, y and z")
    xyz = values[:, [names.index(name) for name in ("x", "y", "z")]]
    qname = "target_support_probability"
    qi = values[:, names.index(qname)] if qname in names else np.ones(count)
    return Cloud(xyz, qi)


def filter_cloud(cloud: Cloud, threshold: float, neighbours: int, std_ratio: float) -> Cloud:
    keep = cloud.qi >= threshold
    xyz, qi = cloud.xyz[keep], cloud.qi[keep]
    print(f"q_i filtering: {len(cloud.xyz)} -> {len(xyz)} points")
    if len(xyz) < 4:
        raise ValueError("fewer than four Gaussian points remain")
    if neighbours > 0:
        k = min(neighbours, len(xyz) - 1)
        distance, _ = cKDTree(xyz).query(xyz, k=k + 1)
        mean_d = distance[:, 1:].mean(axis=1)
        local = mean_d <= mean_d.mean() + std_ratio * mean_d.std()
        print(f"Outlier filtering: {len(xyz)} -> {np.count_nonzero(local)} points")
        xyz, qi = xyz[local], qi[local]
    return Cloud(xyz, qi)


def read_glb(path: Path) -> GLB:
    payload = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or total != len(payload):
        raise ValueError("expected a valid GLB 2.0 file")
    metadata = None
    binary = b""
    offset = 12
    while offset < total:
        length, kind = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset : offset + length]
        offset += length
        if kind == JSON_CHUNK:
            metadata = json.loads(chunk.decode().rstrip(" \t\r\n\0"))
        elif kind == BIN_CHUNK:
            binary = chunk
    if metadata is None or len(metadata.get("buffers", [])) != 1:
        raise ValueError("GLB must contain one embedded buffer")
    return GLB(metadata, bytearray(binary))


DTYPES = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def accessor(glb: GLB, index: int) -> np.ndarray:
    acc = glb.meta["accessors"][index]
    if "sparse" in acc or "bufferView" not in acc:
        raise ValueError("sparse GLB accessors are unsupported")
    view = glb.meta["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(DTYPES[acc["componentType"]])
    components = NCOMP[acc["type"]]
    count = int(acc["count"])
    offset = int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
    packed = components * dtype.itemsize
    stride = int(view.get("byteStride", packed))
    if stride == packed:
        out = np.frombuffer(glb.binary, dtype, count * components, offset).reshape(count, components)
    else:
        out = np.ndarray((count, components), dtype, glb.binary, offset, (stride, dtype.itemsize))
    return np.array(out, copy=True)


def quaternion_matrix(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=float)
    n = x*x + y*y + z*z + w*w
    if n < 1e-15:
        return np.eye(4)
    s = 2.0 / n
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1-s*(y*y+z*z), s*(x*y-w*z), s*(x*z+w*y)],
        [s*(x*y+w*z), 1-s*(x*x+z*z), s*(y*z-w*x)],
        [s*(x*z-w*y), s*(y*z+w*x), 1-s*(x*x+y*y)],
    ]
    return matrix


def local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=float).reshape(4, 4, order="F")
    t = np.eye(4)
    t[:3, 3] = node.get("translation", [0, 0, 0])
    s = np.eye(4)
    s[np.arange(3), np.arange(3)] = node.get("scale", [1, 1, 1])
    return t @ quaternion_matrix(node.get("rotation", [0, 0, 0, 1])) @ s


def node_world_transforms(glb: GLB):
    scene = glb.meta["scenes"][int(glb.meta.get("scene", 0))]
    output = []
    def visit(index: int, parent: np.ndarray):
        node = glb.meta["nodes"][index]
        world = parent @ local_matrix(node)
        output.append((index, world))
        for child in node.get("children", []):
            visit(int(child), world)
    for root in scene.get("nodes", []):
        visit(int(root), np.eye(4))
    return output


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def world_triangles(glb: GLB) -> np.ndarray:
    blocks = []
    for node_index, world in node_world_transforms(glb):
        node = glb.meta["nodes"][node_index]
        if "mesh" not in node:
            continue
        for primitive in glb.meta["meshes"][node["mesh"]]["primitives"]:
            if primitive.get("mode", 4) != 4:
                continue
            pos = accessor(glb, primitive["attributes"]["POSITION"]).astype(float)
            pos = apply_transform(pos, world)
            ind = accessor(glb, primitive["indices"]).reshape(-1) if "indices" in primitive else np.arange(len(pos))
            blocks.append(pos[ind.astype(np.int64)].reshape(-1, 3, 3))
    if not blocks:
        raise ValueError("GLB contains no triangle mesh")
    return np.concatenate(blocks)


def sample_surface(triangles: np.ndarray, count: int, seed: int) -> np.ndarray:
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    valid = area > 1e-15
    triangles, area = triangles[valid], area[valid]
    rng = np.random.default_rng(seed)
    chosen = triangles[rng.choice(len(triangles), count, p=area / area.sum())]
    r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
    return ((1-r1)[:, None] * chosen[:, 0] + (r1*(1-r2))[:, None] * chosen[:, 1]
            + (r1*r2)[:, None] * chosen[:, 2])


def pca(points: np.ndarray) -> np.ndarray:
    x = points - np.median(points, axis=0)
    values, vectors = np.linalg.eigh(x.T @ x / len(x))
    return vectors[:, np.argsort(values)[::-1]]


def robust_radius(points, center) -> float:
    return float(np.percentile(np.linalg.norm(points - center, axis=1), 90))


def initial_alignment(source: np.ndarray, target: np.ndarray, trim: float):
    cs, ct = np.median(source, axis=0), np.median(target, axis=0)
    scale = robust_radius(target, ct) / robust_radius(source, cs)
    bs, bt = pca(source), pca(target)
    tree = cKDTree(target)
    count = max(4, int(trim * len(source)))
    best, best_score = None, np.inf
    for permutation in itertools.permutations(range(3)):
        permuted = bt[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = (permuted * np.asarray(signs)[None, :]) @ bs.T
            if np.linalg.det(rotation) <= 0:
                continue
            transform = np.eye(4)
            transform[:3, :3] = scale * rotation
            transform[:3, 3] = ct - scale * rotation @ cs
            distance, _ = tree.query(apply_transform(source, transform))
            score = float(np.mean(np.partition(distance, count - 1)[:count]))
            if score < best_score:
                best, best_score = transform, score
    return best, best_score


def similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    ms, mt = source.mean(axis=0), target.mean(axis=0)
    xs, xt = source - ms, target - mt
    u, singular, vt = np.linalg.svd(xt.T @ xs / len(xs))
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / np.mean(np.sum(xs * xs, axis=1)))
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = mt - scale * rotation @ ms
    return transform


def icp(source, target, initial, iterations, trim, tolerance):
    transform, tree, previous, history = initial.copy(), cKDTree(target), np.inf, []
    for iteration in range(iterations):
        current = apply_transform(source, transform)
        distance, indices = tree.query(current)
        keep = distance <= np.quantile(distance, trim)
        transform = similarity(current[keep], target[indices[keep]]) @ transform
        distance, _ = tree.query(apply_transform(source, transform))
        kept_distance = distance[distance <= np.quantile(distance, trim)]
        rmse = float(np.sqrt(np.mean(kept_distance ** 2)))
        history.append(rmse)
        print(f"ICP {iteration + 1:02d}: trimmed RMSE={rmse:.6g}")
        if np.isfinite(previous) and abs(previous-rmse) <= tolerance * max(previous, 1.0):
            break
        previous = rmse
    final, _ = tree.query(apply_transform(source, transform))
    metrics = {
        "iterations": len(history), "trimmed_rmse": history[-1],
        "median_nearest_surface_distance": float(np.median(final)),
        "p90_nearest_surface_distance": float(np.percentile(final, 90)),
        "uniform_scale": float(np.cbrt(abs(np.linalg.det(transform[:3, :3])))),
    }
    return transform, metrics


def icosphere(subdivisions: int):
    phi = (1 + np.sqrt(5.0)) / 2
    vertices = np.asarray([(-1,phi,0),(1,phi,0),(-1,-phi,0),(1,-phi,0),(0,-1,phi),
        (0,1,phi),(0,-1,-phi),(0,1,-phi),(phi,0,-1),(phi,0,1),(-phi,0,-1),(-phi,0,1)], float)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = [(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),(1,5,9),(5,11,4),
        (11,10,2),(10,7,6),(7,1,8),(3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)]
    vertices, faces = [v for v in vertices], list(faces)
    for _ in range(subdivisions):
        cache, new_faces = {}, []
        def middle(i, j):
            key = (min(i,j), max(i,j))
            if key not in cache:
                point = vertices[i] + vertices[j]
                vertices.append(point / np.linalg.norm(point))
                cache[key] = len(vertices)-1
            return cache[key]
        for a,b,c in faces:
            ab,bc,ca = middle(a,b),middle(b,c),middle(c,a)
            new_faces += [(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)]
        faces = new_faces
    return np.asarray(vertices), np.asarray(faces, np.uint32)


def viridis(qi):
    x = np.asarray([0,.25,.5,.75,1])
    rgb = np.asarray([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]])
    return np.rint(np.column_stack([np.interp(qi, x, rgb[:,i]) for i in range(3)])).astype(np.uint8)


def distance_palette(distance, near, far):
    """Yellow (near) -> cyan -> blue (far)."""
    value = np.clip((distance-near)/(far-near), 0.0, 1.0)
    stops = np.asarray([0.0, 0.5, 1.0])
    rgb = np.asarray([[255,230,0],[0,220,220],[20,70,255]])
    return np.rint(np.column_stack([
        np.interp(value, stops, rgb[:,i]) for i in range(3)
    ])).astype(np.uint8)


def markers(points, qi, surface_distance, args):
    unit, faces = icosphere(args.point_subdivisions)
    nearest, _ = cKDTree(points).query(points, k=2)
    spacing = float(np.median(nearest[:,1]))
    base = args.point_radius or args.radius_factor * spacing
    radius = base * (args.radius_q_floor + (1-args.radius_q_floor)*qi)
    position = points[:,None,:] + radius[:,None,None]*unit[None,:,:]
    normal = np.repeat(unit[None,:,:], len(points), axis=0)
    offset = (np.arange(len(points), dtype=np.uint32)*len(unit))[:,None,None]
    index = faces[None,:,:] + offset
    near = args.near_distance or 0.5*spacing
    far = args.far_distance or max(3.0*near, float(np.percentile(surface_distance,90)))
    if far <= near:
        raise ValueError("far distance must be greater than near distance")
    rgb = distance_palette(surface_distance,near,far) if args.color_mode=="distance" else viridis(qi)
    color = np.repeat(np.column_stack((rgb,np.full(len(points),255,np.uint8)))[:,None,:],len(unit),1)
    print(f"Markers: {len(points)} points, median radius={np.median(radius):.6g}")
    print(f"Surface distance: min={surface_distance.min():.6g}, median={np.median(surface_distance):.6g}, p90={np.percentile(surface_distance,90):.6g}, max={surface_distance.max():.6g}")
    print(f"Distance colours: yellow <= {near:.6g}, blue >= {far:.6g}")
    arrays = (position.reshape(-1,3).astype("<f4"), normal.reshape(-1,3).astype("<f4"), color.reshape(-1,4), index.reshape(-1).astype("<u4"))
    return arrays, {
        "mode":args.color_mode,
        "near_distance":near,
        "far_distance":far,
        "yellow_point_count":int(np.count_nonzero(surface_distance<=near)),
        "transition_point_count":int(np.count_nonzero((surface_distance>near)&(surface_distance<far))),
        "blue_point_count":int(np.count_nonzero(surface_distance>=far)),
    }


def transparent_original(glb: GLB, alpha: float):
    materials = glb.meta.setdefault("materials", [])
    for material in materials:
        pbr = material.setdefault("pbrMetallicRoughness", {})
        factor = list(pbr.get("baseColorFactor", [1,1,1,1]))
        factor[3] = alpha
        pbr["baseColorFactor"] = factor
        material["alphaMode"], material["doubleSided"] = "BLEND", True
    fallback = {"name":"Transparent fallback","pbrMetallicRoughness":{"baseColorFactor":[.72,.78,.86,alpha],"metallicFactor":0,"roughnessFactor":.8},"alphaMode":"BLEND","doubleSided":True}
    fallback_index = len(materials)
    materials.append(fallback)
    for mesh in glb.meta.get("meshes", []):
        for primitive in mesh["primitives"]:
            primitive.setdefault("material", fallback_index)


def append_array(glb: GLB, array: np.ndarray, target: int):
    while len(glb.binary)%4: glb.binary.append(0)
    offset, payload = len(glb.binary), array.tobytes()
    glb.binary.extend(payload)
    glb.meta["bufferViews"].append({"buffer":0,"byteOffset":offset,"byteLength":len(payload),"target":target})
    return len(glb.meta["bufferViews"])-1


def append_marker_mesh(glb: GLB, position, normal, color, index):
    views = [append_array(glb,a,t) for a,t in ((position,34962),(normal,34962),(color,34962),(index,34963))]
    accessors = [
        {"bufferView":views[0],"componentType":5126,"count":len(position),"type":"VEC3","min":position.min(0).astype(float).tolist(),"max":position.max(0).astype(float).tolist()},
        {"bufferView":views[1],"componentType":5126,"count":len(normal),"type":"VEC3"},
        {"bufferView":views[2],"componentType":5121,"normalized":True,"count":len(color),"type":"VEC4"},
        {"bufferView":views[3],"componentType":5125,"count":len(index),"type":"SCALAR","min":[int(index.min())],"max":[int(index.max())]},
    ]
    first = len(glb.meta["accessors"])
    glb.meta["accessors"].extend(accessors)
    material = {"name":"Gaussian surface-distance markers","pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1],"metallicFactor":0,"roughnessFactor":.6},"alphaMode":"OPAQUE","doubleSided":True,"extensions":{"KHR_materials_unlit":{}}}
    glb.meta["materials"].append(material)
    material_index = len(glb.meta["materials"])-1
    glb.meta["meshes"].append({"name":"Gaussian points","primitives":[{"attributes":{"POSITION":first,"NORMAL":first+1,"COLOR_0":first+2},"indices":first+3,"material":material_index}]})
    glb.meta["nodes"].append({"name":"Gaussian points","mesh":len(glb.meta["meshes"])-1})
    glb.meta["scenes"][int(glb.meta.get("scene",0))].setdefault("nodes",[]).append(len(glb.meta["nodes"])-1)
    glb.meta.setdefault("extensionsUsed",[])
    if "KHR_materials_unlit" not in glb.meta["extensionsUsed"]: glb.meta["extensionsUsed"].append("KHR_materials_unlit")
    while len(glb.binary)%4: glb.binary.append(0)
    glb.meta["buffers"][0]["byteLength"] = len(glb.binary)


def write_glb(glb: GLB, path: Path):
    encoded = json.dumps(glb.meta,separators=(",",":"),ensure_ascii=False).encode()
    encoded += b" "*((4-len(encoded)%4)%4)
    binary = bytes(glb.binary)
    total = 12+8+len(encoded)+8+len(binary)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(struct.pack("<4sII",b"glTF",2,total)+struct.pack("<II",len(encoded),JSON_CHUNK)+encoded+struct.pack("<II",len(binary),BIN_CHUNK)+binary)


def parser():
    p=argparse.ArgumentParser(description="Overlay Gaussian points on a transparent original GLB")
    p.add_argument("--version",action="version",version=f"%(prog)s {SCRIPT_VERSION}")
    p.add_argument("--model",required=True,type=Path); p.add_argument("--points",required=True,type=Path); p.add_argument("--output",required=True,type=Path)
    p.add_argument("--alignment",choices=("auto","none"),default="auto"); p.add_argument("--qi-threshold",type=float,default=.3)
    p.add_argument("--model-alpha",type=float,default=.18); p.add_argument("--point-radius",type=float,default=0); p.add_argument("--radius-factor",type=float,default=.4)
    p.add_argument("--radius-q-floor",type=float,default=1.0); p.add_argument("--point-subdivisions",type=int,choices=(0,1,2),default=1)
    p.add_argument("--color-mode",choices=("distance","qi"),default="distance")
    p.add_argument("--near-distance",type=float,default=0.0,help="yellow threshold; 0 selects it automatically")
    p.add_argument("--far-distance",type=float,default=0.0,help="blue threshold; 0 selects it automatically")
    p.add_argument("--outlier-nb-neighbors",type=int,default=20); p.add_argument("--outlier-std-ratio",type=float,default=2)
    p.add_argument("--surface-samples",type=int,default=100000); p.add_argument("--random-seed",type=int,default=7)
    p.add_argument("--icp-iterations",type=int,default=30); p.add_argument("--icp-trim-fraction",type=float,default=.8); p.add_argument("--icp-tolerance",type=float,default=1e-6)
    p.add_argument("--alignment-report",type=Path,default=None)
    return p


def main():
    args=parser().parse_args()
    try:
        if not args.model.is_file() or not args.points.is_file(): raise ValueError("input file missing")
        if not 0<args.model_alpha<1: raise ValueError("model alpha must be between 0 and 1")
        if args.near_distance<0 or args.far_distance<0: raise ValueError("distance thresholds must be non-negative")
        if args.near_distance>0 and args.far_distance>0 and args.far_distance<=args.near_distance: raise ValueError("far distance must exceed near distance")
        cloud=filter_cloud(read_ply(args.points),args.qi_threshold,args.outlier_nb_neighbors,args.outlier_std_ratio)
        glb=read_glb(args.model); triangles=world_triangles(glb); flat=triangles.reshape(-1,3)
        bounds=(flat.min(0),flat.max(0)); reference=sample_surface(triangles,args.surface_samples,args.random_seed)
        print(f"Original GLB: {len(triangles)} triangles, bounds={bounds}")
        if args.alignment=="auto":
            initial,score=initial_alignment(cloud.xyz,reference,args.icp_trim_fraction); print(f"PCA initial distance={score:.6g}")
            transform,metrics=icp(cloud.xyz,reference,initial,args.icp_iterations,args.icp_trim_fraction,args.icp_tolerance)
        else:
            transform=np.eye(4); distance,_=cKDTree(reference).query(cloud.xyz)
            metrics={"iterations":0,"trimmed_rmse":None,"median_nearest_surface_distance":float(np.median(distance)),"p90_nearest_surface_distance":float(np.percentile(distance,90)),"uniform_scale":1.0}
        aligned=apply_transform(cloud.xyz,transform)
        surface_distance,_=cKDTree(reference).query(aligned)
        marker_arrays,color_info=markers(aligned,cloud.qi,surface_distance,args)
        transparent_original(glb,args.model_alpha); append_marker_mesh(glb,*marker_arrays); write_glb(glb,args.output)
        report=args.alignment_report or args.output.with_suffix(".alignment.json"); diagonal=float(np.linalg.norm(bounds[1]-bounds[0]))
        report.write_text(json.dumps({"transform_point_cloud_to_model":transform.tolist(),"metrics":metrics,"model_diagonal":diagonal,"median_distance_fraction_of_diagonal":metrics["median_nearest_surface_distance"]/diagonal,"retained_points":len(cloud.xyz),"marker_coloring":color_info,"surface_distance_summary":{"min":float(surface_distance.min()),"median":float(np.median(surface_distance)),"p90":float(np.percentile(surface_distance,90)),"max":float(surface_distance.max())}},indent=2)+"\n")
        print(f"Transform point cloud -> model:\n{transform}\nSaved: {args.output}\nReport: {report}")
        return 0
    except (OSError,ValueError,KeyError,IndexError) as exc:
        print(f"error: {exc}",file=sys.stderr); return 2


if __name__=="__main__": raise SystemExit(main())
