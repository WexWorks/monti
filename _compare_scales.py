"""Quick comparison of instance transform scales across scenes."""
import json, os, math

SCENE_DIR = r"C:\Users\wex\src\WexWorks\monti\scenes\extended\Cauldron-Media"

for scene_name, gltf_name in [
    ("BistroInterior", "scene.gltf"),
    ("AbandonedWarehouse", "AbandonedWarehouse.gltf"),
    ("Brutalism", "BrutalistHall.gltf"),
]:
    path = os.path.join(SCENE_DIR, scene_name, gltf_name)
    if not os.path.exists(path):
        print(f"{scene_name}: NOT FOUND")
        continue
    with open(path) as f:
        gltf = json.load(f)

    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    root_nodes = scenes[0]["nodes"] if scenes else []

    def get_local_matrix(node):
        if "matrix" in node:
            m = node["matrix"]
            return [[m[0],m[4],m[8],m[12]], [m[1],m[5],m[9],m[13]], 
                    [m[2],m[6],m[10],m[14]], [m[3],m[7],m[11],m[15]]]
        t = node.get("translation", [0,0,0])
        r = node.get("rotation", [0,0,0,1])
        s = node.get("scale", [1,1,1])
        qx,qy,qz,qw = r
        mat = [[0]*4 for _ in range(4)]
        mat[0][0] = (1-2*(qy*qy+qz*qz))*s[0]; mat[0][1] = (2*(qx*qy+qw*qz))*s[1]; mat[0][2] = (2*(qx*qz-qw*qy))*s[2]; mat[0][3] = t[0]
        mat[1][0] = (2*(qx*qy-qw*qz))*s[0]; mat[1][1] = (1-2*(qx*qx+qz*qz))*s[1]; mat[1][2] = (2*(qy*qz+qw*qx))*s[2]; mat[1][3] = t[1]
        mat[2][0] = (2*(qx*qz+qw*qy))*s[0]; mat[2][1] = (2*(qy*qz-qw*qx))*s[1]; mat[2][2] = (1-2*(qx*qx+qy*qy))*s[2]; mat[2][3] = t[2]
        mat[3][0] = 0; mat[3][1] = 0; mat[3][2] = 0; mat[3][3] = 1
        return mat

    def mat_mul(a, b):
        c = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    c[i][j] += a[i][k] * b[k][j]
        return c

    def extract_scale(m):
        sx = math.sqrt(m[0][0]**2 + m[1][0]**2 + m[2][0]**2)
        sy = math.sqrt(m[0][1]**2 + m[1][1]**2 + m[2][1]**2)
        sz = math.sqrt(m[0][2]**2 + m[1][2]**2 + m[2][2]**2)
        return sx, sy, sz

    identity = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    
    world_scales = []
    local_diags = []
    accessors = gltf.get("accessors", [])
    meshes = gltf.get("meshes", [])
    
    world_transforms = {}
    def compute_world(ni, parent_world):
        local = get_local_matrix(nodes[ni])
        world = mat_mul(parent_world, local)
        world_transforms[ni] = world
        for ci in nodes[ni].get("children", []):
            compute_world(ci, world)
    
    for ri in root_nodes:
        compute_world(ri, identity)
    
    mesh_node_count = 0
    for ni, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        mesh_node_count += 1
        if ni in world_transforms:
            sx, sy, sz = extract_scale(world_transforms[ni])
            world_scales.append((sx+sy+sz)/3)
        
        mi = node["mesh"]
        mesh = meshes[mi]
        for prim in mesh.get("primitives", []):
            pos_idx = prim.get("attributes", {}).get("POSITION")
            if pos_idx is not None:
                acc = accessors[pos_idx]
                vmin = acc.get("min", [0,0,0])
                vmax = acc.get("max", [0,0,0])
                diag = math.sqrt(sum((vmax[i]-vmin[i])**2 for i in range(3)))
                local_diags.append(diag)
    
    print(f"\n{scene_name}: {mesh_node_count} mesh nodes, {len(root_nodes)} roots")
    if world_scales:
        print(f"  World scale:  min={min(world_scales):.6f} max={max(world_scales):.6f} avg={sum(world_scales)/len(world_scales):.6f}")
    if local_diags:
        print(f"  Local diag:   min={min(local_diags):.1f} max={max(local_diags):.1f} avg={sum(local_diags)/len(local_diags):.1f}")
        world_diags = [s*d for s,d in zip(world_scales, local_diags)] if len(world_scales)==len(local_diags) else []
        if world_diags:
            print(f"  World diag:   min={min(world_diags):.4f} max={max(world_diags):.4f} avg={sum(world_diags)/len(world_diags):.4f}")
    
    # Show root node transforms
    for ri in root_nodes[:3]:
        n = nodes[ri]
        name = n.get("name", f"node_{ri}")
        s = n.get("scale", [1,1,1])
        r = n.get("rotation", [0,0,0,1])
        t = n.get("translation", [0,0,0])
        print(f"  Root {ri} '{name}': T={[round(x,4) for x in t]} R={[round(x,4) for x in r]} S={[round(x,4) for x in s]}")
