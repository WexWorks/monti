"""Analyze AbandonedWarehouse and Brutalism glTF scene structures."""
import json
import sys


def analyze_gltf(path: str, label: str):
    print(f"\n{'='*80}")
    print(f"  {label}: {path}")
    print(f"{'='*80}")

    with open(path, 'r') as f:
        data = json.load(f)

    # Basic counts
    nodes = data.get('nodes', [])
    meshes = data.get('meshes', [])
    materials = data.get('materials', [])
    scenes = data.get('scenes', [])
    accessors = data.get('accessors', [])

    print(f"\nCounts: {len(nodes)} nodes, {len(meshes)} meshes, "
          f"{len(materials)} materials, {len(scenes)} scenes")

    # Scene root nodes
    default_scene = data.get('scene', 0)
    scene = scenes[default_scene]
    root_nodes = scene.get('nodes', [])
    print(f"\nDefault scene '{scene.get('name','unnamed')}' has "
          f"{len(root_nodes)} root nodes: {root_nodes}")

    # Build parent map and child tree
    children_of = {}
    for i, n in enumerate(nodes):
        children_of[i] = n.get('children', [])

    # Walk hierarchy from scene roots and count mesh nodes
    visited = set()
    mesh_nodes = []
    max_depth = [0]

    def walk(nidx, depth):
        if nidx in visited:
            return
        visited.add(nidx)
        max_depth[0] = max(max_depth[0], depth)
        node = nodes[nidx]
        if 'mesh' in node:
            mesh_nodes.append((nidx, node))
        for c in node.get('children', []):
            walk(c, depth + 1)

    for r in root_nodes:
        walk(r, 0)

    print(f"Traversal: visited {len(visited)}/{len(nodes)} nodes, "
          f"max depth {max_depth[0]}, {len(mesh_nodes)} mesh-bearing nodes")

    # Check for unreachable nodes
    unreachable = set(range(len(nodes))) - visited
    unreachable_with_mesh = [i for i in unreachable if 'mesh' in nodes[i]]
    print(f"Unreachable nodes: {len(unreachable)} total, "
          f"{len(unreachable_with_mesh)} with meshes")

    # Count unique meshes referenced
    referenced_meshes = set()
    for nidx, node in mesh_nodes:
        referenced_meshes.add(node['mesh'])
    print(f"Unique meshes referenced by reachable nodes: "
          f"{len(referenced_meshes)}/{len(meshes)}")

    # Mesh instance counts
    mesh_instance_count = {}
    for nidx, node in mesh_nodes:
        mid = node['mesh']
        mesh_instance_count[mid] = mesh_instance_count.get(mid, 0) + 1

    # Show meshes with many instances
    print("\nMesh instance counts (>1):")
    for mid, count in sorted(mesh_instance_count.items(),
                             key=lambda x: -x[1]):
        if count > 1:
            mname = meshes[mid].get('name', f'mesh_{mid}')
            prims = meshes[mid].get('primitives', [])
            vtx = 0
            for p in prims:
                attrs = p.get('attributes', {})
                if 'POSITION' in attrs:
                    acc = accessors[attrs['POSITION']]
                    vtx += acc.get('count', 0)
            print(f"  mesh {mid} ({mname}): {count} instances, "
                  f"{vtx} vertices/instance")

    # Vertex count summary
    total_vertices = 0
    total_indices = 0
    for nidx, node in mesh_nodes:
        mid = node['mesh']
        for p in meshes[mid].get('primitives', []):
            attrs = p.get('attributes', {})
            if 'POSITION' in attrs:
                total_vertices += accessors[attrs['POSITION']].get('count', 0)
            if 'indices' in p:
                total_indices += accessors[p['indices']].get('count', 0)
    print(f"\nTotal geometry (all instances): "
          f"{total_vertices:,} vertices, {total_indices:,} indices")

    # Alpha modes
    print("\nMaterial alpha modes:")
    alpha_counts = {}
    for i, mat in enumerate(materials):
        mode = mat.get('alphaMode', 'OPAQUE(default)')
        alpha_counts[mode] = alpha_counts.get(mode, 0) + 1
    for mode, count in alpha_counts.items():
        print(f"  {mode}: {count} materials")

    # Materials with MASK or BLEND
    for i, mat in enumerate(materials):
        mode = mat.get('alphaMode', '')
        if mode in ('MASK', 'BLEND'):
            print(f"  Material {i} ({mat.get('name','unnamed')}): "
                  f"alphaMode={mode}, alphaCutoff={mat.get('alphaCutoff', 0.5)}")

    # Materials that are doubleSided with no alphaMode - possible missing MASK
    billboard_suspicious = []
    for i, mat in enumerate(materials):
        if mat.get('doubleSided', False) and 'alphaMode' not in mat:
            pbr = mat.get('pbrMetallicRoughness', {})
            if 'baseColorTexture' in pbr:
                billboard_suspicious.append(i)
    if billboard_suspicious:
        print(f"\n  Double-sided materials with textures but no alphaMode "
              f"(possible missing MASK): {billboard_suspicious}")

    # Scene bounds analysis
    import math

    print("\n--- Scene Bounds Analysis ---")
    all_positions = []
    for nidx, node in mesh_nodes:
        mid = node['mesh']
        for p in meshes[mid].get('primitives', []):
            attrs = p.get('attributes', {})
            if 'POSITION' in attrs:
                acc = accessors[attrs['POSITION']]
                if 'min' in acc and 'max' in acc:
                    # Get world transform for this node
                    # For simplicity, just report the accessor-local bounds
                    bmin = acc['min']
                    bmax = acc['max']
                    all_positions.append((nidx, mid, bmin, bmax))

    if all_positions:
        # Compute world-space bounds (simplified - just local for now)
        global_min = [float('inf')] * 3
        global_max = [float('-inf')] * 3
        for nidx, mid, bmin, bmax in all_positions:
            for d in range(3):
                global_min[d] = min(global_min[d], bmin[d])
                global_max[d] = max(global_max[d], bmax[d])
        extent = [global_max[d] - global_min[d] for d in range(3)]
        print(f"  Local-space mesh bounds: "
              f"min={[f'{v:.2f}' for v in global_min]}, "
              f"max={[f'{v:.2f}' for v in global_max]}")
        print(f"  Extent: {[f'{v:.2f}' for v in extent]}")

    # Node transform analysis - find outliers
    print("\n--- Node Transform Outliers ---")
    transforms = []
    for nidx, node in mesh_nodes:
        t = node.get('translation', [0, 0, 0])
        s = node.get('scale', [1, 1, 1])
        name = node.get('name', f'node_{nidx}')
        dist = math.sqrt(sum(x*x for x in t))
        max_scale = max(abs(x) for x in s)
        transforms.append((nidx, name, t, s, dist, max_scale))

    # Sort by distance from origin
    transforms.sort(key=lambda x: -x[4])
    print("  Top 10 mesh nodes by distance from origin:")
    for nidx, name, t, s, dist, max_scale in transforms[:10]:
        mid = nodes[nidx]['mesh']
        mname = meshes[mid].get('name', f'mesh_{mid}')
        print(f"  Node {nidx} ({name}): "
              f"translation={[f'{v:.1f}' for v in t]}, "
              f"scale={[f'{v:.2f}' for v in s]}, "
              f"dist={dist:.1f}, mesh={mname}")

    # Look for parent transforms that could cause issues
    print("\n--- Root/Group Node Transforms ---")
    for r in root_nodes:
        node = nodes[r]
        name = node.get('name', f'node_{r}')
        t = node.get('translation', [0, 0, 0])
        s = node.get('scale', [1, 1, 1])
        r_val = node.get('rotation', [0, 0, 0, 1])
        print(f"  Root {r} ({name}): t={t}, s={s}, r={r_val}")
        for c in node.get('children', []):
            cn = nodes[c]
            cname = cn.get('name', f'node_{c}')
            ct = cn.get('translation', [0, 0, 0])
            cs = cn.get('scale', [1, 1, 1])
            cr = cn.get('rotation', [0, 0, 0, 1])
            print(f"    Child {c} ({cname}): t={ct}, s={cs}, r={cr}")
            for cc in cn.get('children', []):
                ccn = nodes[cc]
                ccname = ccn.get('name', f'node_{cc}')
                cct = ccn.get('translation', [0, 0, 0])
                ccs = ccn.get('scale', [1, 1, 1])
                ccr = ccn.get('rotation', [0, 0, 0, 1])
                print(f"      Grandchild {cc} ({ccname}): "
                      f"t={cct}, s={ccs}, r={ccr}")

    # Extensions used
    print(f"\nExtensions used: {data.get('extensionsUsed', [])}")
    print(f"Extensions required: {data.get('extensionsRequired', [])}")


# Analyze both scenes
analyze_gltf(
    'scenes/extended/Cauldron-Media/AbandonedWarehouse/AbandonedWarehouse.gltf',
    'AbandonedWarehouse')

analyze_gltf(
    'scenes/extended/Cauldron-Media/Brutalism/BrutalistHall.gltf',
    'Brutalism')
