import json, struct

with open('scenes/khronos/GlassHurricaneCandleHolder.glb', 'rb') as f:
    magic, version, length = struct.unpack('<III', f.read(12))
    chunk_length, chunk_type = struct.unpack('<II', f.read(8))
    json_data = json.loads(f.read(chunk_length))

for i, mat in enumerate(json_data.get('materials', [])):
    name = mat.get('name', 'unnamed')
    print(f'=== Material {i}: {name} ===')
    pbr = mat.get('pbrMetallicRoughness', {})
    print(f'  baseColorFactor: {pbr.get("baseColorFactor", "not set")}')
    print(f'  baseColorTexture: {pbr.get("baseColorTexture", "not set")}')
    print(f'  metallicFactor: {pbr.get("metallicFactor", "not set")}')
    print(f'  roughnessFactor: {pbr.get("roughnessFactor", "not set")}')
    exts = mat.get('extensions', {})
    print(f'  extensions: {list(exts.keys())}')
    for ext_name, ext_val in exts.items():
        print(f'    {ext_name}: {json.dumps(ext_val, indent=6)}')
    print(f'  alphaMode: {mat.get("alphaMode", "not set")}')
    print(f'  doubleSided: {mat.get("doubleSided", "not set")}')
    print()

print(f'extensionsUsed: {json_data.get("extensionsUsed", [])}')
