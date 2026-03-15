#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : enable

#include "include/vertex.glsl"
#include "include/payload.glsl"
#include "include/sampling.glsl"
#include "include/constants.glsl"

layout(location = 0) rayPayloadInEXT HitPayload payload;

// ── Descriptor bindings used by closest hit ──────────────────────
// Binding 8: Mesh address table (scalar layout, MeshAddressEntry[])
layout(set = 0, binding = 8, scalar) readonly buffer MeshAddressTable {
    MeshAddressEntry entries[];
} mesh_address_table;

// ── Hit attributes ───────────────────────────────────────────────
hitAttributeEXT vec2 hit_attribs;

// ── Entry point ──────────────────────────────────────────────────
void main() {
    // Decode instance custom index: lower 12 bits = mesh address index,
    // upper bits = material index
    uint custom_index = gl_InstanceCustomIndexEXT;
    uint mesh_addr_index = custom_index & kCustomIndexMask;
    uint material_index = (custom_index >> kCustomIndexBits) & kCustomIndexMask;

    // Look up mesh address entry and fetch triangle vertices
    MeshAddressEntry entry = mesh_address_table.entries[mesh_addr_index];
    Vertex v0, v1, v2;
    fetchTriangleVertices(entry, gl_PrimitiveID, v0, v1, v2);

    // Barycentric interpolation
    vec3 bary = vec3(1.0 - hit_attribs.x - hit_attribs.y, hit_attribs.x, hit_attribs.y);
    vec3 object_normal = normalize(v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z);
    vec3 object_tangent = normalize(v0.tangent.xyz * bary.x + v1.tangent.xyz * bary.y + v2.tangent.xyz * bary.z);
    vec2 uv = v0.tex_coord_0 * bary.x + v1.tex_coord_0 * bary.y + v2.tex_coord_0 * bary.z;

    // Transform normal and tangent to world space
    vec3 world_normal = normalize(gl_ObjectToWorldEXT * vec4(object_normal, 0.0));
    vec3 world_tangent = normalize(gl_ObjectToWorldEXT * vec4(object_tangent, 0.0));

    // Compute hit position
    vec3 hit_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;

    // ── Ray cone: compute triangle LOD constant in log-domain ────
    vec3 e1_world = (gl_ObjectToWorldEXT * vec4(v1.position - v0.position, 0.0)).xyz;
    vec3 e2_world = (gl_ObjectToWorldEXT * vec4(v2.position - v0.position, 0.0)).xyz;
    float world_area = length(cross(e1_world, e2_world));  // 2x world-space area

    vec2 uv_e1 = v1.tex_coord_0 - v0.tex_coord_0;
    vec2 uv_e2 = v2.tex_coord_0 - v0.tex_coord_0;
    float uv_area = abs(uv_e1.x * uv_e2.y - uv_e1.y * uv_e2.x);  // 2x UV area

    float tri_lod_constant = 0.5 * safeLog2(uv_area / max(world_area, kMinCosTheta));

    // Populate payload — geometry only, no material fetch or texture sampling
    payload.hit_pos = hit_pos;
    payload.hit_t = gl_HitTEXT;
    payload.normal = world_normal;
    payload.material_index = material_index;
    payload.uv = uv;
    payload.missed = false;
    payload.tangent = world_tangent;
    payload.tangent_w = v0.tangent.w;  // Handedness sign (same for all vertices in a triangle)
    payload.tri_lod_constant = tri_lod_constant;
}
