#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : enable

#include "include/vertex.glsl"
#include "include/payload.glsl"

layout(location = 0) rayPayloadInEXT HitPayload payload;

// ── Descriptor bindings used by closest hit ──────────────────────
// Binding 8: Mesh address table (scalar layout, MeshAddressEntry[])
layout(set = 0, binding = 8, scalar) readonly buffer MeshAddressTable {
    MeshAddressEntry entries[];
} mesh_address_table;

// ── Hit attributes ───────────────────────────────────────────────
hitAttributeEXT vec2 hit_attribs;

// ── Instance custom index encoding ──────────────────────────────
const uint kMeshAddrIndexBits = 12u;
const uint kMeshAddrIndexMask = (1u << kMeshAddrIndexBits) - 1u;

// ── Entry point ──────────────────────────────────────────────────
void main() {
    // Decode instance custom index: lower 12 bits = mesh address index,
    // upper bits = material index
    uint custom_index = gl_InstanceCustomIndexEXT;
    uint mesh_addr_index = custom_index & kMeshAddrIndexMask;
    uint material_index = (custom_index >> kMeshAddrIndexBits) & kMeshAddrIndexMask;

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

    // Populate payload — geometry only, no material fetch or texture sampling
    payload.hit_pos = hit_pos;
    payload.hit_t = gl_HitTEXT;
    payload.normal = world_normal;
    payload.material_index = material_index;
    payload.uv = uv;
    payload.missed = false;
    payload.tangent = world_tangent;
    payload.tangent_w = v0.tangent.w;  // Handedness sign (same for all vertices in a triangle)
}
