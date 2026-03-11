#ifndef VERTEX_GLSL
#define VERTEX_GLSL

// Required extensions (must be declared in the including shader file):
// GL_EXT_buffer_reference2, GL_EXT_scalar_block_layout,
// GL_EXT_shader_explicit_arithmetic_types_int64

// ── Vertex struct (matches C++ 56-byte Vertex) ──────────────────
struct Vertex {
    vec3 position;     // offset 0
    vec3 normal;       // offset 12
    vec4 tangent;      // offset 24 (xyz = direction, w = bitangent sign)
    vec2 tex_coord_0;  // offset 40
    vec2 tex_coord_1;  // offset 48
};

// ── Buffer reference layouts for per-mesh vertex/index access ───
layout(buffer_reference, scalar) readonly buffer VertexBufferRef {
    Vertex vertices[];
};

layout(buffer_reference, scalar) readonly buffer IndexBufferRef {
    uint indices[];
};

// ── Mesh address entry (matches C++ 32-byte MeshAddressEntry) ───
struct MeshAddressEntry {
    uint64_t vertex_address;
    uint64_t index_address;
    uint vertex_count;
    uint index_count;
    uint pad_0;
    uint pad_1;
};

// ── Helper: fetch three triangle vertices via buffer_reference ──
void fetchTriangleVertices(MeshAddressEntry entry, uint primitive_id,
                           out Vertex v0, out Vertex v1, out Vertex v2) {
    IndexBufferRef ib = IndexBufferRef(entry.index_address);
    VertexBufferRef vb = VertexBufferRef(entry.vertex_address);
    uint base = primitive_id * 3;
    v0 = vb.vertices[ib.indices[base + 0]];
    v1 = vb.vertices[ib.indices[base + 1]];
    v2 = vb.vertices[ib.indices[base + 2]];
}

#endif // VERTEX_GLSL
