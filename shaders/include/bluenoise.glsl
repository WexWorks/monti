#ifndef BLUENOISE_GLSL
#define BLUENOISE_GLSL

// Blue noise table: 16384 entries (128x128 spatial tile), each uvec4
// (4 packed random values, one per bounce, 4 x 8-bit random bytes per uint).

const uint kBlueNoiseTileSize = 128u;
const uint kBlueNoiseTableSize = 16384u;  // 128 * 128

// Spatial + temporal hash into the blue noise table.
// pixelCoord: screen-space pixel position.
// frameIndex: current frame number for temporal variation.
// Returns: index into the blue noise table [0, 16383].
uint getSpatialHashTemporal(uvec2 pixelCoord, uint frameIndex) {
    uint spatialX = pixelCoord.x & 127u;
    uint spatialY = pixelCoord.y & 127u;
    uint p1 = 73856093u;
    uint p2 = 19349663u;
    uint fullHash32 = (spatialX * p1) ^ (spatialY * p2);
    uint spatialHash = (fullHash32 ^ (fullHash32 >> 14u)) & 16383u;

    // Temporal variation using frame index
    uint temporalHash = (frameIndex * 251u) & 16383u;

    return (spatialHash ^ temporalHash) & 16383u;
}

// Wang hash: decorrelated pseudo-random uint from a seed.
// Used for bounce >= 4 fallback and area light sampling.
uint wangHash(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed ^= seed >> 4u;
    seed *= 0x27d4eb2du;
    seed ^= seed >> 15u;
    return seed;
}

// Produce a decorrelated pseudo-random vec4 in [0,1] from a seed via Wang hash chain.
vec4 wangHashBounceRandoms(uint seed) {
    uint h0 = wangHash(seed);
    uint h1 = wangHash(h0);
    uint h2 = wangHash(h1);
    uint h3 = wangHash(h2);
    return vec4(float(h0), float(h1), float(h2), float(h3)) / 4294967295.0;
}

// Unpack random values from a packed uint32.
// packedBounce: one component of the blue noise table uvec4 entry.
// Returns: vec4 of [0,1] random values.
//   .xy = primary random pair (cosine/diffuse sampling)
//   .zw = secondary random pair (specular/GGX sampling)
vec4 extractBounceRandoms(uint packedBounce) {
    float r0 = float((packedBounce >> 0u) & 0xFFu) / 255.0;
    float r1 = float((packedBounce >> 8u) & 0xFFu) / 255.0;
    float r2 = float((packedBounce >> 16u) & 0xFFu) / 255.0;
    float r3 = float((packedBounce >> 24u) & 0xFFu) / 255.0;

    return vec4(r0, r1, r2, r3);
}

// Get blue noise random values for a specific bounce.
// packed: pre-fetched uvec4 from the blue noise table.
// bounce: bounce index [0..3] maps directly to .x/.y/.z/.w of the packed uvec4.
//         For bounces >= 4, falls back to Wang hash for decorrelated pseudo-random values.
// Returns: vec4 of [0,1] random values for the bounce.
vec4 getBlueNoiseRandom(uvec4 packed, int bounce) {
    if (bounce == 0) return extractBounceRandoms(packed.x);
    if (bounce == 1) return extractBounceRandoms(packed.y);
    if (bounce == 2) return extractBounceRandoms(packed.z);
    if (bounce == 3) return extractBounceRandoms(packed.w);

    // For bounces >= 4, use Wang hash fallback (loses blue noise stratification)
    return wangHashBounceRandoms(packed.x ^ uint(bounce) * 0x9E3779B9u);
}

#endif // BLUENOISE_GLSL
