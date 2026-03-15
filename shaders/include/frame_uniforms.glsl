// Frame-constant uniform buffer — matches C++ FrameUniforms struct (std140).
layout(std140, set = 0, binding = 16) uniform FrameUniformsBlock {
    mat4  inv_view;
    mat4  inv_proj;
    mat4  prev_view_proj;

    uint  env_width;
    uint  env_height;
    float env_avg_luminance;
    float env_max_luminance;

    float env_rotation;
    float skybox_mip_level;
    float jitter_x;
    float jitter_y;

    uint  area_light_count;
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
} frame;
