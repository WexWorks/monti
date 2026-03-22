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

    uint  light_count;
    float env_intensity;
    uint  background_mode;  // 0 = transparent black, 1 = environment map
    uint  _pad2;
} frame;
