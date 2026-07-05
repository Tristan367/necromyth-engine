// Shared frame UBO (set 0, binding 0). Match FrameUniformBufferObject in uniform_buffer.hpp.
struct FrameUniforms {
    float4x4 view;
    float4x4 proj;
    float4 cameraPosition;
    float4x4 viewSky;
    float4 lightDirection;
    float4 lightColor;
    float4x4 lightViewProj[2];     // 0-1 = directional cascades
    float4x4 spotLightViewProj[4]; // 2-5 = spot lights
    // Point light dual-paraboloid shadow (Godot omni model). One shadow-casting
    // point light supported. pointLightView is the world->light-local rigid
    // transform (no projection); depth stored as normalized radial distance.
    float4x4 pointLightView;      // world -> light-local (Godot shadow_matrix)
    float4 pointLightParams;      // x = inv_radius (1/range), yz = flip_offset(uv), w = enabled
    float4 pointLightAtlasRect;   // xy = uv offset, zw = uv scale (full DP region)
    // x = split view-space Z (negative); y = blend half-width in view meters (0 = hard pick).
    float4 cascadeParams;
    // x = UV coverage fade width (0 = hard edge)
    float4 shadowFadeWidth;
};
[[vk::binding(0, 0)]]
ConstantBuffer<FrameUniforms> ubo;
