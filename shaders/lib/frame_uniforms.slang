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
    // x = split view-space Z (negative); y = blend half-width in view meters (0 = hard pick).
    float4 cascadeParams;
    // x = UV coverage fade width (0 = hard edge)
    float4 shadowFadeWidth;
};
[[vk::binding(0, 0)]]
ConstantBuffer<FrameUniforms> ubo;
