// Shared frame UBO (set 0, binding 0). Match FrameUniformBufferObject in uniform_buffer.hpp.
struct FrameUniforms {
    float4x4 view;
    float4x4 proj;
    float4 cameraPosition;
    float4x4 viewSky;
    float4 lightDirection;
    float4 lightColor;
    float4x4 lightViewProj;
};
[[vk::binding(0, 0)]]
ConstantBuffer<FrameUniforms> ubo;
