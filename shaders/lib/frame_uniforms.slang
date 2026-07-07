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
    // Point light cubemap shadow (Sascha omni model). One shadow-casting
    // point light. pointLightView is world->light-local (translate -pos).
    // pointFaceVP[6] = 90° perspective * rotated face views.
    float4x4 pointLightView;      // world -> light-local translation
    float4x4 pointFaceVP[6];      // VP per cube face (perspective * rotation)
    float4 pointLightPos;         // world-space light position (for fragment)
    float4 pointLightParams;      // x=inv_radius, y=range, z=face_size, w=enabled
    // x = split view-space Z (negative); y = blend half-width in view meters (0 = hard pick).
    float4 cascadeParams;
    // x = UV coverage fade width (0 = hard edge)
    float4 shadowFadeWidth;
};
[[vk::binding(0, 0)]]
ConstantBuffer<FrameUniforms> ubo;
