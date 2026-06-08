struct VSInput {
    float3 inPosition;
    float3 inNormal;
    float3 inColor;
    float2 inTexCoord;
};

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

struct PushConstants {
    float4x4 model;
    uint textureArrayLayer;
    uint sampleTextureArray;
};
[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;

[shader("vertex")]
float4 vertMain(VSInput input) : SV_Position {
    return mul(ubo.lightViewProj, mul(pushConstants.model, float4(input.inPosition, 1.0)));
}
