struct VSInput {
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;
};

struct FrameUniforms {
    float4x4 view;
    float4x4 proj;
    float4 cameraPosition;
    float4x4 viewSky;
};
ConstantBuffer<FrameUniforms> ubo;

struct PushConstants {
    float4x4 model;
    uint textureArrayLayer;
    uint sampleTextureArray;
};
[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;

struct VSOutput {
    float4 pos : SV_Position;
    float3 fragColor;
    float2 fragTexCoord;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pos = mul(ubo.proj, mul(ubo.view, mul(pushConstants.model, float4(input.inPosition, 1.0))));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    return output;
}

Sampler2D texture;
Sampler2DArray textureArray;

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target {
    if (pushConstants.sampleTextureArray != 0)
        return textureArray.Sample(float3(vertIn.fragTexCoord, pushConstants.textureArrayLayer));
    return texture.Sample(vertIn.fragTexCoord);
}
