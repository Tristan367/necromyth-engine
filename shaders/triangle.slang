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
    float3 worldNormal;
    float3 fragColor;
    float2 fragTexCoord;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    float4 worldPos = mul(pushConstants.model, float4(input.inPosition, 1.0));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));

    // Demo meshes use uniform scale; a full normal matrix can move into push constants later.
    output.worldNormal = normalize(mul((float3x3)pushConstants.model, input.inNormal));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    return output;
}

Sampler2D texture;
Sampler2DArray textureArray;

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target {
    float3 baseColor;
    if (pushConstants.sampleTextureArray != 0)
        baseColor = textureArray.Sample(float3(vertIn.fragTexCoord, pushConstants.textureArrayLayer)).rgb;
    else
        baseColor = texture.Sample(vertIn.fragTexCoord).rgb;

    baseColor *= vertIn.fragColor;

    float3 lightDir = normalize(ubo.lightDirection.xyz);
    float ndotl = max(dot(normalize(vertIn.worldNormal), lightDir), 0.0);
    float ambient = ubo.lightColor.w;
    float3 diffuse = ubo.lightColor.rgb * ndotl;
    float3 lit = baseColor * (ambient + diffuse);
    return float4(lit, 1.0);
}
