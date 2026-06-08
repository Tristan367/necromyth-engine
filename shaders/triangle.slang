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
    float4 shadowCoord;
};

static const float4x4 biasMatrix = float4x4(
    0.5, 0.0, 0.0, 0.5,
    0.0, 0.5, 0.0, 0.5,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0);

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    float4 worldPos = mul(pushConstants.model, float4(input.inPosition, 1.0));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));

    output.worldNormal = normalize(mul((float3x3)pushConstants.model, input.inNormal));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    output.shadowCoord = mul(biasMatrix, mul(ubo.lightViewProj, mul(pushConstants.model, float4(input.inPosition, 1.0))));
    return output;
}

Sampler2D texture;
Sampler2DArray textureArray;
Sampler2D shadowMap;

float sampleShadow(float4 shadowCoord) {
    float3 projection = shadowCoord.xyz / shadowCoord.w;
    if (projection.x < 0.0 || projection.x > 1.0 || projection.y < 0.0 || projection.y > 1.0 ||
        projection.z < 0.0 || projection.z > 1.0)
        return 1.0;

    const float scale = 1.5;
    const float dx = scale / 2048.0;
    const float dy = scale / 2048.0;

    float accum = 0.0;
    int count = 0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float2 offset = float2(dx * x, dy * y);
            float depth = shadowMap.Sample(float2(projection.xy + offset)).r;
            accum += depth < projection.z ? 0.0 : 1.0;
            count++;
        }
    }
    return accum / count;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target {
    float3 baseColor;
    if (pushConstants.sampleTextureArray != 0)
        baseColor = textureArray.Sample(float3(vertIn.fragTexCoord, pushConstants.textureArrayLayer)).rgb;
    else
        baseColor = texture.Sample(vertIn.fragTexCoord).rgb;

    baseColor *= vertIn.fragColor;

    // direction_toward_light points at the sun; surfaces facing that direction get diffuse.
    float3 lightDir = normalize(ubo.lightDirection.xyz);
    float ndotl = max(dot(normalize(vertIn.worldNormal), lightDir), 0.0);
    float ambient = ubo.lightColor.w;
    float3 diffuse = ubo.lightColor.rgb * ndotl;
    float shadow = sampleShadow(vertIn.shadowCoord);
    float3 lit = baseColor * (ambient + diffuse * shadow);
    return float4(lit, 1.0);
}
