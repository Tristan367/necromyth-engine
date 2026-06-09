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
    float4 shadowParams;
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

struct VSOutput {
    float4 pos : SV_Position;
    float3 worldNormal;
    float3 worldPos;
    float3 fragColor;
    float2 fragTexCoord;
};

static const float4x4 biasMatrix = float4x4(
    0.5, 0.0, 0.0, 0.5,
    0.0, 0.5, 0.0, 0.5,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0);

static const float kMinDepthBias = 0.00015;
static const float kSlopeScale = 0.0025;
static const float kNormalOffsetScale = 0.0012;

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    float4 worldPos = mul(pushConstants.model, float4(input.inPosition, 1.0));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));
    output.worldPos = worldPos.xyz;
    output.worldNormal = normalize(mul((float3x3)pushConstants.model, input.inNormal));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    return output;
}

[[vk::binding(0, 1)]]
Sampler2D texture;
[[vk::binding(1, 0)]]
Sampler2DArray textureArray;
[[vk::binding(2, 0)]]
Sampler2D shadowMap;

float shadowDepthBias(float cosTheta) {
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    return max(kMinDepthBias, kSlopeScale * sinTheta);
}

float textureProj(float3 projection, float2 offset, float bias) {
    float depth = shadowMap.Sample(projection.xy + offset).r;
    return depth < projection.z - bias ? 0.0 : 1.0;
}

float filterPCF(float3 projection, float bias) {
    int2 texDim;
    shadowMap.GetDimensions(texDim.x, texDim.y);
    const float scale = 0.75;
    const float dx = scale / float(texDim.x);
    const float dy = scale / float(texDim.y);

    float accum = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y)
            accum += textureProj(projection, float2(dx * x, dy * y), bias);
    }
    return accum / 9.0;
}

float sampleShadow(float3 worldPos, float3 normal, float3 lightDir) {
    float cosTheta = clamp(dot(normalize(normal), lightDir), 0.0, 1.0);
    float3 receiverPos = worldPos + normalize(normal) * (kNormalOffsetScale * (1.0 - cosTheta));

    float4 shadowCoord = mul(biasMatrix, mul(ubo.lightViewProj, float4(receiverPos, 1.0)));
    float3 projection = shadowCoord.xyz / shadowCoord.w;
    if (projection.x < 0.0 || projection.x > 1.0 || projection.y < 0.0 || projection.y > 1.0 ||
        projection.z < 0.0 || projection.z > 1.0)
        return 1.0;

    float bias = shadowDepthBias(cosTheta);
    if (ubo.shadowParams.x > 0.5)
        return filterPCF(projection, bias);

    return textureProj(projection, float2(0.0, 0.0), bias);
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target {
    float3 baseColor;
    if (pushConstants.sampleTextureArray != 0)
        baseColor = textureArray.Sample(float3(vertIn.fragTexCoord, pushConstants.textureArrayLayer)).rgb;
    else
        baseColor = texture.Sample(vertIn.fragTexCoord).rgb;

    baseColor *= vertIn.fragColor;

    float3 lightDir = normalize(ubo.lightDirection.xyz);
    float3 normal = normalize(vertIn.worldNormal);
    float ndotl = max(dot(normal, lightDir), 0.0);
    float ambient = ubo.lightColor.w;
    float3 diffuse = ubo.lightColor.rgb * ndotl;
    float shadow = sampleShadow(vertIn.worldPos, normal, lightDir);
    float3 lit = baseColor * (ambient + diffuse * shadow);
    return float4(lit, 1.0);
}
