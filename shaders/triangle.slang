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

static const int kPoissonSampleCount = 16;
static const float2 kPoissonDisk16[kPoissonSampleCount] = {
    float2(-0.94201624, -0.39906216),
    float2(0.94558609, -0.76890725),
    float2(-0.09418401, -0.92938870),
    float2(0.34495938, 0.29387760),
    float2(-0.91588581, 0.45771432),
    float2(-0.81544232, -0.87912464),
    float2(-0.38277543, 0.27676845),
    float2(0.97484398, 0.75648379),
    float2(0.44323325, -0.97511554),
    float2(0.53742981, -0.47373420),
    float2(-0.26496911, -0.41893023),
    float2(0.79197514, 0.19090188),
    float2(-0.24188840, 0.99706507),
    float2(-0.81409955, 0.91437590),
    float2(0.19984126, 0.78641367),
    float2(0.14383161, -0.14100790),
};

float poissonRotation(float2 uv) {
    return 6.2831853 * frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
}

float2 rotatePoissonOffset(float2 offset, float angle) {
    const float s = sin(angle);
    const float c = cos(angle);
    return float2(offset.x * c - offset.y * s, offset.x * s + offset.y * c);
}

float filterPoisson(float3 projection, float bias) {
    int2 texDim;
    shadowMap.GetDimensions(texDim.x, texDim.y);
    const float scale = 2.5;
    const float2 texel = float2(scale / float(texDim.x), scale / float(texDim.y));
    const float angle = poissonRotation(projection.xy);

    float accum = 0.0;
    for (int i = 0; i < kPoissonSampleCount; ++i) {
        const float2 offset = rotatePoissonOffset(kPoissonDisk16[i], angle) * texel;
        accum += textureProj(projection, offset, bias);
    }
    return accum / float(kPoissonSampleCount);
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
    const int filterMode = int(ubo.shadowParams.x + 0.5);
    if (filterMode == 2)
        return filterPoisson(projection, bias);
    if (filterMode == 1)
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
