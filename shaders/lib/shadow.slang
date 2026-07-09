[[vk::binding(2, 0)]]
Sampler2DArray shadowMap;

static const float4x4 biasMatrix = float4x4(
    0.5, 0.0, 0.0, 0.5,
    0.0, 0.5, 0.0, 0.5,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0);

static const float kMinDepthBias = 0.00015;
static const float kSlopeScale = 0.0025;
static const float kNormalOffsetScale = 0.0012;
// Receiver-side bias (main pass). Caster polygon offset: k_shadow_depth_bias_* in shadow_utils.hpp.

float shadowDepthBias(float cosTheta) {
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    return max(kMinDepthBias, kSlopeScale * sinTheta);
}

float textureProj(float3 projection, uint layer, float2 offset, float bias) {
    return shadowMap.SampleCmp(float3(projection.xy + offset, float(layer)), projection.z - bias);
}

float filterPCF(float3 projection, uint layer, float bias) {
    uint width, height, layers;
    shadowMap.GetDimensions(width, height, layers);
    const float scale = 0.75;
    const float dx = scale / float(width);
    const float dy = scale / float(height);

    float accum = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y)
            accum += textureProj(projection, layer, float2(dx * x, dy * y), bias);
    }
    return accum / 9.0;
}

float applyCoverageFade(float shadow, float2 projectionXY) {
    const float fadeWidth = ubo.shadowFadeWidth.x;
    if (fadeWidth <= 0.0)
        return shadow;

    float2 distToEdge = min(projectionXY, 1.0 - projectionXY);
    float edge = min(distToEdge.x, distToEdge.y);
    float fade = smoothstep(0.0, fadeWidth, edge);
    return lerp(1.0, shadow, fade);
}

float sampleCascadeShadow(float3 worldPos, float3 normal, float3 lightDir, uint layer, int filterMode) {
    float cosTheta = clamp(dot(normalize(normal), lightDir), 0.0, 1.0);
    float3 receiverPos = worldPos + normalize(normal) * (kNormalOffsetScale * (1.0 - cosTheta));

    float4 shadowCoord = mul(biasMatrix, mul(ubo.lightViewProj[layer], float4(receiverPos, 1.0)));
    if (shadowCoord.w <= 0.0) return 1.0;
    float3 projection = shadowCoord.xyz / shadowCoord.w;
    if (projection.x < 0.0 || projection.x > 1.0 || projection.y < 0.0 || projection.y > 1.0 ||
        projection.z < 0.0 || projection.z > 1.0)
        return 1.0;

    float bias = shadowDepthBias(cosTheta);
    float shadow = filterMode == 1
        ? filterPCF(projection, layer, bias)
        : textureProj(projection, layer, float2(0.0, 0.0), bias);
    return applyCoverageFade(shadow, projection.xy);
}

float shadowVisibilitySingle(float3 worldPos, float3 normal, float3 lightDir, int filterMode) {
    return sampleCascadeShadow(worldPos, normal, lightDir, 0, filterMode);
}

float shadowVisibilityDual(float3 worldPos, float3 normal, float3 lightDir, int filterMode) {
    const float viewZ = (mul(ubo.view, float4(worldPos, 1.0))).z;
    const float split = ubo.cascadeParams.x;
    const float blendRange = ubo.cascadeParams.y;
    const uint cascadeNear = 0u;
    const uint cascadeFar = 1u;
    const uint primary = viewZ < split ? cascadeFar : cascadeNear;

    if (blendRange <= 0.0)
        return sampleCascadeShadow(worldPos, normal, lightDir, primary, filterMode);

    const float distFromSplit = viewZ - split;
    if (abs(distFromSplit) >= blendRange)
        return sampleCascadeShadow(worldPos, normal, lightDir, primary, filterMode);

    const float shadowNear = sampleCascadeShadow(worldPos, normal, lightDir, cascadeNear, filterMode);
    const float shadowFar = sampleCascadeShadow(worldPos, normal, lightDir, cascadeFar, filterMode);
    const float blend = smoothstep(split - blendRange, split + blendRange, viewZ);
    return lerp(shadowFar, shadowNear, blend);
}
