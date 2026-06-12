[[vk::binding(2, 0)]]
Sampler2D shadowMap;

static const float4x4 biasMatrix = float4x4(
    0.5, 0.0, 0.0, 0.5,
    0.0, 0.5, 0.0, 0.5,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0);

static const float kMinDepthBias = 0.00015;
static const float kSlopeScale = 0.0025;
static const float kNormalOffsetScale = 0.0012;

float shadowDepthBias(float cosTheta) {
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    return max(kMinDepthBias, kSlopeScale * sinTheta);
}

float textureProj(float3 projection, float2 offset, float bias) {
    return shadowMap.SampleCmp(projection.xy + offset, projection.z - bias);
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

float applyCoverageFade(float shadow, float2 projectionXY) {
    if (ubo.shadowFadeParams.y <= 0.5)
        return shadow;

    float2 distToEdge = min(projectionXY, 1.0 - projectionXY);
    float edge = min(distToEdge.x, distToEdge.y);
    float fade = smoothstep(0.0, ubo.shadowFadeParams.x, edge);
    return lerp(1.0, shadow, fade);
}

float shadowVisibility(float3 worldPos, float3 normal, float3 lightDir, int filterMode) {
    float cosTheta = clamp(dot(normalize(normal), lightDir), 0.0, 1.0);
    float3 receiverPos = worldPos + normalize(normal) * (kNormalOffsetScale * (1.0 - cosTheta));

    float4 shadowCoord = mul(biasMatrix, mul(ubo.lightViewProj, float4(receiverPos, 1.0)));
    float3 projection = shadowCoord.xyz / shadowCoord.w;
    if (projection.x < 0.0 || projection.x > 1.0 || projection.y < 0.0 || projection.y > 1.0 ||
        projection.z < 0.0 || projection.z > 1.0)
        return 1.0;

    float bias = shadowDepthBias(cosTheta);
    float shadow = filterMode == 1
        ? filterPCF(projection, bias)
        : textureProj(projection, float2(0.0, 0.0), bias);
    return applyCoverageFade(shadow, projection.xy);
}
