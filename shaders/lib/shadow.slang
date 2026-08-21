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

// Projects a world-space position into a cascade's shadow-map UVZ.
// Returns false if the projection falls outside [0,1] or behind the camera.
bool projectCascadeShadow(float3 worldPos, float3 normal, float3 lightDir, uint layer,
                          out float3 outProjection, out float outCosTheta) {
    // Initialised before any early return: callers are expected to check the
    // return value, but leaving an out parameter undefined makes one forgotten
    // check read garbage instead of a harmless zero.
    outProjection = float3(0.0, 0.0, 0.0);

    float3 n = normalize(normal);
    outCosTheta = clamp(dot(n, lightDir), 0.0, 1.0);
    float3 receiverPos = worldPos + n * (kNormalOffsetScale * (1.0 - outCosTheta));

    float4 shadowCoord = mul(biasMatrix, mul(ubo.lightViewProj[layer], float4(receiverPos, 1.0)));
    if (shadowCoord.w <= 0.0) return false;
    outProjection = shadowCoord.xyz / shadowCoord.w;
    return all(outProjection >= 0.0) && all(outProjection <= 1.0);
}

// Samples the shadow map at a pre-computed projection.
float sampleFromProjection(float3 projection, uint layer, int filterMode, float cosTheta) {
    float bias = shadowDepthBias(cosTheta);
    float shadow = filterMode == 1
        ? filterPCF(projection, layer, bias)
        : textureProj(projection, layer, float2(0.0, 0.0), bias);
    return applyCoverageFade(shadow, projection.xy);
}

float sampleCascadeShadow(float3 worldPos, float3 normal, float3 lightDir, uint layer, int filterMode) {
    float3 projection;
    float cosTheta;
    if (!projectCascadeShadow(worldPos, normal, lightDir, layer, projection, cosTheta))
        return 1.0;
    return sampleFromProjection(projection, layer, filterMode, cosTheta);
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

    // Blend zone: if one cascade doesn't cover this fragment,
    // smoothly transition to the other instead of a hard switch.
    float3 projNear, projFar;
    float cosNear, cosFar;
    bool nearOk = projectCascadeShadow(worldPos, normal, lightDir, cascadeNear, projNear, cosNear);
    bool farOk  = projectCascadeShadow(worldPos, normal, lightDir, cascadeFar,  projFar,  cosFar);

    // Coverage confidence for the near cascade: 0 at UV edge, 1 well inside.
    // This mirrors the existing coverage fade so the transition is invisible.
    float nearConfidence = 0.0;
    if (nearOk) {
        float2 nearDist = min(projNear.xy, 1.0 - projNear.xy);
        float nearEdge = min(nearDist.x, nearDist.y);
        nearConfidence = smoothstep(0.0, 0.05, nearEdge);
    }

    // A cascade that does not cover this fragment must abstain, not vote.
    //
    // Falling back to 1.0 means "fully lit", and 1.0 blended against a real
    // shadow lightens it -- so wherever one cascade ran out of coverage the
    // cross-fade washed the shadow out, leaving a bright band across the ground
    // at the split. Deferring to the other cascade keeps the blend between two
    // real answers, or one real answer and itself.
    const float rawNear = nearOk
        ? sampleFromProjection(projNear, cascadeNear, filterMode, cosNear) : -1.0;
    const float rawFar  = farOk
        ? sampleFromProjection(projFar, cascadeFar, filterMode, cosFar)   : -1.0;

    if (rawNear < 0.0 && rawFar < 0.0)
        return 1.0; // genuinely outside both: nothing can shadow it

    const float shadowNear = rawNear >= 0.0 ? rawNear : rawFar;
    const float shadowFar  = rawFar  >= 0.0 ? rawFar  : rawNear;

    const float blend = smoothstep(split - blendRange, split + blendRange, viewZ);
    const float blendedResult = lerp(shadowFar, shadowNear, blend);

    // Fade from blended (both) to far-only as near coverage drops.
    return lerp(shadowFar, blendedResult, nearConfidence);
}
