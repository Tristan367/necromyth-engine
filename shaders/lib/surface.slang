#include "lib/lighting.slang"

static const float kAlphaCutoff = 0.5;

// Baked voxel light, applied on top of the dynamic lighting.
//
// Skylight scales the sun's contribution, so a forest floor or a cave stays dark
// even in full daylight; block light adds a warm term that does not care what
// time it is. A vertex with no baked light reads as (1, 0), which is exactly the
// unlit behaviour everything had before.
float3 applyBakedLight(float3 lit, float2 bakedLight) {
    const float3 kBlockLightTint = float3(1.0, 0.82, 0.55);
    // A floor of a little skylight keeps shaded ground readable rather than pure
    // black -- pitch dark is realistic and unplayable.
    float sky = lerp(0.10, 1.0, bakedLight.x);
    return lit * sky + kBlockLightTint * bakedLight.y;
}

float3 shadeDirectional(float3 worldPos, float3 normal, int filterModeSingle, int filterModeDual) {
    float3 lightDir = normalize(ubo.lightDirection.xyz);
    float ndotl = max(dot(normal, lightDir), 0.0);
    float ambient = ubo.lightColor.w;
    float3 diffuse = ubo.lightColor.rgb * ndotl;
    float shadow = (filterModeDual >= 0)
        ? shadowVisibilityDual(worldPos, normal, lightDir, filterModeDual)
        : shadowVisibilitySingle(worldPos, normal, lightDir, filterModeSingle);
    return ambient + diffuse * shadow;
}

float4 shadeOpaqueSingle(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord, vertIn.textureLayer, vertIn.sampleTextureArray);
    float3 baseColor = texColor.rgb * vertIn.fragColor;
    float3 normal = normalize(vertIn.worldNormal);
    float3 viewDir = normalize(ubo.cameraPosition.xyz - vertIn.worldPos);
    float3 lit = applyBakedLight(shadeDirectional(vertIn.worldPos, normal, filterMode, -1),
                                 vertIn.bakedLight)
               + evaluateAllLights(vertIn.worldPos, normal, viewDir);
    return float4(baseColor * lit, texColor.a);
}

float4 shadeOpaqueDual(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord, vertIn.textureLayer, vertIn.sampleTextureArray);
    float3 baseColor = texColor.rgb * vertIn.fragColor;
    float3 normal = normalize(vertIn.worldNormal);
    float3 viewDir = normalize(ubo.cameraPosition.xyz - vertIn.worldPos);
    float3 lit = applyBakedLight(shadeDirectional(vertIn.worldPos, normal, -1, filterMode),
                                 vertIn.bakedLight)
               + evaluateAllLights(vertIn.worldPos, normal, viewDir);
    return float4(baseColor * lit, texColor.a);
}

float4 shadeCutoutSingle(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord, vertIn.textureLayer, vertIn.sampleTextureArray);
    if (texColor.a < kAlphaCutoff)
        discard;
    return shadeOpaqueSingle(vertIn, filterMode);
}

float4 shadeCutoutDual(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord, vertIn.textureLayer, vertIn.sampleTextureArray);
    if (texColor.a < kAlphaCutoff)
        discard;
    return shadeOpaqueDual(vertIn, filterMode);
}

float4 shadeAlphaToCoverageSingle(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord, vertIn.textureLayer, vertIn.sampleTextureArray);
    if (texColor.a <= 0.0)
        discard;
    return shadeOpaqueSingle(vertIn, filterMode);
}

float4 shadeAlphaToCoverageDual(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord, vertIn.textureLayer, vertIn.sampleTextureArray);
    if (texColor.a <= 0.0)
        discard;
    return shadeOpaqueDual(vertIn, filterMode);
}
