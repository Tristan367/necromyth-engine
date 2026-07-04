#include "lib/lighting.slang"

static const float kAlphaCutoff = 0.5;

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
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord);
    float3 baseColor = texColor.rgb * vertIn.fragColor;
    float3 normal = normalize(vertIn.worldNormal);
    float3 viewDir = normalize(ubo.cameraPosition.xyz - vertIn.worldPos);
    float3 lit = shadeDirectional(vertIn.worldPos, normal, filterMode, -1)
               + evaluateAllLights(vertIn.worldPos, normal, viewDir);
    return float4(baseColor * saturate(lit), texColor.a);
}

float4 shadeOpaqueDual(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord);
    float3 baseColor = texColor.rgb * vertIn.fragColor;
    float3 normal = normalize(vertIn.worldNormal);
    float3 viewDir = normalize(ubo.cameraPosition.xyz - vertIn.worldPos);
    float3 lit = shadeDirectional(vertIn.worldPos, normal, -1, filterMode)
               + evaluateAllLights(vertIn.worldPos, normal, viewDir);
    return float4(baseColor * saturate(lit), texColor.a);
}

float4 shadeCutoutSingle(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord);
    if (texColor.a < kAlphaCutoff)
        discard;
    return shadeOpaqueSingle(vertIn, filterMode);
}

float4 shadeCutoutDual(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord);
    if (texColor.a < kAlphaCutoff)
        discard;
    return shadeOpaqueDual(vertIn, filterMode);
}

float4 shadeAlphaToCoverageSingle(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord);
    if (texColor.a <= 0.0)
        discard;

    float3 baseColor = texColor.rgb * vertIn.fragColor;
    float3 normal = normalize(vertIn.worldNormal);
    float3 viewDir = normalize(ubo.cameraPosition.xyz - vertIn.worldPos);
    float3 lit = shadeDirectional(vertIn.worldPos, normal, filterMode, -1)
               + evaluateAllLights(vertIn.worldPos, normal, viewDir);
    return float4(baseColor * saturate(lit), texColor.a);
}

float4 shadeAlphaToCoverageDual(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord);
    if (texColor.a <= 0.0)
        discard;

    float3 baseColor = texColor.rgb * vertIn.fragColor;
    float3 normal = normalize(vertIn.worldNormal);
    float3 viewDir = normalize(ubo.cameraPosition.xyz - vertIn.worldPos);
    float3 lit = shadeDirectional(vertIn.worldPos, normal, -1, filterMode)
               + evaluateAllLights(vertIn.worldPos, normal, viewDir);
    return float4(baseColor * saturate(lit), texColor.a);
}
