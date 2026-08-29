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

// Never fully black: a surface pointing straight away from the sun still shows
// its own colour rather than becoming a silhouette.
static const float kMinDiffuse = 0.2;
// How much light a shadowed surface keeps. Zero would undo the wrap below --
// the shaded half of the world would be flat again, just flat and dark instead
// of flat and lit -- so a shadow darkens rather than erases.
static const float kShadowBrightness = 0.55;

// Wrapped diffuse, in place of clamped Lambert.
//
// Lambert clamps everything more than ninety degrees off the sun to exactly
// zero, so a whole half of the world collapses to one flat value and the shapes
// in it stop being readable: a roofline, a stack of crates and a smooth
// hillside are the same block of colour on their shaded side.
//
// Wrapping spreads the dot product across the whole sphere instead of half of
// it, so a surface pointing away is still lighter than one pointing further
// away and the geometry survives. Spreading it that way costs half the
// contrast, which the cube puts back -- it leaves a face pointed straight at
// the sun untouched and pushes everything else down, so the gradient across a
// shaded wall is worth looking at rather than a wash of near-identical greys.
//
// This is what the C# client does (Voxel.gdshader, `nSky`), and it is most of
// why that game reads as solid shapes rather than as flat regions.
float wrappedDiffuse(float3 normal, float3 lightDir) {
    float wrapped = dot(normal, lightDir) * 0.5 + 0.5;
    wrapped = wrapped * wrapped * wrapped;
    return wrapped * (1.0 - kMinDiffuse) + kMinDiffuse;
}

float3 shadeDirectional(float3 worldPos, float3 normal, int filterModeSingle, int filterModeDual) {
    float3 lightDir = normalize(ubo.lightDirection.xyz);
    float ambient = ubo.lightColor.w;
    float shadow = (filterModeDual >= 0)
        ? shadowVisibilityDual(worldPos, normal, lightDir, filterModeDual)
        : shadowVisibilitySingle(worldPos, normal, lightDir, filterModeSingle);
    float sun = min(shadow + kShadowBrightness, 1.0);
    return ambient + ubo.lightColor.rgb * wrappedDiffuse(normal, lightDir) * sun;
}

// Snow on anything facing up.
//
// The C# lays an actual snow voxel on the topmost solid voxel of every column
// and flips a `cold` shader parameter -- VoxelClient.AddSnow, VoxelServer.AddSnow.
// Doing it in the shader instead means winter arriving and leaving costs nothing,
// touches no saved voxels, and cannot get out of step with the world.
//
// Weighted by how far up the surface faces, so a roof and a lawn go white while
// a wall stays what it is, and by depth under an overhang -- which comes free
// from the baked light, since a surface with no sky above it never gets snowed
// on.
float3 applySnow(float3 color, float3 normal, float sky) {
    float amount = ubo.shadowFadeWidth.z;
    if (amount <= 0.001)
        return color;
    float facing = smoothstep(0.35, 0.85, normal.y);
    float outside = smoothstep(0.25, 0.7, sky);
    const float3 kSnow = float3(0.92, 0.94, 0.98);
    return lerp(color, kSnow, saturate(amount * facing * outside));
}

// What winter does to everything else: the light goes blue and loses a little
// warmth. `cold` in the C#, which sets it on every material at once.
float3 applyCold(float3 color) {
    float amount = ubo.shadowFadeWidth.z;
    if (amount <= 0.001)
        return color;
    float3 chilled = color * float3(0.88, 0.94, 1.06);
    return lerp(color, chilled, saturate(amount));
}

float4 shadeOpaqueSingle(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord, vertIn.textureLayer, vertIn.sampleTextureArray);
    float3 baseColor = texColor.rgb * vertIn.fragColor;
    float3 normal = normalize(vertIn.worldNormal);
    float3 viewDir = normalize(ubo.cameraPosition.xyz - vertIn.worldPos);
    float3 lit = applyBakedLight(shadeDirectional(vertIn.worldPos, normal, filterMode, -1),
                                 vertIn.bakedLight)
               + evaluateAllLights(vertIn.worldPos, normal, viewDir);
    baseColor = applySnow(baseColor, normal, vertIn.bakedLight.x);
    return float4(applyCold(baseColor * lit), texColor.a);
}

float4 shadeOpaqueDual(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord, vertIn.textureLayer, vertIn.sampleTextureArray);
    float3 baseColor = texColor.rgb * vertIn.fragColor;
    float3 normal = normalize(vertIn.worldNormal);
    float3 viewDir = normalize(ubo.cameraPosition.xyz - vertIn.worldPos);
    baseColor = applySnow(baseColor, normal, vertIn.bakedLight.x);
    float3 lit = applyBakedLight(shadeDirectional(vertIn.worldPos, normal, -1, filterMode),
                                 vertIn.bakedLight)
               + evaluateAllLights(vertIn.worldPos, normal, viewDir);
    return float4(applyCold(baseColor * lit), texColor.a);
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
