// Point + spot light evaluation with shadow atlas support (Godot-style).
// Lights stored in SSBO at set 0 binding 3.
// Spot shadow depth atlas sampled at set 0 binding 4.
//
// GPU layout (16-byte aligned):
//   PointLight: vec4(pos.xyz, range), vec4(color.rgb, intensity)
//   SpotLight:  vec4(pos.xyz, range), vec4(color.rgb, intensity),
//               vec4(dir.xyz, pad),   vec4(inner, outer, 0, 0),
//               mat4 shadowMatrix,    vec4(uv_offset.xy, uv_scale.xy)
// Buffer: [uvec4(numPoint, numSpot, 0, 0)][PointLight...][SpotLight...]

// Spot shadow atlas (set 0 binding 4, shared by all spot lights)
[[vk::binding(4, 0)]]
Texture2D spotShadowAtlas;
[[vk::binding(4, 0)]]
SamplerState spotShadowSampler;

// Point light cubemap array (set 0 binding 5, Samper oen:omni model)
[[vk::binding(5, 0)]]
TextureCubeArray pointShadowCube;
[[vk::binding(5, 0)]]
SamplerState pointShadowCubeSampler;

struct GpuPointLight {
    float4 posRange;
    float4 colorIntensity;
    float4x4 shadowMatrix;
    float4 atlasRect;
};

struct GpuSpotLight {
    float4 posRange;
    float4 colorIntensity;
    float4 dirPad;
    float4 angles;
    float4x4 shadowMatrix;
    float4 atlasRect;
};

[[vk::binding(3, 0)]] ByteAddressBuffer lightBuffer : register(t3);

float3 calcPointLight(GpuPointLight l, float3 fragPos, float3 normal, float3 viewDir) {
    float3 toLight = l.posRange.xyz - fragPos;
    float dist = length(toLight);
    float3 lightDir = toLight / max(dist, 0.0001);

    float atten = 1.0 / (1.0 + dist * dist / max(l.posRange.w * l.posRange.w, 0.0001));
    if (l.posRange.w <= 0.0) atten = 1.0 / max(dist * dist, 0.0001);

    // Range fade: smooth cutoff near max range (last 50%), matching spot lights.
    if (l.posRange.w > 0.0)
        atten *= 1.0 - smoothstep(l.posRange.w * 0.5, l.posRange.w, dist);

    float ndotl = max(dot(normal, lightDir), 0.0);
    float3 diffuse = l.colorIntensity.rgb * l.colorIntensity.w * ndotl;

    float3 halfVec = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfVec), 0.0), 32.0);
    float3 specular = l.colorIntensity.rgb * l.colorIntensity.w * spec * 0.3;

    float lightContrib = (diffuse.r + diffuse.g + diffuse.b + specular.r + specular.g + specular.b) * atten;
    if (lightContrib <= 0.0001) return float3(0, 0, 0);

    // Point shadow: cubemap array (Sascha omni model). Enabled when invRadius > 0.
    float shadowVis = 1.0;
    if (ubo.pointLightParams.x > 0.0) {
        shadowVis = samplePointShadowCubemap(fragPos, normal, ndotl);
    }

    return min((diffuse + specular) * atten * shadowVis, l.colorIntensity.rgb * l.colorIntensity.w);
}

// Sascha Willems omni shadow: sample cubemap array, manual distance comparison.
// Each face stores the distance from light to occluder (R32F).
// Compare actual distance against stored + epsilon (absolute world units).
float samplePointShadowCubemap(float3 worldPos, float3 normal, float ndotl) {
    float3 lightPos = ubo.pointLightPos.xyz;

    float3 lightVec = worldPos - lightPos;
    float dist = length(lightVec);

    // Normal bias: offset sample direction along surface normal
    float3 localNormal = normalize(mul((float3x3)ubo.pointLightView, normal));
    float3 sampleDir = normalize(lightVec);
    float ndotd = abs(dot(localNormal, sampleDir));
    float nBias = 0.02 * (1.0 - ndotd);
    sampleDir = normalize(sampleDir + localNormal * nBias);

    // Sample cubemap array layer 0 (first shadow-casting point light)
    float storedDist = pointShadowCube.Sample(pointShadowCubeSampler, float4(sampleDir, 0.0)).r;
    const float kEpsilon = 0.1; // absolute world-space bias (Sascha: 0.15)

    return dist <= storedDist + kEpsilon ? 1.0 : 0.0;
}

float sampleSpotShadow(float3 worldPos, float4x4 shadowMatrix, float4 atlasRect,
                        float3 lightDir, float3 normal, float bias) {
    float4 pos = mul(shadowMatrix, float4(worldPos.xyz - normal * 0.001, 1.0));
    // Reject fragments behind the light's projection plane (w <= 0) — otherwise
    // the perspective divide flips coordinates and smears the wrong face across
    // the surface ("paint splatter").
    if (pos.w <= 0.0)
        return 1.0;
    float3 proj = pos.xyz / pos.w;

    // Face-local UV must be inside [0,1]; a 90° frustum only covers its own
    // face. Outside that (or outside the valid depth range) the fragment is
    // not seen by this face -> treat as lit. Mirrors the directional path.
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 ||
        proj.z < 0.0 || proj.z > 1.0)
        return 1.0;

    float lightDepth = proj.z;

    uint atlasW, atlasH;
    spotShadowAtlas.GetDimensions(atlasW, atlasH);
    float texelSize = 1.0 / float(atlasW);

    // Clamp the PCF footprint to the interior of this sub-rect so 3x3 taps
    // near a face/light boundary do not bleed into a neighbouring atlas tile.
    float2 rectMin = atlasRect.xy + texelSize;
    float2 rectMax = atlasRect.xy + atlasRect.zw - texelSize;

    float2 atlasUV = proj.xy * atlasRect.zw + atlasRect.xy;

    float shadow = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float2 uv = clamp(atlasUV + float2(x, y) * texelSize, rectMin, rectMax);
            float depth = spotShadowAtlas.SampleLevel(spotShadowSampler, uv, 0).r;
            shadow += (lightDepth - bias) <= depth ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
}

float3 calcSpotLight(GpuSpotLight l, float3 fragPos, float3 normal, float3 viewDir) {
    float3 toLight = l.posRange.xyz - fragPos;
    float dist = length(toLight);
    float3 lightDir = toLight / max(dist, 0.0001);

    float atten = 1.0 / (1.0 + dist * dist / max(l.posRange.w * l.posRange.w, 0.0001));
    if (l.posRange.w <= 0.0) atten = 1.0 / max(dist * dist, 0.0001);

    // Spot cone falloff
    float3 spotDir = normalize(l.dirPad.xyz);
    float cosTheta = dot(-lightDir, spotDir);
    float cosInner = cos(l.angles.x);
    float cosOuter = cos(l.angles.y);
    float spotAtten = smoothstep(cosOuter, cosInner, cosTheta);
    atten *= spotAtten;

    // Range fade: smooth cutoff near max range (last 50%)
    if (l.posRange.w > 0.0)
        atten *= 1.0 - smoothstep(l.posRange.w * 0.5, l.posRange.w, dist);

    float ndotl = max(dot(normal, lightDir), 0.0);
    float3 diffuse = l.colorIntensity.rgb * l.colorIntensity.w * ndotl;

    float3 halfVec = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfVec), 0.0), 32.0);
    float3 specular = l.colorIntensity.rgb * l.colorIntensity.w * spec * 0.3;

    float lightContrib = (diffuse.r + diffuse.g + diffuse.b + specular.r + specular.g + specular.b) * atten;
    if (lightContrib <= 0.0001) return float3(0, 0, 0);

    // Shadow (Godot-style: sample from atlas using sub-rect)
    float shadowVis = 1.0;
    if (l.shadowMatrix[0][0] != 0.0 || l.shadowMatrix[1][1] != 0.0) {
        float bias = max(0.001 * (1.0 - ndotl), 0.0002);
        shadowVis = sampleSpotShadow(fragPos, l.shadowMatrix, l.atlasRect, lightDir, normal, bias);
    }

    return min((diffuse + specular) * atten * shadowVis, l.colorIntensity.rgb * l.colorIntensity.w);
}

float3 evaluatePointLights(float3 fragPos, float3 normal, float3 viewDir) {
    uint numPoint = lightBuffer.Load<uint>(0);
    if (numPoint == 0) return float3(0, 0, 0);

    float3 contrib = float3(0, 0, 0);
    uint offset = 16;
    for (uint i = 0; i < numPoint; ++i) {
        GpuPointLight l;
        l.posRange = asfloat(lightBuffer.Load4(offset));
        l.colorIntensity = asfloat(lightBuffer.Load4(offset + 16));
        // +32 bytes (shadow_matrix + atlas_rect) skipped — point shadow uses the
        // dual-paraboloid UBO fields (pointLightView / pointLightParams / rect).
        contrib += calcPointLight(l, fragPos, normal, viewDir);
        offset += 32 + 64;
    }
    return contrib;
}

float3 evaluateSpotLights(float3 fragPos, float3 normal, float3 viewDir) {
    uint numSpot = lightBuffer.Load<uint>(4);
    if (numSpot == 0) return float3(0, 0, 0);

    float3 contrib = float3(0, 0, 0);
    uint numPoint = lightBuffer.Load<uint>(0);
    uint offset = 16 + numPoint * 32;
    for (uint i = 0; i < numSpot; ++i) {
        GpuSpotLight l;
        l.posRange = asfloat(lightBuffer.Load4(offset));
        l.colorIntensity = asfloat(lightBuffer.Load4(offset + 16));
        l.dirPad = asfloat(lightBuffer.Load4(offset + 32));
        l.angles = asfloat(lightBuffer.Load4(offset + 48));
        l.shadowMatrix[0] = asfloat(lightBuffer.Load4(offset + 64));
        l.shadowMatrix[1] = asfloat(lightBuffer.Load4(offset + 80));
        l.shadowMatrix[2] = asfloat(lightBuffer.Load4(offset + 96));
        l.shadowMatrix[3] = asfloat(lightBuffer.Load4(offset + 112));
        l.atlasRect = asfloat(lightBuffer.Load4(offset + 128));
        contrib += calcSpotLight(l, fragPos, normal, viewDir);
        offset += 144;
    }
    return contrib;
}

float3 evaluateAllLights(float3 fragPos, float3 normal, float3 viewDir) {
    return evaluatePointLights(fragPos, normal, viewDir) +
           evaluateSpotLights(fragPos, normal, viewDir);
}
