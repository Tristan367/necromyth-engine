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
Sampler2DShadow spotShadowAtlas;

struct GpuPointLight {
    float4 posRange;
    float4 colorIntensity;
};

struct GpuSpotLight {
    float4 posRange;
    float4 colorIntensity;
    float4 dirPad;
    float4 angles;
    float4x4 shadowMatrix;
    float4 atlasRect; // xy = UV offset in atlas, zw = UV scale
};

[[vk::binding(3, 0)]] ByteAddressBuffer lightBuffer : register(t3);

float3 calcPointLight(GpuPointLight l, float3 fragPos, float3 normal, float3 viewDir) {
    float3 toLight = l.posRange.xyz - fragPos;
    float dist = length(toLight);
    float3 lightDir = toLight / max(dist, 0.0001);

    float atten = 1.0 / (1.0 + dist * dist / max(l.posRange.w * l.posRange.w, 0.0001));
    if (l.posRange.w <= 0.0) atten = 1.0 / max(dist * dist, 0.0001);

    float ndotl = max(dot(normal, lightDir), 0.0);
    float3 diffuse = l.colorIntensity.rgb * l.colorIntensity.w * ndotl;

    float3 halfVec = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfVec), 0.0), 32.0);
    float3 specular = l.colorIntensity.rgb * l.colorIntensity.w * spec * 0.3;

    return (diffuse + specular) * atten;
}

float sampleSpotShadow(float3 worldPos, float4x4 shadowMatrix, float4 atlasRect,
                        float3 lightDir, float3 normal, float bias) {
    float4 pos = mul(shadowMatrix, float4(worldPos.xyz - normal * 0.001, 1.0));
    float3 proj = pos.xyz / max(pos.w, 0.0001);
    float2 atlasUV = proj.xy * atlasRect.zw + atlasRect.xy;
    float lightDepth = proj.z;

    // Simple PCF 3x3 on the spot shadow atlas (set 0 binding 4)
    float shadow = 0.0;
    float texelSize = atlasRect.z * 0.0005;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float2 uv = atlasUV + float2(x, y) * texelSize;
            shadow += spotShadowAtlas.SampleCmpLevelZero(uv, lightDepth - bias);
        }
    }
    return shadow / 9.0;
}

float3 calcSpotLightWithShadow(GpuSpotLight l, float3 fragPos, float3 normal, float3 viewDir) {
    return calcSpotLight(l, fragPos, normal, viewDir);
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
        float bias = max(0.0005 * (1.0 - ndotl), 0.0001);
        shadowVis = sampleSpotShadow(fragPos, l.shadowMatrix, l.atlasRect, lightDir, normal, bias);
    }

    return (diffuse + specular) * atten * shadowVis;
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
        contrib += calcPointLight(l, fragPos, normal, viewDir);
        offset += 32;
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
