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

    // Shadow (dual-paraboloid). For now always run if x=invRadius > 0.
    float shadowVis = 1.0;
    if (ubo.pointLightParams.x > 0.0) {
        shadowVis = samplePointShadowDP(fragPos, normal, ndotl);
    }

    return min((diffuse + specular) * atten * shadowVis, l.colorIntensity.rgb * l.colorIntensity.w);
}

// Godot light_process_omni hard-shadow path (scene_forward_lights_inc.glsl).
// Transforms the world position into light-local space, picks the paraboloid
// hemisphere by the sign of z, paraboloid-projects to atlas UV, and compares
// against the stored NORMALIZED RADIAL DISTANCE. This is the key to stability:
// both the depth pass and this sampler compute radial distance the same way.
float samplePointShadowDP(float3 worldPos, float3 normal, float ndotl) {
    float invRadius = ubo.pointLightParams.x;
    float2 flipOffset = ubo.pointLightParams.yz;
    float4 baseUvRect = ubo.pointLightAtlasRect;

    // Inset the rect by one texel to avoid bleeding across the hemisphere seam.
    uint atlasW, atlasH;
    spotShadowAtlas.GetDimensions(atlasW, atlasH);
    float2 texel = float2(1.0 / float(atlasW), 1.0 / float(atlasH));
    baseUvRect.xy += texel;
    baseUvRect.zw -= texel * 2.0;

    // World -> light-local.
    float3 localVert = mul(ubo.pointLightView, float4(worldPos, 1.0)).xyz;
    float3 localNormal = normalize(mul((float3x3)ubo.pointLightView, normal));

    float shadowLen = length(localVert);
    float3 shadowDir = localVert / max(shadowLen, 1e-6);

    // Normal-offset bias in light-local space (Godot shadow_normal_bias).
    float normalBias = 2.0 * texel.x * (1.0 - abs(dot(localNormal, shadowDir)));
    float3 shadowSample = normalize(shadowDir + localNormal * normalBias);

    // Pick hemisphere: +Z half lives at base + flip_offset (bottom of atlas),
    // -Z half at the base rect (top). Must match the render pass assignment.
    float4 uvRect = baseUvRect;
    if (shadowSample.z >= 0.0) {
        uvRect.xy += flipOffset;
    }

    // Paraboloid: must match vertex shader's xy / (1 + n.z) with n.z >= 0.
    // Vertex mirrors z to positive (dpSide), so n.z is always ≥ 0 and
    // outputs xy / (1 + |z|). Godot uses this formulation because 1 + |z|
    // avoids the pole singularity of 1 - |z|.
    float denom = 1.0 + abs(shadowSample.z);
    float2 pos = shadowSample.xy / max(denom, 1e-4);
    pos = pos * 0.5 + 0.5;

    // Godot: depth = (shadow_len - bias) * inv_radius, then depth = 1.0 - depth
    // because Godot's shadow atlas comparison expects inverted depth.
    // Our atlas stores distance/range directly (not inverted), so we compare
    // refDepth <= stored (closer fragments are lit, farther are shadowed).
    float depthBias = max(0.0015 * (1.0 - ndotl), 0.0004);
    float refDepth = shadowLen * invRadius - depthBias;

    // 3x3 PCF within the hemisphere sub-rect.
    float2 pcfStep = texel;
    float shadow = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float2 uv = pos + float2(x, y) * pcfStep;
            uv = clamp(uv, float2(0.0, 0.0), float2(1.0, 1.0));
            uv = uvRect.xy + uv * uvRect.zw;
            float stored = spotShadowAtlas.SampleLevel(spotShadowSampler, uv, 0).r;
            shadow += refDepth <= stored ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
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
