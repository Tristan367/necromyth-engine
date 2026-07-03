// Point + spot light evaluation. Lights are stored in an SSBO (set 0, binding 3).
// GPU layout (16-byte aligned):
//   PointLight: vec4(pos.xyz, range), vec4(color.rgb, intensity)
//   SpotLight:  vec4(pos.xyz, range), vec4(color.rgb, intensity),
//               vec4(dir.xyz, pad),   vec4(inner, outer, 0, 0)
// Buffer: [uint numPoint, uint numSpot, uint pad0, uint pad1][PointLight...][SpotLight...]

struct GpuPointLight {
    float4 posRange;      // xyz = position, w = range
    float4 colorIntensity; // rgb = color, w = intensity
};

struct GpuSpotLight {
    float4 posRange;
    float4 colorIntensity;
    float4 dirPad;         // xyz = direction, w = unused
    float4 angles;         // x = inner_angle, y = outer_angle, zw = unused
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

    // Blinn-Phong specular
    float3 halfVec = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfVec), 0.0), 32.0);
    float3 specular = l.colorIntensity.rgb * l.colorIntensity.w * spec * 0.3;

    return (diffuse + specular) * atten;
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
    float spot = smoothstep(cosOuter, cosInner, cosTheta);
    atten *= spot;

    float ndotl = max(dot(normal, lightDir), 0.0);
    float3 diffuse = l.colorIntensity.rgb * l.colorIntensity.w * ndotl;

    float3 halfVec = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfVec), 0.0), 32.0);
    float3 specular = l.colorIntensity.rgb * l.colorIntensity.w * spec * 0.3;

    return (diffuse + specular) * atten;
}

float3 evaluatePointLights(float3 fragPos, float3 normal, float3 viewDir) {
    uint numPoint = lightBuffer.Load<uint>(0);
    if (numPoint == 0) return float3(0, 0, 0);

    float3 contrib = float3(0, 0, 0);
    uint offset = 16; // skip header
    for (uint i = 0; i < numPoint; ++i) {
        GpuPointLight l;
        float4 a = asfloat(lightBuffer.Load4(offset));
        float4 b = asfloat(lightBuffer.Load4(offset + 16));
        l.posRange = a;
        l.colorIntensity = b;
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
    uint offset = 16 + numPoint * 32; // skip header + point lights
    for (uint i = 0; i < numSpot; ++i) {
        GpuSpotLight l;
        float4 a = asfloat(lightBuffer.Load4(offset));
        float4 b = asfloat(lightBuffer.Load4(offset + 16));
        float4 c = asfloat(lightBuffer.Load4(offset + 32));
        float4 d = asfloat(lightBuffer.Load4(offset + 48));
        l.posRange = a;
        l.colorIntensity = b;
        l.dirPad = c;
        l.angles = d;
        contrib += calcSpotLight(l, fragPos, normal, viewDir);
        offset += 64;
    }
    return contrib;
}

float3 evaluateAllLights(float3 fragPos, float3 normal, float3 viewDir) {
    return evaluatePointLights(fragPos, normal, viewDir) +
           evaluateSpotLights(fragPos, normal, viewDir);
}
