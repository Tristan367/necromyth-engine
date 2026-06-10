static const float kAlphaCutoff = 0.5;

float4 shadeOpaque(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord);
    float3 baseColor = texColor.rgb * vertIn.fragColor;
    float3 lightDir = normalize(ubo.lightDirection.xyz);
    float3 normal = normalize(vertIn.worldNormal);
    float ndotl = max(dot(normal, lightDir), 0.0);
    float ambient = ubo.lightColor.w;
    float3 diffuse = ubo.lightColor.rgb * ndotl;
    float shadow = shadowVisibility(vertIn.worldPos, normal, lightDir, filterMode);
    float3 lit = baseColor * (ambient + diffuse * shadow);
    return float4(lit, texColor.a);
}

float4 shadeCutout(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord);
    if (texColor.a < kAlphaCutoff)
        discard;
    return shadeOpaque(vertIn, filterMode);
}

float4 shadeAlphaToCoverage(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord);
    if (texColor.a <= 0.0)
        discard;

    float3 baseColor = texColor.rgb * vertIn.fragColor;
    float3 lightDir = normalize(ubo.lightDirection.xyz);
    float3 normal = normalize(vertIn.worldNormal);
    float ndotl = max(dot(normal, lightDir), 0.0);
    float ambient = ubo.lightColor.w;
    float3 diffuse = ubo.lightColor.rgb * ndotl;
    float shadow = shadowVisibility(vertIn.worldPos, normal, lightDir, filterMode);
    float3 lit = baseColor * (ambient + diffuse * shadow);
    return float4(lit, texColor.a);
}
