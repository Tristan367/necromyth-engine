static const float kAlphaCutoff = 0.5;

float4 shadeOpaqueSingle(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord);
    float3 baseColor = texColor.rgb * vertIn.fragColor;
    float3 lightDir = normalize(ubo.lightDirection.xyz);
    float3 normal = normalize(vertIn.worldNormal);
    float ndotl = max(dot(normal, lightDir), 0.0);
    float ambient = ubo.lightColor.w;
    float3 diffuse = ubo.lightColor.rgb * ndotl;
    float shadow = shadowVisibilitySingle(vertIn.worldPos, normal, lightDir, filterMode);
    float3 lit = baseColor * (ambient + diffuse * shadow);
    return float4(lit, texColor.a);
}

float4 shadeOpaqueDual(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord);
    float3 baseColor = texColor.rgb * vertIn.fragColor;
    float3 lightDir = normalize(ubo.lightDirection.xyz);
    float3 normal = normalize(vertIn.worldNormal);
    float ndotl = max(dot(normal, lightDir), 0.0);
    float ambient = ubo.lightColor.w;
    float3 diffuse = ubo.lightColor.rgb * ndotl;
    float shadow = shadowVisibilityDual(vertIn.worldPos, normal, lightDir, filterMode);
    float3 lit = baseColor * (ambient + diffuse * shadow);
    return float4(lit, texColor.a);
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
    float3 lightDir = normalize(ubo.lightDirection.xyz);
    float3 normal = normalize(vertIn.worldNormal);
    float ndotl = max(dot(normal, lightDir), 0.0);
    float ambient = ubo.lightColor.w;
    float3 diffuse = ubo.lightColor.rgb * ndotl;
    float shadow = shadowVisibilitySingle(vertIn.worldPos, normal, lightDir, filterMode);
    float3 lit = baseColor * (ambient + diffuse * shadow);
    return float4(lit, texColor.a);
}

float4 shadeAlphaToCoverageDual(VSOutput vertIn, int filterMode) {
    float4 texColor = sampleBaseColor(vertIn.fragTexCoord);
    if (texColor.a <= 0.0)
        discard;

    float3 baseColor = texColor.rgb * vertIn.fragColor;
    float3 lightDir = normalize(ubo.lightDirection.xyz);
    float3 normal = normalize(vertIn.worldNormal);
    float ndotl = max(dot(normal, lightDir), 0.0);
    float ambient = ubo.lightColor.w;
    float3 diffuse = ubo.lightColor.rgb * ndotl;
    float shadow = shadowVisibilityDual(vertIn.worldPos, normal, lightDir, filterMode);
    float3 lit = baseColor * (ambient + diffuse * shadow);
    return float4(lit, texColor.a);
}
