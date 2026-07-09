// Shared cascade VP selection for directional/spot shadow passes.
float4x4 selectShadowVP(uint cascade) {
    float4x4 vp;
    if (cascade < 2)      vp = ubo.lightViewProj[cascade];
    else                  vp = ubo.spotLightViewProj[cascade - 2];
    return vp;
}
