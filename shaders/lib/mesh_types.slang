struct VSInput {
    float3 inPosition;
    float3 inNormal;
    float3 inColor;
    float2 inTexCoord;
};

struct VSOutput {
    float4 pos : SV_Position;
    float3 worldNormal;
    float3 worldPos;
    float3 fragColor;
    float2 fragTexCoord;
    // Carried from the instance record. nointerpolation because these are
    // per-instance selectors, not surface quantities -- interpolating a texture
    // layer across a triangle is meaningless, and on a batched draw the
    // neighbouring vertices may belong to a different instance entirely.
    nointerpolation uint textureLayer;
    nointerpolation uint sampleTextureArray;
};
