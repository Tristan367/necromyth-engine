// Locations are explicit because Slang refuses to mix implicit and explicit
// bindings, and inMaterial has to sit at 6 to leave 4 and 5 for the skinning
// attributes.
struct VSInput {
    [[vk::location(0)]] float3 inPosition;
    [[vk::location(1)]] float3 inNormal;
    [[vk::location(2)]] float3 inColor;
    [[vk::location(3)]] float2 inTexCoord;
    // Per-vertex texture array layer, added to the instance's base layer. One
    // mesh can therefore carry many materials.
    [[vk::location(6)]] uint inMaterial;
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
