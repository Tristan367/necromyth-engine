struct PushConstants {
    float4x4 model;
    uint textureArrayLayer;
    uint sampleTextureArray;
    uint shadowCascadeIndex;
    float _pad; // alignment padding
};
[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;
