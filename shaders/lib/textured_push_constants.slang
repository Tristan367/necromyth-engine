struct PushConstants {
    float4x4 model;
    uint textureArrayLayer;
    uint sampleTextureArray;
    uint shadowCascadeIndex;
};
[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;
