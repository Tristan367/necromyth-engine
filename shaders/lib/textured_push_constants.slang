struct PushConstants {
    float4x4 model;
    uint textureArrayLayer;
    uint sampleTextureArray;
    uint shadowCascadeIndex;
    uint pointLightIndex;
    float _pad0;
    float _pad1;
    float _pad2;
};
[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;
