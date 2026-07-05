struct PushConstants {
    float4x4 model;
    uint textureArrayLayer;
    uint sampleTextureArray;
    uint shadowCascadeIndex;
    // Dual-paraboloid point shadow params (see textured_push_constants.hpp).
    float dpSide;   // +1.0 = +Z half, -1.0 = -Z half
    float dpZFar;   // point light range
};
[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;
