struct PushConstants {
    uint instanceBase;        // index of this draw's first instance record
    uint shadowCascadeIndex;
    uint pointLightIndex;
    uint _pad;
};
[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;
