#include "lib/frame_uniforms.slang"
#include "lib/mesh_types_skinned.slang"
#include "lib/textured_push_constants.slang"

[[vk::binding(1, 1)]]
StructuredBuffer<float4x4> jointMatrices;

[shader("vertex")]
float4 vertMainSkinned(VSInputSkinned input) : SV_Position {
    float4x4 skinMat =
        input.inJointWeights.x * jointMatrices[(int)input.inJointIndices.x] +
        input.inJointWeights.y * jointMatrices[(int)input.inJointIndices.y] +
        input.inJointWeights.z * jointMatrices[(int)input.inJointIndices.z] +
        input.inJointWeights.w * jointMatrices[(int)input.inJointIndices.w];

    uint cascade = pushConstants.shadowCascadeIndex;
    float4x4 vp = cascade < 2 ? ubo.lightViewProj[cascade] : ubo.spotLightViewProj[cascade - 2];
    return mul(vp, mul(pushConstants.model, mul(skinMat, float4(input.inPosition, 1.0))));
}
