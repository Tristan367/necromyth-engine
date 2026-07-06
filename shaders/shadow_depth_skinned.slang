#include "lib/frame_uniforms.slang"
#include "lib/mesh_types_skinned.slang"
#include "lib/textured_push_constants.slang"

[[vk::binding(1, 1)]]
StructuredBuffer<float4x4> jointMatrices;

struct VSOutput {
    float4 position : SV_Position;
};
[shader("vertex")]
VSOutput vertMainSkinned(VSInputSkinned input) {
    float4x4 skinMat =
        input.inJointWeights.x * jointMatrices[(int)input.inJointIndices.x] +
        input.inJointWeights.y * jointMatrices[(int)input.inJointIndices.y] +
        input.inJointWeights.z * jointMatrices[(int)input.inJointIndices.z] +
        input.inJointWeights.w * jointMatrices[(int)input.inJointIndices.w];

    VSOutput output;
    uint cascade = pushConstants.shadowCascadeIndex;
    float4 skinned = mul(skinMat, float4(input.inPosition, 1.0));

    float4x4 vp;
    if (cascade < 2)      vp = ubo.lightViewProj[cascade];
    else                  vp = ubo.spotLightViewProj[cascade - 2];
    output.position = mul(vp, mul(pushConstants.model, skinned));
    return output;
}
