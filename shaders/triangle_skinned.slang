#include "lib/frame_uniforms.slang"
#include "lib/mesh_types.slang"
#include "lib/mesh_types_skinned.slang"
#include "lib/textured_push_constants.slang"

[[vk::binding(1, 1)]]
StructuredBuffer<float4x4> jointMatrices;

[shader("vertex")]
VSOutput vertMainSkinned(VSInputSkinned input) {
    VSOutput output;

    float4x4 skinMat =
        input.inJointWeights.x * jointMatrices[(int)input.inJointIndices.x] +
        input.inJointWeights.y * jointMatrices[(int)input.inJointIndices.y] +
        input.inJointWeights.z * jointMatrices[(int)input.inJointIndices.z] +
        input.inJointWeights.w * jointMatrices[(int)input.inJointIndices.w];

    float4 worldPos = mul(pushConstants.model, mul(skinMat, float4(input.inPosition, 1.0)));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));
    output.worldPos = worldPos.xyz;
    output.worldNormal = normalize(mul((float3x3)mul(pushConstants.model, skinMat), input.inNormal));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    return output;
}
