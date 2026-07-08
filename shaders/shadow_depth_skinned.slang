#include "lib/frame_uniforms.slang"
#include "lib/mesh_types_skinned.slang"
#include "lib/shadow_vp.slang"
#include "lib/skinning.slang"
#include "lib/textured_push_constants.slang"

[[vk::binding(1, 1)]]
StructuredBuffer<float4x4> jointMatrices;

struct VSOutput {
    float4 position : SV_Position;
};
[shader("vertex")]
VSOutput vertMainSkinned(VSInputSkinned input) {
    float4x4 skinMat = computeSkinMatrix(input, jointMatrices);

    VSOutput output;
    uint cascade = pushConstants.shadowCascadeIndex;
    float4 skinned = mul(skinMat, float4(input.inPosition, 1.0));

    float4x4 vp = selectShadowVP(cascade);
    output.position = mul(vp, mul(pushConstants.model, skinned));
    return output;
}
