// Cubemap point-light shadow pass.
// Depth-only: writes linear distance / range to SV_Depth.
// Hardware depth attachment provides z-test with early-Z.
// Sampled in lighting.slang via comparison sampler (SampleCmp + hardware PCF).
#include "lib/frame_uniforms.slang"
#include "lib/mesh_types_skinned.slang"
#include "lib/textured_push_constants.slang"

struct VSInput {
    float3 inPosition : POSITION0;
};

struct VSOutput {
    float4 position : SV_Position;
    float  lightDist : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertMain(VSInput input, uint viewID : SV_ViewID) {
    VSOutput output;
    float4 worldPos = mul(pushConstants.model, float4(input.inPosition, 1.0));
    float4 lightLocal = mul(ubo.pointLightView, worldPos);
    output.position = mul(ubo.pointFaceVP[viewID], lightLocal);
    output.lightDist = length(worldPos.xyz - ubo.pointLightPos.xyz);
    return output;
}

// Skinned variant: joint matrices at set 1, binding 1
[[vk::binding(1, 1)]]
StructuredBuffer<float4x4> jointMatrices;

[shader("vertex")]
VSOutput vertMainSkinned(VSInputSkinned input, uint viewID : SV_ViewID) {
    float4x4 skinMat =
        input.inJointWeights.x * jointMatrices[(int)input.inJointIndices.x] +
        input.inJointWeights.y * jointMatrices[(int)input.inJointIndices.y] +
        input.inJointWeights.z * jointMatrices[(int)input.inJointIndices.z] +
        input.inJointWeights.w * jointMatrices[(int)input.inJointIndices.w];

    VSOutput output;
    float4 skinned = mul(skinMat, float4(input.inPosition, 1.0));
    float4 worldPos = mul(pushConstants.model, skinned);
    float4 lightLocal = mul(ubo.pointLightView, worldPos);
    output.position = mul(ubo.pointFaceVP[viewID], lightLocal);
    output.lightDist = length(worldPos.xyz - ubo.pointLightPos.xyz);
    return output;
}

[shader("fragment")]
float fragMain(VSOutput input) : SV_Depth {
    return input.lightDist / ubo.pointLightParams.y;
}
