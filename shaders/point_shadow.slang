// Cubemap point-light shadow pass (Sascha Willems omni model).
// Renders 6 cube faces to an R32F cubemap array, storing distance from light.
// Sampled in lighting.slang via TextureCubeArray + manual depth comparison.
#include "lib/frame_uniforms.slang"
#include "lib/mesh_types_skinned.slang"
#include "lib/textured_push_constants.slang"

struct VSInput {
    float3 inPosition : POSITION0;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    uint face = pushConstants.shadowCascadeIndex - 10; // 0-5
    float4 worldPos = mul(pushConstants.model, float4(input.inPosition, 1.0));
    output.worldPos = worldPos.xyz;
    // Transform: world -> light-local -> face rotation -> perspective
    float4 lightLocal = mul(ubo.pointLightView, worldPos);
    output.position = mul(ubo.pointFaceVP[face], lightLocal);
    return output;
}

// Skinned variant: joint matrices at set 1, binding 1
[[vk::binding(1, 1)]]
StructuredBuffer<float4x4> jointMatrices;

[shader("vertex")]
VSOutput vertMainSkinned(VSInputSkinned input) {
    float4x4 skinMat =
        input.inJointWeights.x * jointMatrices[(int)input.inJointIndices.x] +
        input.inJointWeights.y * jointMatrices[(int)input.inJointIndices.y] +
        input.inJointWeights.z * jointMatrices[(int)input.inJointIndices.z] +
        input.inJointWeights.w * jointMatrices[(int)input.inJointIndices.w];

    VSOutput output;
    uint face = pushConstants.shadowCascadeIndex - 10;
    float4 skinned = mul(skinMat, float4(input.inPosition, 1.0));
    float4 worldPos = mul(pushConstants.model, skinned);
    output.worldPos = worldPos.xyz;
    float4 lightLocal = mul(ubo.pointLightView, worldPos);
    output.position = mul(ubo.pointFaceVP[face], lightLocal);
    return output;
}

[shader("fragment")]
float fragMain(VSOutput input) : SV_Target0 {
    float3 lightVec = input.worldPos.xyz - ubo.pointLightPos.xyz;
    return length(lightVec);
}
