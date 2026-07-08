// Cubemap point-light shadow pass.
// Depth-only: writes linear distance / range to SV_Depth.
// Hardware depth attachment provides z-test with early-Z.
// Sampled in lighting.slang via comparison sampler (SampleCmp + hardware PCF).
// Per-light transforms read from storage buffer at set 0, binding 6,
// indexed by pushConstants.pointLightIndex — avoids CPU/GPU UBO overwrite.
#include "lib/mesh_types_skinned.slang"
#include "lib/textured_push_constants.slang"

struct PointLightShadowData {
    float4x4 view;          // translate(-lightPos)
    float4x4 faceVP[6];     // proj * faceView per face
    float4 pos;             // world-space light position
    float range;            // max range
    float _pad0;
    float _pad1;
    float _pad2;
};

[[vk::binding(6, 0)]]
StructuredBuffer<PointLightShadowData> perLightData;

struct VSInput {
    float3 inPosition : POSITION0;
};

struct VSOutput {
    float4 position : SV_Position;
    float  lightDist : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertMain(VSInput input, uint viewID : SV_ViewID) {
    uint li = pushConstants.pointLightIndex;
    PointLightShadowData d = perLightData[li];

    VSOutput output;
    float4 worldPos = mul(pushConstants.model, float4(input.inPosition, 1.0));
    float4 lightLocal = mul(d.view, worldPos);
    output.position = mul(d.faceVP[viewID], lightLocal);
    output.lightDist = length(worldPos.xyz - d.pos.xyz);
    return output;
}

// Skinned variant: joint matrices at set 1, binding 1
[[vk::binding(1, 1)]]
StructuredBuffer<float4x4> jointMatrices;

[shader("vertex")]
VSOutput vertMainSkinned(VSInputSkinned input, uint viewID : SV_ViewID) {
    uint li = pushConstants.pointLightIndex;
    PointLightShadowData d = perLightData[li];

    float4x4 skinMat =
        input.inJointWeights.x * jointMatrices[(int)input.inJointIndices.x] +
        input.inJointWeights.y * jointMatrices[(int)input.inJointIndices.y] +
        input.inJointWeights.z * jointMatrices[(int)input.inJointIndices.z] +
        input.inJointWeights.w * jointMatrices[(int)input.inJointIndices.w];

    VSOutput output;
    float4 skinned = mul(skinMat, float4(input.inPosition, 1.0));
    float4 worldPos = mul(pushConstants.model, skinned);
    float4 lightLocal = mul(d.view, worldPos);
    output.position = mul(d.faceVP[viewID], lightLocal);
    output.lightDist = length(worldPos.xyz - d.pos.xyz);
    return output;
}

[shader("fragment")]
float fragMain(VSOutput input, uint viewID : SV_ViewID) : SV_Depth {
    uint li = pushConstants.pointLightIndex;
    float r = perLightData[li].range;
    return input.lightDist / max(r, 0.001);
}
