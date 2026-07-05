#include "lib/frame_uniforms.slang"
#include "lib/mesh_types_skinned.slang"
#include "lib/textured_push_constants.slang"

[[vk::binding(1, 1)]]
StructuredBuffer<float4x4> jointMatrices;

struct VSOutput {
    float4 position : SV_Position;
    float clipDistance : SV_ClipDistance0;
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

    if (cascade < 6) {
        float4x4 vp;
        if (cascade < 2)      vp = ubo.lightViewProj[cascade];
        else                  vp = ubo.spotLightViewProj[cascade - 2];
        output.position = mul(vp, mul(pushConstants.model, skinned));
        output.clipDistance = 1.0;
        return output;
    }

    // Point light: dual-paraboloid (matches shadow_depth.slang).
    float4 world = mul(pushConstants.model, skinned);
    float3 v = mul(ubo.pointLightView, world).xyz;

    v.z *= pushConstants.dpSide;
    output.clipDistance = v.z;

    float dist = length(v);
    float3 n = v / max(dist, 1e-6);
    float2 xy = n.xy / (1.0 + n.z);
    float z01 = dist / max(pushConstants.dpZFar, 1e-6);

    output.position = float4(xy, z01, 1.0);
    return output;
}
