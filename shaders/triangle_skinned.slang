#include "lib/frame_uniforms.slang"
#include "lib/mesh_types.slang"
#include "lib/mesh_types_skinned.slang"
#include "lib/textured_push_constants.slang"
#include "lib/instance_data.slang"
#include "lib/skinning.slang"

[[vk::binding(1, 1)]]
StructuredBuffer<float4x4> jointMatrices;

[shader("vertex")]
VSOutput vertMainSkinned(VSInputSkinned input, uint instanceId : SV_InstanceID) {
    GpuInstance inst = loadInstance(instanceId);
    VSOutput output;

    float4x4 skinMat = computeSkinMatrix(input, jointMatrices, inst.boneBase);

    float4 worldPos = mul(inst.model, mul(skinMat, float4(input.inPosition, 1.0)));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));
    output.worldPos = worldPos.xyz;
    output.worldNormal = normalize(mul((float3x3)mul(inst.model, skinMat), input.inNormal));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    output.textureLayer = inst.textureLayer;
    output.sampleTextureArray = inst.sampleTextureArray;
    return output;
}
