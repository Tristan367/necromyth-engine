#include "lib/frame_uniforms.slang"
#include "lib/shadow_vp.slang"
#include "lib/textured_push_constants.slang"
#include "lib/instance_data.slang"

struct VSInputPosition {
    float3 inPosition;
};

struct VSOutput {
    float4 position : SV_Position;
};

// Depth-only pass (VS):
//   cascade 0-1 = directional cascades (perspective/ortho VP)
//   cascade 2-5 = spot lights (perspective VP)
[shader("vertex")]
VSOutput vertMain(VSInputPosition input, uint instanceId : SV_InstanceID) {
    GpuInstance inst = loadInstance(instanceId);
    VSOutput output;
    uint cascade = pushConstants.shadowCascadeIndex;

    float4x4 vp = selectShadowVP(cascade);
    output.position = mul(vp, mul(inst.model, float4(input.inPosition, 1.0)));
    return output;
}
