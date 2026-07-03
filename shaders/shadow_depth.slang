#include "lib/frame_uniforms.slang"
#include "lib/textured_push_constants.slang"

struct VSInputPosition {
    float3 inPosition;
};

// Depth-only pass (VS): cascade 0-1 = directional, cascade 2-5 = spot
[shader("vertex")]
float4 vertMain(VSInputPosition input) : SV_Position {
    uint cascade = pushConstants.shadowCascadeIndex;
    float4x4 vp = cascade < 2 ? ubo.lightViewProj[cascade] : ubo.spotLightViewProj[cascade - 2];
    return mul(vp, mul(pushConstants.model, float4(input.inPosition, 1.0)));
}
