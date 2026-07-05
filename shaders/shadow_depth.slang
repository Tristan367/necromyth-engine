#include "lib/frame_uniforms.slang"
#include "lib/textured_push_constants.slang"

struct VSInputPosition {
    float3 inPosition;
};

// Depth-only pass (VS): cascade 0-1 = directional, 2-5 = spot, 6-11 = point
[shader("vertex")]
float4 vertMain(VSInputPosition input) : SV_Position {
    uint cascade = pushConstants.shadowCascadeIndex;
    float4x4 vp;
    if (cascade < 2)      vp = ubo.lightViewProj[cascade];
    else if (cascade < 6) vp = ubo.spotLightViewProj[cascade - 2];
    else                  vp = ubo.pointLightViewProj[cascade - 6];
    return mul(vp, mul(pushConstants.model, float4(input.inPosition, 1.0)));
}
