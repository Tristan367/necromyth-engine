#include "lib/frame_uniforms.slang"
#include "lib/textured_push_constants.slang"

struct VSInputPosition {
    float3 inPosition;
};

// Depth-only pass (VS): cutout/A2C cast opaque silhouettes until alpha-tested shadow FS exists.
[shader("vertex")]
float4 vertMain(VSInputPosition input) : SV_Position {
    uint cascade = pushConstants.shadowCascadeIndex;
    return mul(ubo.lightViewProj[cascade], mul(pushConstants.model, float4(input.inPosition, 1.0)));
}
