#include "lib/frame_uniforms.slang"
#include "lib/mesh_types.slang"
#include "lib/textured_push_constants.slang"

[shader("vertex")]
float4 vertMain(VSInput input) : SV_Position {
    return mul(ubo.lightViewProj, mul(pushConstants.model, float4(input.inPosition, 1.0)));
}
