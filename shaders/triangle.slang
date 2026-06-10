#include "lib/frame_uniforms.slang"
#include "lib/mesh_types.slang"
#include "lib/textured_push_constants.slang"
#include "lib/textured_bindings.slang"
#include "lib/shadow.slang"
#include "lib/surface.slang"

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    float4 worldPos = mul(pushConstants.model, float4(input.inPosition, 1.0));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));
    output.worldPos = worldPos.xyz;
    output.worldNormal = normalize(mul((float3x3)pushConstants.model, input.inNormal));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    return output;
}

[shader("fragment")] float4 fragOpaqueHard(VSOutput vertIn) : SV_Target { return shadeOpaque(vertIn, 0); }
[shader("fragment")] float4 fragOpaquePcf(VSOutput vertIn) : SV_Target { return shadeOpaque(vertIn, 1); }

[shader("fragment")] float4 fragCutoutHard(VSOutput vertIn) : SV_Target { return shadeCutout(vertIn, 0); }
[shader("fragment")] float4 fragCutoutPcf(VSOutput vertIn) : SV_Target { return shadeCutout(vertIn, 1); }

[shader("fragment")] float4 fragA2CHard(VSOutput vertIn) : SV_Target { return shadeAlphaToCoverage(vertIn, 0); }
[shader("fragment")] float4 fragA2CPcf(VSOutput vertIn) : SV_Target { return shadeAlphaToCoverage(vertIn, 1); }
