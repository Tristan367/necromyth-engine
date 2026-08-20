#include "lib/frame_uniforms.slang"
#include "lib/mesh_types.slang"
#include "lib/textured_push_constants.slang"
#include "lib/instance_data.slang"
#include "lib/textured_bindings.slang"
#include "lib/shadow.slang"
#include "lib/surface.slang"

[shader("vertex")]
VSOutput vertMain(VSInput input, uint instanceId : SV_InstanceID) {
    GpuInstance inst = loadInstance(instanceId);
    VSOutput output;
    float4 worldPos = mul(inst.model, float4(input.inPosition, 1.0));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));
    output.worldPos = worldPos.xyz;
    output.worldNormal = normalize(mul((float3x3)inst.model, input.inNormal));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    // Instance base layer plus the vertex's own material, so a chunk of many
    // block types is one draw.
    output.textureLayer = inst.textureLayer + input.inMaterial;
    output.sampleTextureArray = inst.sampleTextureArray;
    return output;
}

[shader("fragment")] float4 fragOpaqueHard(VSOutput vertIn) : SV_Target { return shadeOpaqueSingle(vertIn, 0); }
[shader("fragment")] float4 fragOpaquePcf(VSOutput vertIn) : SV_Target { return shadeOpaqueSingle(vertIn, 1); }
[shader("fragment")] float4 fragOpaqueHardCsm2(VSOutput vertIn) : SV_Target { return shadeOpaqueDual(vertIn, 0); }
[shader("fragment")] float4 fragOpaquePcfCsm2(VSOutput vertIn) : SV_Target { return shadeOpaqueDual(vertIn, 1); }

[shader("fragment")] float4 fragCutoutHard(VSOutput vertIn) : SV_Target { return shadeCutoutSingle(vertIn, 0); }
[shader("fragment")] float4 fragCutoutPcf(VSOutput vertIn) : SV_Target { return shadeCutoutSingle(vertIn, 1); }
[shader("fragment")] float4 fragCutoutHardCsm2(VSOutput vertIn) : SV_Target { return shadeCutoutDual(vertIn, 0); }
[shader("fragment")] float4 fragCutoutPcfCsm2(VSOutput vertIn) : SV_Target { return shadeCutoutDual(vertIn, 1); }

[shader("fragment")] float4 fragA2CHard(VSOutput vertIn) : SV_Target { return shadeAlphaToCoverageSingle(vertIn, 0); }
[shader("fragment")] float4 fragA2CPcf(VSOutput vertIn) : SV_Target { return shadeAlphaToCoverageSingle(vertIn, 1); }
[shader("fragment")] float4 fragA2CHardCsm2(VSOutput vertIn) : SV_Target { return shadeAlphaToCoverageDual(vertIn, 0); }
[shader("fragment")] float4 fragA2CPcfCsm2(VSOutput vertIn) : SV_Target { return shadeAlphaToCoverageDual(vertIn, 1); }
