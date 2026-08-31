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
    // The material field packs the texture layer in its low byte, baked voxel
    // light in the next byte, and a flag above that saying the light is real.
    //
    // The flag is the whole point. This used to test `packedLight == 0` for
    // "this mesh carries no baked light, draw it lit" -- but a voxel in the
    // middle of a windowless room packs to zero as well, because zero light IS
    // its light. So interior lighting fell off smoothly to nothing and then
    // snapped to FULL brightness at the darkest point, which is the opposite of
    // what it should do and reads exactly like an integer wrapping around.
    //
    // Meshes that never set the field still read as layer 0 with no flag, so
    // unlit geometry is unchanged.
    output.textureLayer = inst.textureLayer + (input.inMaterial & 0xFFu);
    const uint packedLight = input.inMaterial >> 8u;
    output.bakedLight = (packedLight & 0x100u) == 0u
        ? float2(1.0, 0.0)
        : float2(float(packedLight & 0xFu) / 15.0, float((packedLight >> 4u) & 0xFu) / 15.0);
    output.sampleTextureArray = inst.sampleTextureArray;
    return output;
}

// The packed terrain path: identical to vertMain except the vertex carries no
// colour (terrain is always white there) -- which is most of why its vertex is
// 36 bytes instead of 80.
[shader("vertex")]
VSOutput vertMainTerrain(VSInputTerrain input, uint instanceId : SV_InstanceID) {
    GpuInstance inst = loadInstance(instanceId);
    VSOutput output;
    float4 worldPos = mul(inst.model, float4(input.inPosition, 1.0));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));
    output.worldPos = worldPos.xyz;
    output.worldNormal = normalize(mul((float3x3)inst.model, input.inNormal));
    output.fragColor = float3(1.0, 1.0, 1.0);
    output.fragTexCoord = input.inTexCoord;
    // Same packing as vertMain: layer in the low byte, baked light above it,
    // with the bit-16 flag saying the light is real. See the note there.
    output.textureLayer = inst.textureLayer + (input.inMaterial & 0xFFu);
    const uint packedLight = input.inMaterial >> 8u;
    output.bakedLight = (packedLight & 0x100u) == 0u
        ? float2(1.0, 0.0)
        : float2(float(packedLight & 0xFu) / 15.0, float((packedLight >> 4u) & 0xFu) / 15.0);
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
