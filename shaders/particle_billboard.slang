// Ultra-minimal billboard particle shader.
// Push constants: viewProj(64) + camRight(16) + camUp(16) + size(4) + color(16) = 116 bytes
// SSBO (set 0, binding 7): StructuredBuffer<float4> — per-particle position (xyz, w=unused)
// gl_VertexIndex (0,1,2) picks equilateral triangle corner; gl_InstanceIndex indexes particle.
// No backface cull — billboards always face camera.

struct ParticlePushConstants {
    float4x4 viewProj;
    float4 camRight;
    float4 camUp;
    float size;
    float4 color;
};

[[vk::push_constant]] ConstantBuffer<ParticlePushConstants> pc;

[[vk::binding(7, 0)]]
StructuredBuffer<float4> particlePositions;

struct VSOutput {
    float4 position : SV_Position;
};

// Equilateral triangle corners (unit size): bottom-left, bottom-right, top
static const float2 corners[3] = {
    float2(-0.5, -0.28867513459),
    float2( 0.5, -0.28867513459),
    float2( 0.0,  0.57735026919),
};

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) {
    float3 worldPos = particlePositions[instanceID].xyz;
    float2 c = corners[vertexID] * pc.size;
    float3 cornerWorld = worldPos + pc.camRight.xyz * c.x + pc.camUp.xyz * c.y;

    VSOutput output;
    output.position = mul(pc.viewProj, float4(cornerWorld, 1.0));
    return output;
}

[shader("fragment")]
float4 fragMain() : SV_Target {
    return pc.color;
}
