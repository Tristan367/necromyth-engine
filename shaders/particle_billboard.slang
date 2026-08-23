// Billboard particles.
//
// Push constants: viewProj(64) + camRight(16) + camUp(16) = 96 bytes.
// SSBO (set 0, binding 7): per-particle { float4 pos_size; float4 color; }.
//
// Colour and size used to be push constants, shared by every particle in the
// frame. That is enough for a dust puff and nothing else: a flame is yellow at
// its base and dark red at its tip and smaller as it dies, and none of that can
// be said with one colour. They are per particle now, and the draw call count
// is unchanged -- it was always one instanced draw over the whole buffer.
//
// gl_VertexIndex (0,1,2) picks a triangle corner; gl_InstanceIndex indexes the
// particle. No backface cull -- billboards always face the camera.

struct ParticlePushConstants {
    float4x4 viewProj;
    float4 camRight;
    float4 camUp;
};

[[vk::push_constant]] ConstantBuffer<ParticlePushConstants> pc;

struct GpuParticle {
    float4 posSize;
    float4 color;
};

[[vk::binding(7, 0)]]
StructuredBuffer<GpuParticle> particles;

struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR0;
    // Barycentric-ish coordinate across the triangle, so the fragment shader can
    // round the corners off. A particle drawn as a bare triangle reads as a
    // triangle, which is fine for gravel and wrong for a flame.
    float2 offset : TEXCOORD0;
};

// An equilateral triangle with its vertices on the unit circle. Its INCIRCLE
// therefore has radius 0.5, and that is the number the fragment shader has to
// fade within: a blob that reaches radius 1.0 reaches the vertices and gets
// sliced flat by the three edges between them, which is how a cloud of soft
// sparks comes out looking like a pile of paper darts. It did.
static const float2 corners[3] = {
    float2(-0.866, -0.5),
    float2( 0.866, -0.5),
    float2( 0.0,    1.0),
};

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) {
    GpuParticle p = particles[instanceID];
    float2 c = corners[vertexID];
    float3 cornerWorld = p.posSize.xyz + pc.camRight.xyz * (c.x * p.posSize.w)
                                       + pc.camUp.xyz * (c.y * p.posSize.w);

    VSOutput output;
    output.position = mul(pc.viewProj, float4(cornerWorld, 1.0));
    output.color = p.color;
    output.offset = c;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target {
    // A soft round particle: opaque in the middle, gone by the edge. One
    // smoothstep, and it is the difference between a spark and a paper triangle.
    // Fades to nothing by 0.5, which is the incircle -- see the corner table.
    float r = length(vertIn.offset);
    float alpha = vertIn.color.a * (1.0 - smoothstep(0.12, 0.5, r));
    if (alpha <= 0.004)
        discard;
    return float4(vertIn.color.rgb, alpha);
}
