// The game's own UI: screen-space quads, one instanced draw for the whole HUD.
//
// No vertex buffer. gl_VertexIndex picks a corner and gl_InstanceIndex picks a
// quad out of an SSBO, which is the same trick particle_billboard.slang uses and
// it earns its keep for the same reason: the UI is rebuilt from scratch every
// frame, so the cost that matters is getting this frame's quads to the GPU, and
// a structured buffer written straight into the frame's staging ring costs one
// memcpy and no vertex input state at all.
//
// Everything is in PIXELS on the way in. The shader divides by the screen size,
// so the C++ side never has to think in clip space and a HUD laid out at integer
// pixel positions lands on integer pixels -- which is the whole point of a pixel
// font, and the thing that goes wrong the moment any part of the chain works in
// normalised coordinates.

struct UiPushConstants {
    // 1/width and 1/height, so the shader multiplies rather than divides.
    float2 inverseScreen;
    float2 pad;
};

[[vk::push_constant]] ConstantBuffer<UiPushConstants> pc;

// One rectangle of the HUD.
//
// `uv` is a rect too, not a per-vertex attribute: a quad is axis-aligned and its
// texture window is axis-aligned, so four corners' worth of UV is two corners'
// worth of information. Untextured rectangles -- panels, bars, the crosshair --
// point at the atlas's solid white texel and pay nothing extra.
struct UiQuad {
    float4 rect;  // x, y, width, height, in pixels from the top left
    float4 uv;    // u0, v0, u1, v1
    float4 color; // straight alpha, multiplied into the sample
};

[[vk::binding(9, 0)]]
StructuredBuffer<UiQuad> quads;

[[vk::binding(10, 0)]]
Sampler2D fontAtlas;

struct VSOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
};

[shader("vertex")]
VSOutput vertMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID) {
    const UiQuad quad = quads[instanceId];

    // Two triangles, as six corners. Written out rather than indexed because six
    // vertices is cheaper than an index buffer to own.
    const float2 corners[6] = {
        float2(0, 0), float2(1, 0), float2(1, 1),
        float2(0, 0), float2(1, 1), float2(0, 1),
    };
    const float2 corner = corners[vertexId];

    const float2 pixel = quad.rect.xy + corner * quad.rect.zw;

    VSOutput output;
    // Pixels to clip space. Y is flipped because a HUD counts down from the top
    // and Vulkan's clip space counts up.
    output.position = float4(pixel * pc.inverseScreen * 2.0 - 1.0, 0.0, 1.0);
    output.uv = lerp(quad.uv.xy, quad.uv.zw, corner);
    output.color = quad.color;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    // The atlas is coverage in one channel; the colour comes from the quad. That
    // keeps the atlas tiny and lets the same glyph be drawn in any colour
    // without a second texture.
    const float coverage = fontAtlas.Sample(input.uv).r;
    return float4(input.color.rgb, input.color.a * coverage);
}
