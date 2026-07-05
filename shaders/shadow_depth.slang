#include "lib/frame_uniforms.slang"
#include "lib/textured_push_constants.slang"

struct VSInputPosition {
    float3 inPosition;
};

struct VSOutput {
    float4 position : SV_Position;
    // Clip the wrong paraboloid hemisphere (Godot's dp_clip). One clip plane;
    // fragments with clipDistance < 0 are discarded by the rasterizer.
    float clipDistance : SV_ClipDistance0;
};

// Depth-only pass (VS):
//   cascade 0-1 = directional cascades (perspective/ortho VP)
//   cascade 2-5 = spot lights (perspective VP)
//   cascade 6-7 = point light dual-paraboloid halves (+Z side, -Z side)
//                 Godot omni model: render normalized RADIAL distance into a
//                 paraboloid-warped position. No projection matrix.
[shader("vertex")]
VSOutput vertMain(VSInputPosition input) {
    VSOutput output;
    uint cascade = pushConstants.shadowCascadeIndex;

    if (cascade < 6) {
        float4x4 vp;
        if (cascade < 2)      vp = ubo.lightViewProj[cascade];
        else                  vp = ubo.spotLightViewProj[cascade - 2];
        output.position = mul(vp, mul(pushConstants.model, float4(input.inPosition, 1.0)));
        output.clipDistance = 1.0; // never clipped
        return output;
    }

    // --- Point light: dual-paraboloid (Godot MODE_DUAL_PARABOLOID) ---
    // World position of the vertex.
    float4 world = mul(pushConstants.model, float4(input.inPosition, 1.0));
    // Into light-local space (rigid, no projection).
    float3 v = mul(ubo.pointLightView, world).xyz;

    float dpSide = pushConstants.dpSide; // +1 for +Z half, -1 for -Z half
    float zFar   = pushConstants.dpZFar;

    // Fold onto this hemisphere: mirror z by the side, then clip the far half.
    v.z *= dpSide;
    output.clipDistance = v.z; // discard vertices on the opposite hemisphere

    float dist = length(v);
    float3 n = v / max(dist, 1e-6);
    // Paraboloid: Godot uses 1 + abs(z) (stable at poles, disc radius 0.5)
    float2 xy = n.xy / (1.0 + n.z);
    float z01 = dist / max(zFar, 1e-6);

    // Position in clip space. xy already in [-1,1] (paraboloid disc);
    // Vulkan NDC z in [0,1] (GLM_FORCE_DEPTH_ZERO_TO_ONE), so write z01 directly.
    output.position = float4(xy, z01, 1.0);
    return output;
}
