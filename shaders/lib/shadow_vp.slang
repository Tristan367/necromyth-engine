// Shared cascade VP selection for directional/spot shadow passes.
float4x4 selectShadowVP(uint cascade) {
    float4x4 vp;
    if (cascade < 2)      vp = ubo.lightViewProj[cascade];
    else                  vp = ubo.spotLightViewProj[cascade - 2];
    return vp;
}

// Keeps casters that sit in front of a directional cascade's near plane instead
// of clipping them away -- Godot's "shadow pancake" (scene_forward_clustered.glsl
// clamps gl_Position.z under SCENE_DATA_FLAGS_USE_PANCAKE_SHADOWS).
//
// A cascade is fitted to the receivers it covers, so a caster far up the light
// direction -- a tower shadowing the pavement at its foot -- lies outside it.
// Clipped, it casts nothing. Clamped to the near plane it still occludes, which
// is the right answer: it is above everything in the box, so it shadows whatever
// is under it. k_shadow_depth_extension keeps ordinary geometry well clear of
// this; the clamp is the backstop for the genuinely tall.
//
// Directional cascades only. Spot lights are perspective, where z/w is not a
// depth you can clamp this way.
float4 pancakeShadowClip(float4 clipPos, uint cascade) {
    if (cascade < 2)
        clipPos.z = max(clipPos.z, 0.0);
    return clipPos;
}
