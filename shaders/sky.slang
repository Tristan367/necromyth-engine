#include "lib/frame_uniforms.slang"
// Shares the engine-wide push constant and instance layouts: background
// geometry goes through the same draw list and gets an instance record like
// everything else, so it must agree on both.
#include "lib/textured_push_constants.slang"
#include "lib/instance_data.slang"
#include "lib/textured_bindings.slang"

struct VSInput {
    float3 inPosition;
};

struct VSOutput {
    float4 pos : SV_Position;
    float3 direction;
};

[shader("vertex")]
VSOutput vertMain(VSInput input, uint instanceId : SV_InstanceID) {
    VSOutput output;
    output.direction = input.inPosition;

    float4 world = mul(loadInstance(instanceId).model, float4(input.inPosition, 1.0));
    float4 viewPos = mul(ubo.viewSky, world);
    float4 clipPos = mul(ubo.proj, viewPos);
    clipPos.z = clipPos.w;
    output.pos = clipPos;
    return output;
}

// Five colour stops, eased between with smoothstep inside each segment.
//
// This is PiecewiseSmoothstep from SkyManager.cs. The C# bakes the result into
// a cubemap because Godot's sky wanted a texture; we can evaluate it per pixel,
// which costs nothing here and means the ramp can change with the sun rather
// than being rebaked when it does.
float3 rampFive(float3 a, float3 b, float3 c, float3 d, float3 e, float t) {
    float s = saturate((t + 1.0) * 0.5);
    float seg = s * 4.0;
    int idx = clamp(int(floor(seg)), 0, 3);
    float local = smoothstep(0.0, 1.0, seg - float(idx));
    float3 from = idx == 0 ? a : (idx == 1 ? b : (idx == 2 ? c : d));
    float3 to   = idx == 0 ? b : (idx == 1 ? c : (idx == 2 ? d : e));
    return lerp(from, to, local);
}

// A cheap stable hash over a 3D cell.
//
// Cells are taken from the direction itself rather than from a projection of
// it: projecting on to xz and dividing by y stretches the cells near the
// horizon, and the stars came out as dashes lying on their sides.
float hash31(float3 p) {
    p = frac(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return frac((p.x + p.y) * p.z);
}

// Value noise and a small fbm, for cloud cover.
//
// The C# bakes cloud textures from seamless cellular noise because Godot's sky
// wanted a texture and generating one needed an async round trip through a
// NoiseTexture2D. We are already in a shader, so the noise can just be
// evaluated -- which also means the cover can change without regenerating
// anything.
float valueNoise(float2 p) {
    float2 i = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash31(float3(i, 0.0));
    float b = hash31(float3(i + float2(1.0, 0.0), 0.0));
    float c = hash31(float3(i + float2(0.0, 1.0), 0.0));
    float d = hash31(float3(i + float2(1.0, 1.0), 0.0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

float fbm(float2 p) {
    float sum = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 5; ++i) {
        sum += valueNoise(p) * amp;
        p *= 2.03;
        amp *= 0.5;
    }
    return sum;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target {
    float3 direction = normalize(vertIn.direction);
    float3 sunDir = normalize(ubo.lightDirection.xyz);

    // How high the sun is, which is the only clock the sky needs.
    //
    // From lightDirection.w rather than sunDir.y, because at night the shading
    // direction is swung to a moon vector above the horizon so that shadows
    // still have somewhere to come from -- and reading .y there paints a bright
    // blue midnight.
    float sunElev = ubo.lightDirection.w;
    float day   = smoothstep(-0.08, 0.22, sunElev);   // full daylight
    float night = smoothstep(0.02, -0.20, sunElev);   // full dark
    float twilight = saturate(1.0 - day - night);     // the band between them

    // Three palettes, five stops each: nadir, lower, horizon, upper, zenith.
    float3 dayA = float3(0.24, 0.30, 0.38), dayB = float3(0.52, 0.62, 0.76),
           dayC = float3(0.62, 0.73, 0.88), dayD = float3(0.30, 0.50, 0.82),
           dayE = float3(0.12, 0.30, 0.70);
    float3 duskA = float3(0.10, 0.09, 0.13), duskB = float3(0.42, 0.24, 0.20),
           duskC = float3(0.86, 0.46, 0.24), duskD = float3(0.36, 0.30, 0.48),
           duskE = float3(0.10, 0.13, 0.34);
    float3 nightA = float3(0.02, 0.02, 0.04), nightB = float3(0.03, 0.04, 0.08),
           nightC = float3(0.05, 0.07, 0.13), nightD = float3(0.03, 0.05, 0.11),
           nightE = float3(0.01, 0.02, 0.06);

    float3 a = dayA * day + duskA * twilight + nightA * night;
    float3 b = dayB * day + duskB * twilight + nightB * night;
    float3 c = dayC * day + duskC * twilight + nightC * night;
    float3 d = dayD * day + duskD * twilight + nightD * night;
    float3 e = dayE * day + duskE * twilight + nightE * night;

    float3 color = rampFive(a, b, c, d, e, direction.y);

    // Stars, from the NASA map, before the sun and the glow go over the top.
    //
    // This was a hash-based scatter of dots and it looked like a hash-based
    // scatter of dots: no constellations, no Milky Way, and a bug that left half
    // the sky empty because the 3D cell hash degenerated along one axis. The
    // map is 4096x2048 equirectangular and has been sitting in the C# project's
    // textures folder the whole time -- SkyManager.cs:361 loads exactly this
    // file and turns it into a cubemap. We can sample it directly.
    if (night > 0.001) {
        // Rotated about an axis tilted from vertical, so the sky wheels the way
        // it does at a mid latitude rather than spinning flat about the zenith:
        // "tilt it and then rotate it on that axis as if we were at forty five
        // degrees latitude".
        const float kLatitude = 0.7854;
        float3 axis = normalize(float3(0.0, cos(kLatitude), sin(kLatitude)));
        float angle = atan2(sunDir.x, -sunDir.z);
        float ca = cos(angle), sa = sin(angle);
        float3 sd = direction * ca + cross(axis, direction) * sa
                  + axis * dot(axis, direction) * (1.0 - ca);

        float2 uv = float2(atan2(sd.z, sd.x) / 6.2831853 + 0.5,
                           acos(clamp(sd.y, -1.0, 1.0)) / 3.14159265);
        // SampleLevel, not Sample. The uv comes out of an atan2, whose
        // derivative jumps a whole turn at the seam behind the camera -- and a
        // Sample picks its mip from that derivative, so the entire sky was
        // being read from a tiny mip and came out as a blurry smear.
        float3 stars = texture.SampleLevel(uv, 0.0).rgb;
        // Lifted, not dimmed. The map is genuinely dark -- mean brightness 3.5
        // out of 255 and a peak of 57 -- so it needs multiplying up to be seen
        // at all. Squaring it first, which is what a bright photograph would
        // want, made it invisible.
        color += stars * night * 4.0 * smoothstep(-0.05, 0.15, direction.y);
    }

    // The horizon glow that makes a sunrise legible.
    //
    // SkyManager.cs samples its sunset cubemap with
    //     t = dir[axis] - pow(abs(dir.y), 2)
    // which is what keeps the colour pinned to the horizon instead of smeared up
    // the dome. Same shape here, aimed along the sun rather than an axis.
    float3 sunFlat = normalize(float3(sunDir.x, 0.0, sunDir.z) + float3(1e-5, 0.0, 0.0));
    float toward = dot(normalize(float3(direction.x, 0.0, direction.z) + float3(1e-5, 0.0, 0.0)),
                       sunFlat);
    float glowShape = saturate(toward - direction.y * direction.y * 2.0);
    float lowSun = 1.0 - saturate(abs(sunElev) * 3.2);
    float3 glowColor = lerp(float3(0.95, 0.42, 0.18), float3(1.0, 0.72, 0.38), day);
    color += glowColor * pow(glowShape, 3.0) * lowSun * 0.85;

    // Clouds: big and puffy, with a hard edge.
    //
    // The first version was fbm faded smoothly by density, which reads as mist.
    // What is wanted is the C#'s look -- pixelated, and either the cloud is
    // there or it is not, with no soft gradient in between. So the noise is
    // quantised on to a grid and then thresholded, which gives a chunky edge
    // that matches a game made of voxels instead of fighting it.
    if (direction.y > 0.02) {
        float time = ubo.shadowFadeWidth.y;
        float2 plane = direction.xz / direction.y;
        float2 drift = float2(time * 0.004, time * 0.0016);

        // Quantised before the noise is read, which is what makes the edge
        // blocky rather than smooth. Coarser far away is wrong -- the cells are
        // in plane space, so they get smaller toward the horizon on their own,
        // exactly like the voxels do.
        const float kCell = 0.05;
        float2 cellular = floor((plane * 0.22 + drift) / kCell) * kCell;
        float shape = fbm(cellular) * 0.7 + fbm(cellular * 0.36 - drift) * 0.5;

        // Hard cutoff: in or out, one pixel wide so it does not alias to death.
        // Threshold high enough that the sky is mostly sky. At 0.575 a single
        // mass swallowed a whole quarter of it.
        float cover = smoothstep(0.655, 0.668, shape);
        cover *= smoothstep(0.02, 0.09, direction.y); // still nothing at the seam

        // Lit from the sun's side. Fully lit or fully shadowed, no gradient --
        // the same argument as the edge.
        float toSun = saturate(dot(normalize(float3(direction.x, 0.0, direction.z)
                                             + float3(1e-5, 0.0, 0.0)), sunFlat));
        float sunlit = smoothstep(0.35, 0.55, toSun);
        float3 litCloud = lerp(float3(0.72, 0.74, 0.80), float3(1.0, 0.99, 0.96), sunlit);
        float3 duskCloud = lerp(float3(0.30, 0.22, 0.30), float3(1.0, 0.60, 0.30), sunlit);
        float3 nightCloud = float3(0.06, 0.07, 0.11);
        float3 cloudColor = litCloud * day + duskCloud * twilight + nightCloud * night;

        color = lerp(color, cloudColor, cover);
    }

    // The sun itself.
    const float kSunCosRadius = 0.99965;
    const float kSunEdge = 0.00035;
    float sun = smoothstep(kSunCosRadius - kSunEdge, 1.0, dot(direction, sunDir));
    float3 sunColor = min(ubo.lightColor.rgb, float3(1.0, 1.0, 1.0));
    color = lerp(color, max(sunColor, float3(1.0, 0.95, 0.85)), sun);

    return float4(color, 1.0);
}
