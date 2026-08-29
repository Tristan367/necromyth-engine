#include "lib/frame_uniforms.slang"
// Shares the engine-wide push constant and instance layouts: background
// geometry goes through the same draw list and gets an instance record like
// everything else, so it must agree on both.
#include "lib/textured_push_constants.slang"
#include "lib/instance_data.slang"

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

    // Stars, before the sun and the horizon glow go over the top.
    //
    // Rotated about an axis tilted from vertical, so the field wheels the way it
    // does at a mid latitude rather than spinning flat about the zenith. Asked
    // for explicitly: "tilt it and then rotate it on that axis as if we were at
    // forty five degrees latitude".
    if (night > 0.001) {
        const float kLatitude = 0.7854; // 45 degrees, in radians
        float3 axis = float3(0.0, cos(kLatitude), sin(kLatitude));
        // Sidereal angle taken from the sun's own rotation about the same axis,
        // so the stars and the day agree without another uniform.
        float angle = atan2(sunDir.x, -sunDir.z);
        float ca = cos(angle), sa = sin(angle);
        float3 sd = direction * ca + cross(axis, direction) * sa
                  + axis * dot(axis, direction) * (1.0 - ca);
        const float kStarDensity = 260.0;
        float3 cell = floor(sd * kStarDensity);
        float star = hash31(cell);
        // A few are brighter than the rest, which is most of what makes a sky
        // read as stars rather than as noise.
        float magnitude = hash31(cell + 7.77);
        float bright = smoothstep(0.9955, 0.9995, star) * (0.35 + 0.65 * magnitude);
        // Nothing below the horizon, and a soft edge so they do not pop on.
        bright *= smoothstep(-0.01, 0.22, sd.y);
        color += float3(bright) * night * 1.1;
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

    // Clouds, on a plane above the world.
    //
    // Projected by dividing the view direction by its height, which is what
    // puts them on a flat ceiling and makes them crowd together toward the
    // horizon the way real cloud cover does. Nothing above the horizon line is
    // drawn, and the whole layer fades out as it approaches it, because a plane
    // seen edge-on is infinitely far away and would otherwise be a hard seam.
    if (direction.y > 0.02) {
        float time = ubo.shadowFadeWidth.y;
        float2 plane = direction.xz / direction.y;
        float2 drift = float2(time * 0.004, time * 0.0016);
        float cover = fbm(plane * 0.55 + drift);
        // A second, slower layer, so the cover has depth rather than being one
        // sheet of noise.
        cover = cover * 0.65 + fbm(plane * 0.19 - drift * 0.6) * 0.45;

        float density = smoothstep(0.52, 0.78, cover);
        density *= smoothstep(0.02, 0.20, direction.y);   // fade into the horizon

        // Lit from the sun's side, and much darker underneath at night.
        float toSun = saturate(dot(normalize(float3(direction.x, 0.0, direction.z)
                                             + float3(1e-5, 0.0, 0.0)), sunFlat));
        float3 litCloud = lerp(float3(0.62, 0.63, 0.68), float3(1.0, 0.97, 0.92), toSun);
        float3 duskCloud = lerp(float3(0.35, 0.24, 0.28), float3(1.0, 0.62, 0.34), toSun);
        float3 nightCloud = float3(0.05, 0.06, 0.10);
        float3 cloudColor = litCloud * day + duskCloud * twilight + nightCloud * night;

        color = lerp(color, cloudColor, density * 0.9);
    }

    // The sun itself.
    const float kSunCosRadius = 0.99965;
    const float kSunEdge = 0.00035;
    float sun = smoothstep(kSunCosRadius - kSunEdge, 1.0, dot(direction, sunDir));
    float3 sunColor = min(ubo.lightColor.rgb, float3(1.0, 1.0, 1.0));
    color = lerp(color, max(sunColor, float3(1.0, 0.95, 0.85)), sun);

    return float4(color, 1.0);
}
