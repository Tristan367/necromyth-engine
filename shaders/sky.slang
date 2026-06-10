#include "lib/frame_uniforms.slang"

struct VSInput {
    float3 inPosition;
};

struct PushConstants {
    float4x4 model;
};
[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;

struct VSOutput {
    float4 pos : SV_Position;
    float3 direction;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.direction = input.inPosition;

    float4 world = mul(pushConstants.model, float4(input.inPosition, 1.0));
    float4 viewPos = mul(ubo.viewSky, world);
    float4 clipPos = mul(ubo.proj, viewPos);
    clipPos.z = clipPos.w;
    output.pos = clipPos;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target {
    float3 direction = normalize(vertIn.direction);
    float height = direction.y * 0.5 + 0.5;

    float3 zenith = float3(0.18, 0.35, 0.72);
    float3 horizon = float3(0.55, 0.62, 0.78);
    float3 ground = float3(0.08, 0.09, 0.12);

    float3 sky = lerp(horizon, zenith, saturate(height * 1.35));
    float3 color = lerp(ground, sky, smoothstep(0.08, 0.42, height));

    const float kSunCosRadius = 0.99965;
    const float kSunEdge = 0.00035;
    float3 sunDir = normalize(ubo.lightDirection.xyz);
    float sunDot = dot(direction, sunDir);
    float sun = smoothstep(kSunCosRadius - kSunEdge, 1.0, sunDot);
    float3 sunColor = min(ubo.lightColor.rgb, float3(1.0, 1.0, 1.0));
    color = lerp(color, sunColor, sun);

    return float4(color, 1.0);
}
