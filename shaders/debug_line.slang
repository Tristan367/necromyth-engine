#include "lib/frame_uniforms.slang"

struct VSInput {
    float3 inPos : POSITION;
    float4 inColor : COLOR;
};

struct VSOutput {
    float4 pos : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pos = mul(ubo.proj, mul(ubo.view, float4(input.inPos, 1.0)));
    output.color = input.inColor;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    return input.color;
}
