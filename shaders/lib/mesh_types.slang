struct VSInput {
    float3 inPosition;
    float3 inNormal;
    float3 inColor;
    float2 inTexCoord;
};

struct VSOutput {
    float4 pos : SV_Position;
    float3 worldNormal;
    float3 worldPos;
    float3 fragColor;
    float2 fragTexCoord;
};
