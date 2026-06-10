[[vk::binding(0, 1)]]
Sampler2D texture;
[[vk::binding(1, 0)]]
Sampler2DArray textureArray;

float4 sampleBaseColor(float2 uv) {
    if (pushConstants.sampleTextureArray != 0)
        return textureArray.Sample(float3(uv, pushConstants.textureArrayLayer));
    return texture.Sample(uv);
}
