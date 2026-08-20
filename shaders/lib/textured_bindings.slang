[[vk::binding(0, 1)]]
Sampler2D texture;
[[vk::binding(1, 0)]]
Sampler2DArray textureArray;

float4 sampleBaseColor(float2 uv, uint textureLayer, uint sampleTextureArray) {
    if (sampleTextureArray != 0)
        return textureArray.Sample(float3(uv, textureLayer));
    return texture.Sample(uv);
}
