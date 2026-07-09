// Shared 4-weight linear blend skinning helper.
// Used by: triangle_skinned.slang, shadow_depth_skinned.slang, point_shadow.slang.
float4x4 computeSkinMatrix(VSInputSkinned input, StructuredBuffer<float4x4> jointMatrices) {
    return input.inJointWeights.x * jointMatrices[(int)input.inJointIndices.x] +
           input.inJointWeights.y * jointMatrices[(int)input.inJointIndices.y] +
           input.inJointWeights.z * jointMatrices[(int)input.inJointIndices.z] +
           input.inJointWeights.w * jointMatrices[(int)input.inJointIndices.w];
}
