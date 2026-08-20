// Shared 4-weight linear blend skinning helper.
// Used by: triangle_skinned.slang, shadow_depth_skinned.slang, point_shadow.slang.
//
// `boneBase` is the instance's first matrix in the shared palette. The palette
// holds every skinned instance in the scene, so the joint index from the vertex
// is relative and has to be offset -- without this every character would read
// the first instance's pose.
float4x4 computeSkinMatrix(VSInputSkinned input, StructuredBuffer<float4x4> jointMatrices, uint boneBase) {
    return input.inJointWeights.x * jointMatrices[boneBase + (uint)input.inJointIndices.x] +
           input.inJointWeights.y * jointMatrices[boneBase + (uint)input.inJointIndices.y] +
           input.inJointWeights.z * jointMatrices[boneBase + (uint)input.inJointIndices.z] +
           input.inJointWeights.w * jointMatrices[boneBase + (uint)input.inJointIndices.w];
}
