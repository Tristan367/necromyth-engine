// Per-draw instance records (set 0, binding 8).
//
// The model matrix and material selection used to arrive in push constants,
// which meant one draw call could describe exactly one object. Reading them from
// a buffer indexed by instance lets N objects that share a mesh and material
// collapse into a single drawIndexed -- a horde, a forest, or a field of
// impostor billboards becomes one draw call instead of N.
//
// Must match engine::GpuInstance (renderer/instance_buffer.hpp), 80 bytes.

static const uint kNoBoneBase = 0xFFFFFFFFu;

struct GpuInstance {
    float4x4 model;
    uint textureLayer;
    uint sampleTextureArray;
    uint boneBase;   // first bone matrix of this instance's palette slice
    uint pad;
};

[[vk::binding(8, 0)]]
StructuredBuffer<GpuInstance> instanceBuffer;

// SV_InstanceID is zero-based within the draw, so the batch's first record index
// comes from a push constant rather than from firstInstance -- which Slang and
// Vulkan disagree about often enough not to rely on.
GpuInstance loadInstance(uint instanceId) {
    return instanceBuffer[pushConstants.instanceBase + instanceId];
}
