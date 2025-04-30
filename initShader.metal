#include <metal_stdlib>

using namespace metal;

// Must exactly match the host code.
constant uint BLOCK_DIM = 256;
constant uint TEST_SIZE = 65535;

kernel void init(uint3 id [[thread_position_in_grid]],
              uint3 griddim [[threadgroups_per_grid]],
              device uint* scan_bump [[buffer(0)]],
              device uint* scan [[buffer(1)]],
              device uint* errors [[buffer(2)]]) {
  // Clear the scan bump
  if (!id.x) {
    *scan_bump = 0;
  }

  // Clear scan buffer
  for (uint i = id.x; i < TEST_SIZE * 2; i += griddim.x * BLOCK_DIM) {
    scan[i] = 0;
  }

  // Clear error buffer
  for (uint i = id.x; i < TEST_SIZE * 4; i += griddim.x * BLOCK_DIM) {
    errors[i] = 0;
  }
}