/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  When building transient clusters, or when instantiating combined templates,
  we store the low tessellation information at the end
  of the build.partTriangles array, while "real" tessellation info
  for triangles are stored at the front.

  These utilities allow us to implement this two-sided list

  a single u64 encodes
  lower 32bit:  actual partTriangles (same as if the feature wasn't used)
  higher 32bit: back of partTriangles array contains the data for those clusters

*/

// reading the counter is only relevant for actual partTriangles
uint buildRW_partTriangleCounter()
{
#if TESS_USE_LOWTESS_PARTS
  return unpackUint2x32(buildRW.dualPartTriangleCounter).x;
#else
  return buildRW.partTriangleCounter;
#endif
}

uint buildRW_partTriangleCounterLowtess()
{
#if TESS_USE_LOWTESS_PARTS
 return unpackUint2x32(buildRW.dualPartTriangleCounter).y;
#else
  return 0;
#endif
}

// The front of the array can only grow until it meets the information
// that is stored at the back. Both counters may exceed the limits when we
// run out of space, hence clamp the number of usable part triangles.
uint buildRW_partTriangleCounterClamped()
{
#if TESS_USE_LOWTESS_PARTS
  uvec2 unpacked = unpackUint2x32(buildRW.dualPartTriangleCounter);
  return min(unpacked.x, uint(MAX_PART_TRIANGLES) - min(unpacked.y, uint(MAX_PART_TRIANGLES)));
#else
  return min(buildRW.partTriangleCounter, uint(MAX_PART_TRIANGLES));
#endif
}

#if TESS_USE_LOWTESS_PARTS

uint build_atomicAdd_partTriangleCounterLowtess(uint lowtessParts)
{
  uint64_t res = atomicAdd(buildRW.dualPartTriangleCounter, packUint2x32(uvec2(0,lowtessParts)));
  uvec2 unpacked = unpackUint2x32(res);
  
  // sum not within limit
  // return an out of bounds offset that will be rejected by code that uses the returned offset
  if (unpacked.x + unpacked.y + lowtessParts + 1 > MAX_PART_TRIANGLES)
    return MAX_PART_TRIANGLES;
  else
    return uint(MAX_PART_TRIANGLES) - unpacked.y - lowtessParts;
}
#endif

uint build_atomicAdd_partTriangleCounter(uint partTriangles)
{
#if TESS_USE_LOWTESS_PARTS
  uint64_t res = atomicAdd(buildRW.dualPartTriangleCounter, packUint2x32(uvec2(partTriangles,0)));
  uvec2 unpacked = unpackUint2x32(res);
  
  // sum not within limit
  // return an out of bounds offset that will be rejected by code that uses the returned offset
  if (unpacked.x + unpacked.y + partTriangles + 1 > MAX_PART_TRIANGLES)
    return MAX_PART_TRIANGLES;
  else
    return unpacked.x;
#else
  return atomicAdd(buildRW.partTriangleCounter,partTriangles);
#endif
}