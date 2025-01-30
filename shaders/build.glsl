/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

/*
  When building transient clusters we store meta information at the end
  of the build.partTriangles array, while "real" tessellation info
  for triangles are stored at the front.

  These utilities allow us to implement this two-sided list

  a single u64 encodes
  lower 32bit:  actual partTriangles (same as if the feature wasn't used)
  higher 32bit: back of partTriangles array contains meta data for transient clusters

*/

// reading the counter is only relevant for actual partTriangles
uint buildRW_partTriangleCounter()
{
#if TESS_USE_TRANSIENTBUILDS
  return unpackUint2x32(buildRW.dualPartTriangleCounter).x;
#else
  return buildRW.partTriangleCounter;
#endif
}

uint buildRW_partTriangleCounterTransient()
{
#if TESS_USE_TRANSIENTBUILDS
 return unpackUint2x32(buildRW.dualPartTriangleCounter).y;
#else
  return 0;
#endif
}

#if TESS_USE_TRANSIENTBUILDS

uint build_atomicAdd_partTriangleCounterTransient(uint transientParts)
{
  uint64_t res = atomicAdd(buildRW.dualPartTriangleCounter, packUint2x32(uvec2(0,transientParts)));
  uvec2 unpacked = unpackUint2x32(res);
  
  // sum not within limit
  // return an out of bounds offset that will be rejected by code that uses the returned offset
  if (unpacked.x + unpacked.y + transientParts + 1 > MAX_PART_TRIANGLES)
    return MAX_PART_TRIANGLES;
  else
    return uint(MAX_PART_TRIANGLES) - unpacked.y - transientParts;
}
#endif

uint build_atomicAdd_partTriangleCounter(uint partTriangles)
{
#if TESS_USE_TRANSIENTBUILDS
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