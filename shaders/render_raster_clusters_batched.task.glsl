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
  
  Shader Description
  ==================

  This task shader performs packing multiple tessellated triangles
  into batches for the mesh shader.

  When we have triangles of low tessellation
  it would be quite wasteful to emit them as a single mesh
  shader workgroup which pre-allocates the worst-case space 
  a single tessellated triangle may have than needed.

  That is why we batch a few into this space.

*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable

#extension GL_NV_mesh_shader : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require

#include "shaderio.h"

layout(push_constant) uniform pushData
{
  uint instanceID;
}
push;

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar,binding=BINDINGS_READBACK_SSBO,set=0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBuffer
{
  SceneBuilding buildRW;  
};

layout(scalar, binding = BINDINGS_TESSTABLE_UBO, set = 0) uniform tessTableBuffer
{
  TessellationTable tessTable;  
};

////////////////////////////////////////////

#include "tessellation.glsl"

////////////////////////////////////////////

out taskNV TaskExchange {
  uint16_t  batchStartCount[SUBGROUP_SIZE];
  uint16_t  prefixsumTriangles[SUBGROUP_SIZE];
  uint16_t  prefixsumVertices[SUBGROUP_SIZE];
  uint      baseIndex;
} TASK;

////////////////////////////////////////////

layout(local_size_x = SUBGROUP_SIZE) in;

void main()
{
  // task shaders operate on SUBGROUP_SIZE granularity

  // figure out which tessellated partial triangles
  // this workgroup operates on.
  uint partIndex = gl_WorkGroupID.x * SUBGROUP_SIZE + gl_SubgroupInvocationID;
  uint partTotalCount = build.partTriangleCounter;
  uint partLocalCount = min(partTotalCount, gl_WorkGroupID.x * SUBGROUP_SIZE + SUBGROUP_SIZE) - gl_WorkGroupID.x * SUBGROUP_SIZE;
  
  // grab the data
  TessTriangleInfo tessInfo = build.partTriangles.d[partIndex];
  
  uint cfg          = 0;
  uint numVertices  = TESS_RASTER_BATCH_VERTICES;
  uint numTriangles = TESS_RASTER_BATCH_TRIANGLES;
  
  // get the tessellation level for this triangle
  // and the number of vertices/triangles it would use
  if (partIndex < partTotalCount)
  {
    cfg          = tessInfo.subTriangle.triangleID_config >> 16;
    numVertices  = tess_getConfigVertexCount(cfg);
    numTriangles = tess_getConfigTriangleCount(cfg);
  }
  
  // compute prefix sum over all vertices and triangles
  uint sumVertices   = subgroupInclusiveAdd(numVertices);
  uint sumTriangles  = subgroupInclusiveAdd(numTriangles);
  TASK.prefixsumVertices [gl_SubgroupInvocationID] = uint16_t(sumVertices - numVertices);
  TASK.prefixsumTriangles[gl_SubgroupInvocationID] = uint16_t(sumTriangles - numTriangles);

  // next up we must build the batches, we simply do a linear fill
  // until we fit within TESS_RASTER_BATCH_VERTICES and TESS_RASTER_BATCH_TRIANGLES limits
  //
  // The algorithm computes the sum vertices/triangles across all threads in the the subgroup
  // and tests if they stay within limit. The last thread that fits in the current batch
  // is found.
  // We then writes the batch information (from which part triangle it starts to end) to 
  // task shader output data (TASK).
  //
  // If there are part triangles left, we repeat the process with those
  // triangles left.
  
  uint batchIndex = 0;

  uint lastBatchStart     = 0;
  uint lastBatchVertices  = 0;
  uint lastBatchTriangles = 0;
  
  uint left = partLocalCount;
  uint i = 0;
  while(left != 0 && batchIndex < SUBGROUP_SIZE)
  {
    // compute per-thread an inclusive prefix sum how many triangles/vertices the current batch would use
    // note: negative numbers may occur, but become large positive numbers that will then report
    // that they don't fit. this way we implicitly skip over already processed part triangle threads.
    
    uint batchVertices  = uint(sumVertices  - lastBatchVertices);
    uint batchTriangles = uint(sumTriangles - lastBatchTriangles);
  
    // find the highest thread that has a valid configuration where both vertices and triangles fit
    uvec4 voteFit  = subgroupBallot(batchVertices <= uint(TESS_RASTER_BATCH_VERTICES) && batchTriangles <= uint(TESS_RASTER_BATCH_TRIANGLES));
    uint batchEnd  = subgroupBallotFindMSB(voteFit);

    // Example:
    //   gl_SubgroupInvocationID   | 0 1 2 3 4 5 6 7 8 9 ...

    //   in the first iteration we are able to fit 4 part triangles
    //   first subgroupBallot      | x x x x - - - - - - ...
    //   first batchStart          | 0
    //   first batchEnd            | 3

    //   in the second we fit the next 3
    //   invocations 0..3 would create negative numbers -> become large positive in uint -> exceed the limits
    //   second subgroupBallot     | - - - - x x x - - - ...
    //   second batchStart         | 4
    //   second batchEnd           | 6
    
    // keep track of current batch info
    // batch count is how many part triangles are within this batch
    uint batchStart = lastBatchStart;
    uint batchCount = 1 + batchEnd - batchStart;
    
    if (gl_SubgroupInvocationID == 0){
      TASK.batchStartCount[batchIndex] = uint16_t(batchStart | (batchCount << 8));
    }
    
    // prepare next iteration
    lastBatchStart     = 1 + batchEnd;
    lastBatchVertices  = subgroupShuffle(sumVertices,  batchEnd);
    lastBatchTriangles = subgroupShuffle(sumTriangles, batchEnd);
    
    batchVertices  = subgroupShuffle(batchVertices,  batchEnd);
    batchTriangles = subgroupShuffle(batchTriangles, batchEnd);
    
    left -= min(batchCount, left);
    
    batchIndex++;
  }
  
  if (gl_SubgroupInvocationID == 0){
    TASK.baseIndex = gl_WorkGroupID.x * SUBGROUP_SIZE;
    atomicAdd(readback.numBlasClusters, batchIndex);
    gl_TaskCountNV = batchIndex;
  }
}