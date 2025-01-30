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

  This compute shader handles the recursive splitting of tessellated triangles.

  A single thread can represent one splitting task, as well as a sub-triangle
  child of splitting operations.

  The shader is configured to be run using persistent threads. A fixed amount of 
  threads implement a producer/consumer queuing mechanism to handle the splitting.

  The producer/consumer queue is implemented by the following variables:
  
    - `build.splitTriangles` stores the items that are processed as linear array
    - `build.splitWriteCounter` is used to produce new items into the array
    - `build.splitReadCounter` is used to consume from the array
    - `build.splitTriangleCounter` tracks the total number of tasks in-flight.
      It wil be incremented when new tasks are enqueued, and decremented when they are consumed.
      When it reaches zero we will have no more work left to process and the kernel can complete.

  The queue is seeded within `cluster_classify.comp.glsl` whenever a triangle's subdivison
  was too high.
  
  Furthermore all sub-triangles that are to be rendered are output via:
    - `build.partTriangles` stores all partial sub-triangles that are to be rendered as linear array
    - `build.partTriangleCounter` is used to append the partial sub-triangles

  The split logic attempts to consume items and then tests if their children
  need further processing: further split triangle, or enqueuing into the list of renderable
  partial triangles.
  
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
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_nonuniform_qualifier : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require
  
////////////////////////

#include "shaderio.h"

layout(push_constant) uniform pushData
{
  uint instanceID;
} push;

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) coherent buffer buildBufferRW
{
  volatile SceneBuilding buildRW;  
};

layout(scalar, binding = BINDINGS_TESSTABLE_UBO, set = 0) uniform tessTableBuffer
{
  TessellationTable tessTable;  
};

////////////////////////////////////////////

layout(local_size_x=TRIANGLE_SPLIT_WORKGROUP) in;

#include "build.glsl"
#include "tessellation.glsl"

////////////////////////////////////////////

// work around compiler bug on older drivers not properly handling coherent & volatile
#define USE_ATOMIC_LOAD_STORE 1

////////////////////////////////////////////

// setup the base vertices of the sub-triangle so we can compute the tessellation factors
void fillBaseVertices(TessTriangleInfo tessInfo, out vec3 basePositions[3], out vec3 baseBarycentrics[3])
{
  uint instanceID = tessInfo.cluster.instanceID;
  uint clusterID  = tessInfo.cluster.clusterID;

  RenderInstance instance = instances[instanceID];
  Cluster cluster         = instance.clusters.d[clusterID];

  vec3s_in  oPositions     = vec3s_in(instance.positions);
  vec3s_in  oNormals       = vec3s_in(instance.normals);
  uint8s_in localTriangles = uint8s_in(instance.clusterLocalTriangles);
  
  uint triangleID = tessInfo.subTriangle.triangleID_config & 0xFFFF;

  uvec3 indices = uvec3(localTriangles.d[cluster.firstLocalTriangle + triangleID * 3 + 0],
                        localTriangles.d[cluster.firstLocalTriangle + triangleID * 3 + 1],
                        localTriangles.d[cluster.firstLocalTriangle + triangleID * 3 + 2]);
  
  indices += cluster.firstLocalVertex;

  mat4 worldMatrix = instance.worldMatrix;

  // get vertices
  [[unroll]] for (uint v = 0; v < 3; v++) {
    // TODO add deformation
    basePositions[v]    = (worldMatrix * vec4(oPositions.d[indices[v]],1)).xyz;
    uint vtxEncoded     = tessInfo.subTriangle.vtxEncoded[v];
    baseBarycentrics[v] = tess_decodeBarycentrics(vtxEncoded);
  }
}

uvec4 getFactors(vec3 basePositions[3], vec3 baseBarycentrics[3])
{
  vec3 wPositions[3];
  [[unroll]] for (uint v = 0; v < 3; v++) {
    wPositions[v] = tess_interpolate(basePositions, baseBarycentrics[v]);
  }
  
  uvec3 factors = uvec3(tess_getTessFactors(wPositions[0],
                                            wPositions[1],
                                            wPositions[2]));
                                            
  return uvec4(factors, max(max(factors.x,factors.y),factors.z));
}

// Computes the number of children (subdivided triangles) for an incoming sub-triangle.
// These children are then processed within `processSubTask` a few lines down
uint setupTask(inout TessTriangleInfo tessInfo, uint splitIndex, uint pass)
{
  // get vertices
  vec3 basePositions[3];
  vec3 baseBarycentrics[3];
  fillBaseVertices(tessInfo, basePositions, baseBarycentrics);
  
  uvec4 factors          = getFactors(basePositions, baseBarycentrics);
  
  uvec3 splitFactors = clamp((factors.xyz + TESSTABLE_SIZE - 1) / TESSTABLE_SIZE, uvec3(1), uvec3(8));
  uint cfg = tess_getConfig(splitFactors, tessInfo.subTriangle.vtxEncoded);
  
  tessInfo.subTriangle.triangleID_config &= 0x0000FFFF;
  tessInfo.subTriangle.triangleID_config |= cfg << 16;
  
  // number of sub-triangles of the subdivision pattern
  uint outCount = tess_getConfigTriangleCount(cfg);
  
  return outCount;
}

void processSubTask(const TessTriangleInfo subgroupTasks, uint taskID, uint taskSubID, bool isValid, uint threadReadIndex, uint pass)
{
  // This function handles the primary split work operating on a single child (sub-triangle) of the
  // parent input item (also sub-triangle).
  //
  // It will test if the new child requires further splitting or can be rendered as is.
  //
  // Each thread is a child (`taskSubID`) of an incoming split task (`taskID`).
  // All tasks are stored in registers across the subgroup within `subgroupTasks`.
  // we access what we need via shuffle.
  //
  // The last `taskSubID` may be repeated when `isValid == false`, to allow safe memory access
  // for reads.
  // `threadReadIndex` and `pass` are only meant to aid debugging
  
  // pull required data from subgroupTasks
  // enqueue new splits
  TessTriangleInfo tessInfo;
  tessInfo.cluster.clusterID             = subgroupShuffle(subgroupTasks.cluster.clusterID,  taskID);
  tessInfo.cluster.instanceID            = subgroupShuffle(subgroupTasks.cluster.instanceID, taskID);
  tessInfo.subTriangle.vtxEncoded        = subgroupShuffle(subgroupTasks.subTriangle.vtxEncoded, taskID);
  tessInfo.subTriangle.triangleID_config = subgroupShuffle(subgroupTasks.subTriangle.triangleID_config, taskID);


  // modify encoded vtx according to sub-triangle
  uint cfg = tessInfo.subTriangle.triangleID_config >> 16;
  
  // compute tessellation factor
  uvec4 factors;
  vec3 baseBarycentrics[3];
  if (isValid)
  {
    vec3 basePositions[3];
    fillBaseVertices(tessInfo, basePositions, baseBarycentrics);
    
    uvec3 vertexIndices = tess_getConfigTriangleVertices(cfg, taskSubID);
    [[unroll]] for (uint v = 0; v < 3; v++)
    {
      vec3 vertex = tess_getConfigVertexBarycentrics(cfg, vertexIndices[v]);
      
      precise vec3 rebased = baseBarycentrics[0] * vertex.x +
                             baseBarycentrics[1] * vertex.y +
                             baseBarycentrics[2] * vertex.z;
      tessInfo.subTriangle.vtxEncoded[v] = tess_encodeBarycentrics(rebased);
    }
  
    [[unroll]] for (uint v = 0; v < 3; v++)
    {
      baseBarycentrics[v] = tess_decodeBarycentrics(tessInfo.subTriangle.vtxEncoded[v]);
    }
    
    factors          = getFactors(basePositions, baseBarycentrics);
  }
  
  // throw into appropriate list (per-thread only one can be true)
  
  bool requiresSplitTess = isValid && factors.w >  TESSTABLE_SIZE;
  bool requiresPartTess  = isValid && factors.w <= TESSTABLE_SIZE;
  
  uvec4 voteSplit = subgroupBallot(requiresSplitTess);
  uvec4 votePart  = subgroupBallot(requiresPartTess);
  uint countSplit = subgroupBallotBitCount(voteSplit);
  uint countPart  = subgroupBallotBitCount(votePart);
  
  uint offsetSplit = 0;
  uint offsetPart  = 0;
  
  if (subgroupElect())
  {
    offsetSplit = atomicAdd(buildRW.splitWriteCounter, countSplit);
    atomicAdd(buildRW.splitTriangleCounter, int(countSplit));
    offsetPart  = build_atomicAdd_partTriangleCounter(countPart);
  }
  memoryBarrierBuffer();
  
  offsetSplit = subgroupBroadcastFirst(offsetSplit);
  offsetSplit += subgroupBallotExclusiveBitCount(voteSplit);

  offsetPart = subgroupBroadcastFirst(offsetPart);
  offsetPart += subgroupBallotExclusiveBitCount(votePart);
  
  if (requiresSplitTess && offsetSplit < MAX_SPLIT_TRIANGLES)
  {     
    // need to split again
  #if USE_ATOMIC_LOAD_STORE
    atomicStore(build.splitTriangles.d[offsetSplit].cluster.clusterID, tessInfo.cluster.clusterID, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
    atomicStore(build.splitTriangles.d[offsetSplit].cluster.instanceID, tessInfo.cluster.instanceID, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
    atomicStore(build.splitTriangles.d[offsetSplit].subTriangle.vtxEncoded.x, tessInfo.subTriangle.vtxEncoded.x, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
    atomicStore(build.splitTriangles.d[offsetSplit].subTriangle.vtxEncoded.y, tessInfo.subTriangle.vtxEncoded.y, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
    atomicStore(build.splitTriangles.d[offsetSplit].subTriangle.vtxEncoded.z, tessInfo.subTriangle.vtxEncoded.z, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
    atomicStore(build.splitTriangles.d[offsetSplit].subTriangle.triangleID_config, tessInfo.subTriangle.triangleID_config, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
  #else
    build.splitTriangles.d[offsetSplit] = tessInfo;
  #endif
    memoryBarrierBuffer();    
  }
  else if (requiresPartTess && offsetPart < MAX_PART_TRIANGLES)
  {
    // ready to be rendered
    uint cfg = tess_getConfig(factors.xyz, tessInfo.subTriangle.vtxEncoded);
    tessInfo.subTriangle.triangleID_config &= 0x0000FFFF;
    tessInfo.subTriangle.triangleID_config |= cfg << 16;
    build.partTriangles.d[offsetPart]  = tessInfo;
  #if TESS_USE_TRANSIENTBUILDS
    uint subMax = subgroupMax(offsetPart + 1);
    if (subgroupElect())
    {
      atomicMax(buildRW.partTriangleCounter, subMax);
    }
  #endif
  }
}

////////////////////////

// slow
// process split tasks one at a time, easier code to read
#define SPLIT_DISTRIBUTION_SIMPLE       0
// otherwise distribute multiple tasks across
// the subgroup iterating over virtual threads
#define SPLIT_DISTRIBUTION_BIT_SEARCH   1

#define SPLIT_DISTRIBUTION SPLIT_DISTRIBUTION_BIT_SEARCH

struct TaskInfo {
  uint taskID;
};

shared TaskInfo s_tasks[TRIANGLE_SPLIT_WORKGROUP];

void processAllSubTasks(inout TessTriangleInfo tessInfo, bool splitValid, int subCount, uint threadReadIndex, uint pass)
{
  // Distribute new work across subgroup.
  // Each task may have a variable number of threads to be run.
  // We pack them tightly over a minimum amount of subgroup iterations.
  //
  // `threadReadIndex` and `pass` are only meant to aid debugging
  

  // distribute new splits work across subgroup
  int endOffset    = subgroupInclusiveAdd(subCount);
  int startOffset  = endOffset - subCount;
  int totalThreads = subgroupShuffle(endOffset, SUBGROUP_SIZE-1);
  int totalRuns    = (totalThreads + SUBGROUP_SIZE-1) / SUBGROUP_SIZE;
  
  const uint subgroupOffset = gl_SubgroupID * gl_SubgroupSize;

  bool hasTask     = subCount > 0;
  uvec4 taskVote   = subgroupBallot(hasTask);
  uint taskCount   = subgroupBallotBitCount(taskVote);
  uint taskOffset  = subgroupBallotExclusiveBitCount(taskVote);
  
  if (hasTask) {
    s_tasks[subgroupOffset + taskOffset].taskID = gl_SubgroupInvocationID;
  }
  
  //memoryBarrierShared();
  
  memoryBarrier(gl_ScopeSubgroup,
                gl_StorageSemanticsShared,
                gl_SemanticsAcquireRelease);
  
  // two techniques, both iterate over all thread's tasks.
  // Each task can spawn a variable amount of children work threads.
  
#if SPLIT_DISTRIBUTION == SPLIT_DISTRIBUTION_SIMPLE
  // simple technique: do one thread's task at a time

  for (uint task = 0; task < taskCount; task++)
  {
    uint taskID = int(s_tasks[subgroupOffset + task].taskID);
    uint taskSubCount = subgroupShuffle(subCount, taskID);
  #if 0
    // only relevant for debugging
    uint taskReadIndex = subgroupShuffle(threadReadIndex, taskID); 
  #else
    uint taskReadIndex = 0;
  #endif
    
    uint tRuns = (taskSubCount + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;
    for (uint tRun = 0; tRun < tRuns; tRun++)
    {
      uint t = gl_SubgroupInvocationID + tRun * SUBGROUP_SIZE;
      bool taskValid = t < taskSubCount;
      uint taskSubID = t;
      
      processSubTask(tessInfo, taskID, taskSubID, taskValid, taskReadIndex, pass);
    }
  }

#elif SPLIT_DISTRIBUTION == SPLIT_DISTRIBUTION_BIT_SEARCH

  // Following section contains a mechanism to distribute
  // work across the subgroup for better efficiency.
  // Rather than processing children in per-thread loops
  // we process the sum of all children iteratively across the subgroup.
  //
  // imagine three threads with 4,2,1 children
  // looping individually means we may get poor SIMT utilization.
  // 
  //  T0 T1 T2
  // ----------
  //  A0 B0 C0
  //  A1 B1 
  //  A2
  //  A3
  //
  //  packing across subgroup
  //
  //  TO T1 T2 T3 T4 T5 T6
  // ---------------------
  //  A0 A1 A2 A3 B0 B1 C0
  
  // technique computes a total number of virtual worker threads
  // then iterates over all those. Within each iteration we check
  // which original task a thread belongs to.
  // After that we compute the relative offset to that task's start
  // which yields the child index / taskSubID.
  
  int taskBase = -1;
  for (int r = 0; r < totalRuns; r++)
  {
  
    // this algorithm is explained within `render_clusters_batched.mesh.glsl` in the vertex loop section.
  
    int tFirst = r * SUBGROUP_SIZE;
    int t      = tFirst + int(gl_SubgroupInvocationID);
    
    int  relStart     = startOffset - tFirst;
    // set bit where task starts if within current run
    uint startBits    = subgroupOr(splitValid && relStart >= 0 && relStart < 32 ? (1 << relStart) : 0);
    
    int  task         = bitCount(startBits & gl_SubgroupLeMask.x) + taskBase;
    uint taskID       = s_tasks[subgroupOffset + task].taskID;
    
    uint taskSubID    = t - subgroupShuffle(startOffset, taskID);
    uint taskSubCount = subgroupShuffle(subCount, taskID);
  #if 0
    // only relevant for debugging
    uint taskReadIndex = subgroupShuffle(threadReadIndex, taskID); 
  #else
    uint taskReadIndex = 0;
  #endif
    taskBase          = subgroupShuffle(task, 31); // for next iteration
    
    bool taskValid    = taskSubID < taskSubCount;
    
    // do work
    processSubTask(tessInfo, taskID, taskSubID, taskValid, taskReadIndex, pass);
  }
  
#endif
}

////////////////////////////////////////////

void main()
{
  // This implements a persistent kernel that implements
  // a producer/consumer loop.
  //
  // special thanks to Robert Toth for the core setup.

  // the read index for the global array of `build.splitTriangles`
  uint threadReadIndex = ~0;
  
  for(uint pass = 0; ; pass++)
  {
    // try to consume
  
    // if entire subgroup has no work, acquire new work
  
    if (subgroupAll(threadReadIndex == ~0)) {    
      // pull new work
      if (subgroupElect()){
        threadReadIndex = atomicAdd(buildRW.splitReadCounter, SUBGROUP_SIZE);
      }
      threadReadIndex = subgroupBroadcastFirst(threadReadIndex) + gl_SubgroupInvocationID;
      threadReadIndex = threadReadIndex >= MAX_SPLIT_TRIANGLES ? ~0 : threadReadIndex;
      
      // if all read offsets are out of bounds, we are done for sure
      
      if (subgroupAll(threadReadIndex == ~0)){
        break;
      }
    }
  
    // let's attempt to fetch some valid work from the current state of `threadReadIndex`
  
    bool threadRunnable = false;
    TessTriangleInfo tessInfo;
    
    while(true)
    {
      if (threadReadIndex != ~0)
      {
        memoryBarrierBuffer();
        // get split info
      #if USE_ATOMIC_LOAD_STORE
        tessInfo.cluster.clusterID             = atomicLoad(build.splitTriangles.d[threadReadIndex].cluster.clusterID, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
        tessInfo.cluster.instanceID            = atomicLoad(build.splitTriangles.d[threadReadIndex].cluster.instanceID, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
        tessInfo.subTriangle.vtxEncoded.x      = atomicLoad(build.splitTriangles.d[threadReadIndex].subTriangle.vtxEncoded.x, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
        tessInfo.subTriangle.vtxEncoded.y      = atomicLoad(build.splitTriangles.d[threadReadIndex].subTriangle.vtxEncoded.y, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
        tessInfo.subTriangle.vtxEncoded.z      = atomicLoad(build.splitTriangles.d[threadReadIndex].subTriangle.vtxEncoded.z, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
        tessInfo.subTriangle.triangleID_config = atomicLoad(build.splitTriangles.d[threadReadIndex].subTriangle.triangleID_config, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
      #else
        tessInfo       = build.splitTriangles.d[threadReadIndex];
      #endif
        
        // reading is ahead of writing, might not have finished writing and value is still the cleared value
        threadRunnable = tess_isValid(tessInfo);
      }
      
      if (subgroupAny(threadRunnable))
        break;
      
      // Entire subgroup saw no valid work.
      // We always race ahead with reads compared to writes, but we may also
      // simply have no actual tasks left.
      
      memoryBarrierBuffer();
    #if USE_ATOMIC_LOAD_STORE
      bool isEmpty = atomicLoad(buildRW.splitTriangleCounter, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire) == 0;
    #else
      bool isEmpty = buildRW.splitTriangleCounter == 0;
    #endif
      if (subgroupAny(isEmpty))
      {
        return;
      }
    }
    
    // some threads have data ready to consume
    
    if (subgroupAny(threadRunnable))
    {      
      // each thread sets up a task with a variable number of children
      // this can be child nodes for an incoming node
      // or the clusters of an incoming group
      
      int threadSubCount = 0;
      
      if (threadRunnable)
      {
        threadSubCount = int(setupTask(tessInfo, threadReadIndex, pass));
      }
      
      // Now process all tasks, we do this in a packed fashion, so that
      // we attempt to fill the subgroup densely. This results in processing over work
      // in multiple iterations within the subgroup. As a result tasks may
      // straddle the subgroup.
      
      // we currently mix node/group tasks across the subgroup
      // this should be mostly okayish as both require traversal logic to be run
      // which depends on the same input data types.
      
      processAllSubTasks(tessInfo, threadRunnable, threadSubCount, threadReadIndex, pass);
      
      // All processed items need to decrement the global split task counter
      // and reset their complete state.
      
      uint numRunnable = subgroupBallotBitCount(subgroupBallot(threadRunnable));
      
      if (subgroupElect()) {
        atomicAdd(buildRW.splitTriangleCounter, -int(numRunnable));
      }
      
      if (threadRunnable) {
        // reset read index to invalid
        threadReadIndex = ~0;
      }
    }
  }
}