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

  This compute shader computes the build arguments and vertices
  for template instantiations of a tessellated triangle region.

  The details of the tessellation are provided via `build.partTriangles`

  Depending on the value of TESS_INSTANTIATE_BATCHSIZE
  a single workgroup may operate on multiple tessellated
  triangles at once.This helps with improving utilization of
  low-tessellated triangles.
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
#extension GL_EXT_nonuniform_qualifier : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require

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

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;  
};

layout(scalar, binding = BINDINGS_TESSTABLE_UBO, set = 0) uniform tessTableBuffer
{
  TessellationTable tessTable;  
};

#if HAS_DISPLACEMENT_TEXTURES
layout(binding = BINDINGS_DISPLACED_TEXTURES, set = 0) uniform sampler2D displacementTextures[];
#endif

////////////////////////////////////////////

layout(local_size_x=CLUSTER_TEMPLATE_INSTANTIATE_WORKGROUP) in;

#define SUBGROUP_COUNT (CLUSTER_TEMPLATE_INSTANTIATE_WORKGROUP/SUBGROUP_SIZE)

////////////////////////////////////////////

#include "tessellation.glsl"
#if TESS_USE_PN || DO_ANIMATION
#include "displacement.glsl"
#endif

////////////////////////////////////////////


shared uint s_vertexOffset[TESS_INSTANTIATE_BATCHSIZE];
#if TESS_INSTANTIATE_BATCHSIZE > 1
shared uint s_taskOffsets[TESS_INSTANTIATE_BATCHSIZE];
shared uint s_tasks[TESS_INSTANTIATE_BATCHSIZE];
shared uint s_numThreads;
#endif

////////////////////////////////////////////

void main()
{
#if TESS_INSTANTIATE_BATCHSIZE > 1
  uint partIndex =     gl_WorkGroupID.x * TESS_INSTANTIATE_BATCHSIZE + gl_SubgroupInvocationID;
  uint partLoad  = min(gl_WorkGroupID.x * TESS_INSTANTIATE_BATCHSIZE + min(gl_SubgroupInvocationID,TESS_INSTANTIATE_BATCHSIZE-1), build.partTriangleCounter-1);
#else
  uint partIndex = gl_WorkGroupID.x;
  uint partLoad  = partIndex;
#endif

  //if(true) return;

  //if(gl_WorkGroupID.x > 0) return;

  TessTriangleInfo tessInfo = build.partTriangles.d[partLoad];

  uvec3 vtxEncoded = tessInfo.subTriangle.vtxEncoded;
  uint instanceID  = tessInfo.cluster.instanceID;
  uint clusterID   = tessInfo.cluster.clusterID;
  uint triangleID_config = tessInfo.subTriangle.triangleID_config;
  uint triangleID  = triangleID_config & 0xFFFF;
  uint cfg         = triangleID_config >> 16;
  

  uint numVertices =  partLoad == partIndex ? tess_getConfigVertexCount(cfg) : 0;
  uint vertexOffset = ~0;
  
  // first subgroup in workgroup handles template instantiation setup for all
  // partial triangle regions
  if (gl_LocalInvocationID.x < TESS_INSTANTIATE_BATCHSIZE && partLoad == partIndex)
  {  
    uint genOffset         = atomicAdd(buildRW.genClusterCounter, 1);
    uint64_t dataSize      = uint64_t(tessTable.templateInstantiationSizes.d[tess_getConfigIndex(cfg)]);
    uint64_t dataOffset    = atomicAdd(buildRW.genClusterDataCounter, dataSize);
    vertexOffset           = atomicAdd(buildRW.genVertexCounter, numVertices);
    
    // test if we have enough space to perform the instantiation
    if (  (vertexOffset + numVertices > MAX_GENERATED_VERTICES) || 
          (genOffset + 1 > MAX_GENERATED_CLUSTERS) ||
          (dataOffset + dataSize > (uint64_t(MAX_GENERATED_CLUSTER_MEGS) * 1024 * 1024)))
    {
      vertexOffset = ~0;
      numVertices  = 0;
    }
    else
    {
      TemplateInstantiateInfo tempInfo;
    
      // given we have multiple kinds of clusters (full, partial etc.) we need to tag the
      // clusterID so that the hit shader will later know where to get information from.
      // the template's clusterID is zero
      tempInfo.clusterIdOffset        = partIndex | (RT_CLUSTER_MODE_SINGLE_TESSELLATED << 30);
      tempInfo.geometryIndexOffset    = 0;
      tempInfo.clusterTemplateAddress = tessTable.templateAddresses.d[tess_getConfigIndex(cfg)];
      // vertices are provided through a memory region that we fill every frame
      tempInfo.vertexBufferAddress    = uint64_t(build.genVertices) + uint64_t(vertexOffset * 4 * 3);
      tempInfo.vertexBufferStride     = 4 * 3;
      
      uint tempOffset = atomicAdd(buildRW.tempInstantiateCounter, 1); // actual template instantiations
      
      build.tempInstantiations.d[tempOffset]  = tempInfo;
      // need to know which instance the instantiated CLAS belongs to, so we can insert it into its BLAS
      build.tempInstanceIDs.d[tempOffset]     = instanceID;
      // we instantiate in explicit mode, so we provide the CLAS destination address
      build.tempClusterAddresses.d[tempOffset] = uint64_t(build.genClusterData) + dataOffset;
      
      uint numTriangles = tess_getConfigTriangleCount(cfg);
    
      // increment the instance's BLAS number of CLAS it will reference
      // we need this to later build per-BLAS arrays of references
      atomicAdd(build.blasBuildInfos.d[instanceID].clusterReferencesCount, 1);
      // for stats
      atomicAdd(readback.numTotalTriangles, numTriangles);
    }
  #if TESS_INSTANTIATE_BATCHSIZE < 2
    s_vertexOffset[0] = vertexOffset;
  #endif
  }
  
#if TESS_INSTANTIATE_BATCHSIZE > 1

  // when we batched multiple into one workgroup, then we need to
  // make the information handled by the first subgroup available to
  // all threads, via shared memory

  if (gl_LocalInvocationID.x < TESS_INSTANTIATE_BATCHSIZE)
  {
    uint threadOffset = subgroupInclusiveAdd(numVertices);
    uint numThreads   = subgroupShuffle(threadOffset, TESS_INSTANTIATE_BATCHSIZE-1);
    uvec4 voteTasks   = subgroupBallot(numVertices != 0);
    
    s_numThreads = numThreads;
    
    uint taskOffset = subgroupBallotExclusiveBitCount(voteTasks);
    
    s_taskOffsets[gl_LocalInvocationID.x] = ~0;
    
    memoryBarrierShared();
    
    if (numVertices != 0)
    {
      s_taskOffsets[taskOffset]  = threadOffset - numVertices;
      s_tasks[taskOffset]        = gl_SubgroupInvocationID;
      s_vertexOffset[taskOffset] = vertexOffset;
    }
  }
  
#endif

  
  memoryBarrierShared();
  barrier();
  
#if TESS_INSTANTIATE_BATCHSIZE > 1

  uint numThreads = s_numThreads;

  if (numThreads == 0) return;
  
  uint  in_instanceID         = instanceID;
  uint  in_clusterID          = clusterID;
  uint  in_triangleID_config  = triangleID_config;
  uvec3 in_vtxEncoded         = vtxEncoded;

  // distribute the filling of vertices of multiple tasks across
  // the workgroup's subgroups
  
  // ensure full subgroups are executed for shuffle
  uint numThreadsRun = ((numThreads + SUBGROUP_SIZE - 1) & ~(SUBGROUP_SIZE-1));
  for (uint t = gl_LocalInvocationID.x; t < numThreadsRun; t += CLUSTER_TEMPLATE_INSTANTIATE_WORKGROUP)
  {
    uint start   = 0;
    uint taskID  = s_tasks[0];
    vertexOffset = s_vertexOffset[0];
    
    // binary search wasn't faster
    [[unroll]] for (uint i = 1; i < TESS_INSTANTIATE_BATCHSIZE; i++)
    {
      uint taskOffset = s_taskOffsets[i];
      if (t >= taskOffset) 
      {
        taskID       = s_tasks[i];
        vertexOffset = s_vertexOffset[i];
        start        = taskOffset;
      }
    }
    
    triangleID_config = subgroupShuffle(in_triangleID_config, taskID);
    instanceID        = subgroupShuffle(in_instanceID,        taskID);
    clusterID         = subgroupShuffle(in_clusterID,         taskID);
    vtxEncoded        = subgroupShuffle(in_vtxEncoded,        taskID);
    triangleID        = triangleID_config & 0xFFFF;
    cfg               = triangleID_config >> 16;
    
    if (t >= numThreads) continue;
    
    uint vert = t - start;
    
#else
  // simpler setup we know we only have one task at a time

  vertexOffset = s_vertexOffset[0];
  
  // ran out of memory
  if (vertexOffset == ~0) return;
  
  {
#endif

    RenderInstance instance = instances[instanceID];
    Cluster cluster         = instance.clusters.d[clusterID];
  
    vec3 baseBarycentrics[3];
    
    {
      // get vertices
      [[unroll]] for (uint v = 0; v < 3; v++) {
        uint vtxEncoded     = vtxEncoded[v];
        baseBarycentrics[v] = tess_decodeBarycentrics(vtxEncoded);
      }
    }
    
    vec3s_in  oPositions     = vec3s_in(instance.positions);
    vec3s_in  oNormals       = vec3s_in(instance.normals);
    vec2s_in  oTexcoords     = vec2s_in(instance.texcoords);
    uint8s_in localTriangles = uint8s_in(instance.clusterLocalTriangles);

    mat4 worldMatrix   = instance.worldMatrix;
    mat3 worldMatrixIT = transpose(inverse(mat3(worldMatrix)));
    
    uvec3 baseIndices = uvec3(localTriangles.d[cluster.firstLocalTriangle + triangleID * 3 + 0],
                              localTriangles.d[cluster.firstLocalTriangle + triangleID * 3 + 1],
                              localTriangles.d[cluster.firstLocalTriangle + triangleID * 3 + 2])
                        + uint(cluster.firstLocalVertex);
    
    vec3 basePositions[3];
    vec3 baseNormals[3];
    vec2 baseTexcoords[3];
    
    {
      // get vertices
      [[unroll]] for (uint v = 0; v < 3; v++) {
        basePositions[v]  = oPositions.d[baseIndices[v]];
        baseNormals[v]    = normalize(oNormals.d[baseIndices[v]]);
        baseTexcoords[v]  = oTexcoords.d[baseIndices[v]];
      }
    }
    
  #if TESS_USE_PN
    DeformBasePN basePN;
    deform_setupPN(basePN, basePositions, baseNormals);
  #endif

#if TESS_INSTANTIATE_BATCHSIZE > 1
    // the more sophisticated distribution of vertices of multiple tasks already makes us operate
    // on a per-vertex level at this point
#else
    // otherwise we switch to iteration over all vertices of the single tessellated triangle region we operate on
    for (uint vert = gl_LocalInvocationID.x; vert < numVertices; vert += CLUSTER_TEMPLATE_INSTANTIATE_WORKGROUP)
#endif
    {
      vec3 vertexBarycentrics = tess_getConfigVertexBarycentrics(cfg, vert);
      
      vertexBarycentrics = tess_interpolate(baseBarycentrics, vertexBarycentrics);

    #if TESS_USE_PN
      vec3 oPos    = deform_getPN(basePN, vertexBarycentrics);
    #else
      vec3 oPos    = tess_interpolate(basePositions, vertexBarycentrics);
    #endif
    #if HAS_DISPLACEMENT_TEXTURES
      if (instance.displacementIndex >= 0)
      {
        vec3 oNormal = tess_interpolate(baseNormals, vertexBarycentrics);
        vec2  uv     = tess_interpolate(baseTexcoords, vertexBarycentrics);
        float height = texture(displacementTextures[nonuniformEXT(instance.displacementIndex)], uv).r;
        height = (height * instance.displacementScale * view.displacementScale) + instance.displacementOffset + view.displacementOffset;
        oPos += normalize(oNormal) * height;
      }
    #endif
    #if DO_ANIMATION
      oPos = rippleDeform(oPos, instanceID, instance.geoHi.w);
    #endif
      
      // these are the vertices the template instantiation will use
      build.genVertices.d[vert + vertexOffset] = oPos;
    }
  }
}