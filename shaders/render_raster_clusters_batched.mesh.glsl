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

  This mesh shader performs the rasterization of partially
  tessellated triangles.

  It implements an optimization so that multiple partial
  triangles can be batched and rasterized within one mesh shader
  workgroup.

  `render_raster_clusters_batched.task.glsl` sets up the
  appropriate batching.
  
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
#extension GL_EXT_nonuniform_qualifier : enable

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

#if HAS_DISPLACEMENT_TEXTURES
layout(binding = BINDINGS_DISPLACED_TEXTURES, set = 0) uniform sampler2D displacementTextures[];
#endif

////////////////////////////////////////////

taskNV in TaskExchange {
  uint16_t  batchStartCount[SUBGROUP_SIZE];
  uint16_t  prefixsumTriangles[SUBGROUP_SIZE];
  uint16_t  prefixsumVertices[SUBGROUP_SIZE];
  uint     baseIndex;
} TASK;


////////////////////////////////////////////

layout(location = 0) out Interpolants
{
#if ALLOW_SHADING
  vec3      wPos;
#if ALLOW_VERTEX_NORMALS
  vec3      wNormal;
#endif
#endif
  flat uint clusterID;
  flat uint instanceID;
}
OUT[];

////////////////////////////////////////////

#ifndef MESHSHADER_WORKGROUP_SIZE
#define MESHSHADER_WORKGROUP_SIZE 32
#endif

layout(local_size_x = MESHSHADER_WORKGROUP_SIZE) in;
layout(max_vertices = TESS_RASTER_BATCH_VERTICES, max_primitives = TESS_RASTER_BATCH_TRIANGLES) out;
layout(triangles) out;

////////////////////////////////////////////

#include "tessellation.glsl"
#if TESS_USE_PN || DO_ANIMATION
#include "displacement.glsl"
#endif
////////////////////////////////////////////

void main()
{
  uint batchInfo  = TASK.batchStartCount[gl_WorkGroupID.x];
  uint batchStart = batchInfo & 0xFF;
  uint batchCount = batchInfo >> 8;
  
  uint taskRead  = min(gl_SubgroupInvocationID, batchCount-1);
  bool taskValid = gl_SubgroupInvocationID == taskRead;
  
  TessTriangleInfo tessInfo = build.partTriangles.d[TASK.baseIndex + batchStart + taskRead];
  
  uint baseNumVertices  = TASK.prefixsumVertices [batchStart];
  uint baseNumTriangles = TASK.prefixsumTriangles[batchStart];
  
  uint numTotalVertices  = TASK.prefixsumVertices [batchStart + batchCount - 1] - baseNumVertices; 
  uint numTotalTriangles = TASK.prefixsumTriangles[batchStart + batchCount - 1] - baseNumTriangles;
  
  // we have prefix sum, must add last element 
  {
    uint lastCfg = subgroupShuffle(tessInfo.subTriangle.triangleID_config >> 16, batchCount - 1);
    
    numTotalVertices  += tess_getConfigVertexCount(lastCfg);
    numTotalTriangles += tess_getConfigTriangleCount(lastCfg);
  }
  
  // rebase the offsets from this batch
  // the TASK. prefixsum is over all partial triangles
  // but we want a local prefix sum just over those that are processed in this batch

  int taskVertexStart   = int(TASK.prefixsumVertices [batchStart + taskRead] - baseNumVertices);
  int taskTriangleStart = int(TASK.prefixsumTriangles[batchStart + taskRead] - baseNumTriangles);
  
  if (gl_SubgroupInvocationID == 0)
  {
    gl_PrimitiveCountNV = numTotalTriangles;
    atomicAdd(readback.numTotalTriangles, numTotalTriangles);
    
    // We investigated if it was worth optimizing for batchCount == 1 
    // but in grand scheme of things the vast majority of batched clusters isn't that big.
    // A simple branch at top to use either this shaders code or the `render_raster_clusters_tess.mesh.glsl` code
    // didn't improve perf, we also could fill build.partTriangles from front/back depending on whether something is
    // big (> 64 vertices etc.). and then use both shaders for drawing.
    
    // if (batchCount == 1) atomicAdd(readback.debugA[0], 1);
    // else                 atomicAdd(readback.debugA[1], 1);
  }

  // our batch contains multiple tasks
  // 
  
  // vertex loop
  {
    int taskPreviousStartCount = -1;
    int iterations = int((numTotalVertices + SUBGROUP_SIZE - 1) / (SUBGROUP_SIZE));
    for (int iter = 0; iter < iterations; iter++)
    {
      // This algorithm does a sort of distributed search.
      // for each worker thread that this subgroup iterates over
      // we need to find the task it belongs to.
      
      // e.g. two iterations of subgroup size 32 yields 64 virtual threads
      // and we want to figure out their task assignment (A = 0, B = 1, C = 2)
      
      // 0 Thread:  0, 1, 2, 3, 4, 5, 6, 7, 8, ...
      // 0 Task:    A, A, A, A, A, A, A, A, A, ...
      // 1 Thread: 32,33,34,35,36,37,38,39,40, ...
      // 1 Task:    A, A, A, B, B, B, B, B, C, C, C,...
      
      
      // in the first iteration we get:
      // ------------------------------
      
      // the prefix sum for the 3 tasks A,B,C is
      //                            A,     B,    C
      //   taskVertexStart:         0,    35,   40
      //
      //                            A,     B,    C
      // 0 firstThread:             0,     0,    0
      // 0 taskRelativeStart:       0,    35,   40
      // 0 taskRelativeStartMask:   1<<0,  0,    0 
      // the above masks get or'ed into iterationStartMasks = 1

      //
      // 0 Thread:                     0, 1, 2, ...

      // 0 iterationStartMasks:        1, 1, 1, ...
      
      // 0 gl_SubgroupLeMask.x         1, 3, 7, ...
      //   bitCount(iterationStartMasks & gl_SubgroupLeMask.x)
      // 0 threadInclusiveStartCount:  1, 1, 1, ...
      // 0 taskPreviousStartCount     -1,-1,-1, ...
      // 0 taskIndex:                  0, 0, 0, ...
      // 0 Task:                       A, A, A, ....
      
      // in the second iteration we get:
      // ------------------------------
      
      // the prefix sum for the 3 tasks A,B,C is
      //                                A,    B,    C
      //   taskVertexStart:             0,   35,   40
      
      // 1 firstThread:                32,   32,   32
      // 1 taskRelativeStart:         -32,    3,    8
      // 1 taskRelativeStartMask:       0, 1<<3, 1<<8 
      // the above masks get or'ed into iterationStartMasks = 264

      //
      // 1 Thread:                     32,   33,   34,  35, 36, 37,  38,  39,  40,   41, ...
      // 1 iterationStartMasks:       264,  264,   ...
      
      // 1 gl_SubgroupLeMask.x          1,    3,    7,  15, 31, 63, 127, 255, 511, 1023, ...
      //   bitCount(iterationStartMasks & gl_SubgroupLeMask.x)
      // 1 threadInclusiveStartCount:   0,    0,    0,   1,  1,  1,   1,   1,   2,    2,
      // 1 taskPreviousStartCount       0,    0,   ...
      // 1 taskIndex:                   0,    0,    0,   1,  1,  1,   1,   1,   2,    2,
      // 1 Task:                        A,    A,    A,   B,  B,  B,   B,   B,   C,    C
      
      int  firstThread = iter * SUBGROUP_SIZE;
      int  thread = firstThread + int(gl_SubgroupInvocationID);
      
      int  taskRelativeStart = taskVertexStart - firstThread;
      // Set bit where task end's if within current iteration of subgroup threads.
      uint taskRelativeStartMask = taskValid && taskRelativeStart >= 0 && taskRelativeStart < 32 ? (1 << taskRelativeStart) : 0;
      uint iterationStartMasks   = subgroupOr(taskRelativeStartMask);
      // Count the number of starts that happened in this iteration up until our thread lane, 
      // including ourselves (that is why we had -1 for taskBase).
      // Then add the number of starts from previous iterations.
      // This gives us the the task index for this thread.
      int threadInclusiveStartCount = bitCount(iterationStartMasks & gl_SubgroupLeMask.x);
      int taskIndex = threadInclusiveStartCount + taskPreviousStartCount;
      
      // for next iteration update the count from the last subgroup lane
      taskPreviousStartCount = subgroupShuffle(taskIndex, 31);
      
      // get per-thread tess info based on taskIndex
      uint instanceID   = subgroupShuffle(tessInfo.cluster.instanceID, taskIndex);
      uint clusterID    = subgroupShuffle(tessInfo.cluster.clusterID, taskIndex);
      uvec3 vtxEncoded  = subgroupShuffle(tessInfo.subTriangle.vtxEncoded, taskIndex);
      uint triangleID_config = subgroupShuffle(tessInfo.subTriangle.triangleID_config, taskIndex);
      uint triangleID   = triangleID_config & 0xFFFF;
      uint cfg          = triangleID_config >> 16;
      
      // get vertex offset of the current task
      int  vertexStart  = subgroupShuffle(taskVertexStart, taskIndex);
      
      // convert to local index relative to this thread's task
      int vert      = thread;
      int vertLocal = vert - vertexStart ;
      
      // now let's do the actual work, computing the tessellated triangle's vertices
      
      TessTableEntry entry    = tessTable.entries.d[tess_getConfigIndex(cfg)];
      RenderInstance instance = instances[instanceID];
      Cluster cluster         = instance.clusters.d[clusterID];
      
      vec3s_in oPositions      = vec3s_in(instance.positions);
      vec3s_in oNormals        = vec3s_in(instance.normals);
      vec2s_in oTexcoords      = vec2s_in(instance.texcoords);
      uint8s_in localTriangles = uint8s_in(instance.clusterLocalTriangles);

      mat4 worldMatrix   = instance.worldMatrix;
      mat3 worldMatrixI  = inverse(mat3(worldMatrix));
      
      uvec3 baseIndices = uvec3(localTriangles.d[cluster.firstLocalTriangle + triangleID * 3 + 0],
                                localTriangles.d[cluster.firstLocalTriangle + triangleID * 3 + 1],
                                localTriangles.d[cluster.firstLocalTriangle + triangleID * 3 + 2])
                          + uint(cluster.firstLocalVertex);
      
      vec3 baseBarycentrics[3];
      vec3 basePositions[3];
      vec3 baseNormals[3];
      vec2 baseTexcoords[3];
      
      {
        // get vertices
        [[unroll]] for (uint v = 0; v < 3; v++) {
          uint vtxTemp        = vtxEncoded[v];
          basePositions[v]    = oPositions.d[baseIndices[v]];
          baseNormals[v]      = normalize(oNormals.d[baseIndices[v]]);
          baseTexcoords[v]    = oTexcoords.d[baseIndices[v]];
          baseBarycentrics[v] = tess_decodeBarycentrics(vtxTemp);
        }
      }
       
    #if TESS_USE_PN
      DeformBasePN basePN;
      deform_setupPN(basePN, basePositions, baseNormals);
    #endif
      
      if (vert < numTotalVertices)
      {
      
        vec3 vertexBarycentrics = tess_getConfigVertexBarycentrics(cfg, vertLocal);
        
        vertexBarycentrics = tess_interpolate(baseBarycentrics, vertexBarycentrics);

      #if TESS_USE_PN
        vec3 oPos    = deform_getPN(basePN, vertexBarycentrics);
      #else
        vec3 oPos    = tess_interpolate(basePositions, vertexBarycentrics);
      #endif
      
      #if ALLOW_VERTEX_NORMALS
        vec3 oNormal = tess_interpolate(baseNormals,   vertexBarycentrics);
        vec3 wNormal = oNormal * worldMatrixI;
      #endif
      #if HAS_DISPLACEMENT_TEXTURES
        if (instance.displacementIndex >= 0)
        {
        #if !ALLOW_VERTEX_NORMALS
          vec3 oNormal = tess_interpolate(baseNormals,   vertexBarycentrics);
        #endif
          vec2  uv     = tess_interpolate(baseTexcoords, vertexBarycentrics);
          float height = texture(displacementTextures[nonuniformEXT(instance.displacementIndex)], uv).r;
          height = (height * instance.displacementScale * view.displacementScale) + instance.displacementOffset + view.displacementOffset;
          oPos += normalize(oNormal) * height;
        }
      #endif
      #if DO_ANIMATION
        oPos = rippleDeform(oPos, instanceID, instance.geoHi.w);
      #endif
        vec3 wPos    = (worldMatrix * vec4(oPos,1.0)).xyz;

        gl_MeshVerticesNV[vert].gl_Position = view.viewProjMatrix * vec4(wPos,1);
      #if ALLOW_SHADING
        OUT[vert].wPos                      = wPos.xyz;
      #if ALLOW_VERTEX_NORMALS
        OUT[vert].wNormal                   = normalize(wNormal);
      #endif
      #endif
        OUT[vert].clusterID                 = clusterID;
        OUT[vert].instanceID                = instanceID;
      }
    }
  }

  // triangle loop
  {
    int taskPreviousStartCount = -1;
    int iterations = int((numTotalTriangles + SUBGROUP_SIZE - 1) / (SUBGROUP_SIZE));
    for (int iter = 0; iter < iterations; iter++)
    {
      
      // see above for detailed explanation
    
      int  firstThread = iter * SUBGROUP_SIZE;
      int  thread = firstThread + int(gl_SubgroupInvocationID);
      
      int  taskRelativeStart = taskTriangleStart - firstThread;
      // Set bit where task end's if within current iteration of subgroup threads.
      uint taskRelativeStartMask = taskValid && taskRelativeStart >= 0 && taskRelativeStart < 32 ? (1 << taskRelativeStart) : 0;
      uint iterationStartMasks   = subgroupOr(taskRelativeStartMask);
      // Count the number of starts that happened in this iteration up until our thread lane, 
      // including ourselves (that is why we had -1 for taskBase).
      // Then add the number of starts from previous iterations.
      // This gives us the the task index for this thread.
      int threadInclusiveStartCount = bitCount(iterationStartMasks & gl_SubgroupLeMask.x);
      int taskIndex = threadInclusiveStartCount + taskPreviousStartCount;
      
      // for next iteration update the count from the last subgroup lane
      taskPreviousStartCount = subgroupShuffle(taskIndex, 31);

      // get per-thread tess info
      uvec3 vtxEncoded  = subgroupShuffle(tessInfo.subTriangle.vtxEncoded, taskIndex);
      uint triangleID_config = subgroupShuffle(tessInfo.subTriangle.triangleID_config, taskIndex);
      uint triangleID   = triangleID_config & 0xFFFF;
      uint cfg          = triangleID_config >> 16;
      int  vertexStart  = subgroupShuffle(taskVertexStart, taskIndex);
      
      // convert to local index relative to this thread's task
      int tri      = thread;
      int triLocal = tri - subgroupShuffle(taskTriangleStart, taskIndex);
      
      uvec3 indices = tess_getConfigTriangleVertices(cfg, triLocal);
      
      uint partID = 0;
      
      // get vertices
      [[unroll]] for (uint v = 0; v < 3; v++) {
        uint vtxTemp        = vtxEncoded[v];
        // just for debug coloring
        partID ^= (vtxTemp >> 20) | ((vtxTemp >> 4) & 0xFFF);
      }
      
      if (tri < numTotalTriangles) 
      {
        partID = view.visualize == VISUALIZE_TRIANGLES ? triLocal * 3 : partID; // (((triLocal + 1) << 8) ^ partID)
      
        gl_PrimitiveIndicesNV[tri * 3 + 0] = indices.x + vertexStart;
        gl_PrimitiveIndicesNV[tri * 3 + 1] = indices.y + vertexStart;
        gl_PrimitiveIndicesNV[tri * 3 + 2] = indices.z + vertexStart;
        gl_MeshPrimitivesNV[tri].gl_PrimitiveID = int((triangleID & 0xFF) | ((partID | 1) << 8)); //tri
      }
    }
  }
}