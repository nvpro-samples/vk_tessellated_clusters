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
  
  This compute shader does basic basic culling of all clusters.
  Occlusion and frustum culling can be activated.

  It fills `build.visibleClusters`

  A single thread represents one cluster.

  The sample uses a very basic approach going over all clusters,
  even if we might have detected that an instance has already been
  culled.
  It would be better (but a bit more complex) to handle this differently
  so we don't need to iterate over clusters known to be culled.

  The "vk_lod_cluster" sample implements a more sophisticated
  scene traversal logic.

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

#extension GL_EXT_control_flow_attributes : require
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
  FrameConstants viewLast;
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

////////////////////////////////////////////

layout(local_size_x=CLUSTERS_CULL_WORKGROUP) in;

////////////////////////////////////////////
 
#include "culling.glsl"

////////////////////////////////////////////

void main()
{
  const uint maxEntries = MAX_VISIBLE_CLUSTERS;
  
  //if (gl_WorkGroupID.x > 0) return;

  uint clusterID  = gl_GlobalInvocationID.x;
  uint instanceID = push.instanceID;
  
  RenderInstance instance = instances[instanceID];
  uint instanceState      = build.instanceStates.d[instanceID];
  
  uint clusterLoad = min(clusterID, instance.numClusters -1);
  BBox clusterBbox = instance.clusterBboxes.d[clusterLoad];
  
  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;
  
  bool visible = clusterLoad == clusterID;
  
  // DEBUG
  //visible = visible && (clusterID == 1);
  
#if DO_CULLING && TARGETS_RASTERIZATION
  if ((instanceState & INSTANCE_VISIBLE_BIT) == 0)
    visible = false;

  visible = visible && intersectFrustum(clusterBbox.lo, clusterBbox.hi, instance.worldMatrix, clipMin, clipMax, clipValid);
  
  visible = visible && (!clipValid || (intersectSize(clipMin, clipMax) && intersectHiz(clipMin, clipMax)));
#endif
  
  uvec4 visibleVote   = subgroupBallot(visible);
  uint  visibleCount  = subgroupBallotBitCount(visibleVote);
  uint  visibleOffset = subgroupBallotExclusiveBitCount(visibleVote);
  
  uint atomicOffset = 0;
  if (gl_SubgroupInvocationID == 0 && visibleCount > 0)
  {
    atomicOffset = atomicAdd(buildRW.visibleClusterCounter, visibleCount);
  }
  atomicOffset = subgroupBroadcastFirst(atomicOffset);
  
  
  if (visible && (atomicOffset + visibleOffset) < maxEntries)
  {
    build.visibleClusters.d[atomicOffset + visibleOffset] = ClusterInfo(instanceID, clusterID);
  }
}

