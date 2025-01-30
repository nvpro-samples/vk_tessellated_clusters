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
  
  This compute shader inserts the CLAS clusters that should be rendered
  into the cluster references list for each instance's BLAS.

  A single thread represents one CLAS

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

////////////////////////////////////////////

layout(local_size_x=CLUSTER_BLAS_INSERT_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
#if TESS_USE_TRANSIENTBUILDS
  const bool doTemplates = push.instanceID == 0;
#else
  const bool doTemplates = true;
#endif
 
  uint rtclusterID = gl_GlobalInvocationID.x;
 
  // build.genClusterCounter was reset to be safe within bounds, gets value of build.blasClusterCounter
  // in indirect_build_setup-INDIRECT_SETUP_BUILD_BLAS
    
  uint counter = doTemplates ? build.tempInstantiateCounter : build.transBuildCounter;
  
  if (rtclusterID < counter)
  {
    uint instanceID;
    uint clusterSize;
    uint64_t clusterAddress;
    
    if (doTemplates)
    {
      instanceID     = build.tempInstanceIDs.d[rtclusterID];
      clusterAddress = build.tempClusterAddresses.d[rtclusterID];
      clusterSize    = build.tempClusterSizes.d[rtclusterID];
    }
    else
    {
      instanceID     = build.transInstanceIDs.d[rtclusterID];
      clusterAddress = build.transClusterAddresses.d[rtclusterID];
      clusterSize    = build.transClusterSizes.d[rtclusterID];
    }
    
    uint idx = atomicAdd(build.blasBuildInfos.d[instanceID].clusterReferencesCount,1);
    uint64s_inout clusterReferences = uint64s_inout(build.blasBuildInfos.d[instanceID].clusterReferences);
    clusterReferences.d[idx] = clusterAddress;
    atomicAdd(readback.numGenActualDatas, uint64_t(clusterSize));
  }
}