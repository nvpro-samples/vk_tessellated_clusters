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
  
  This compute shader does basic operations on a single thread.
  For example clamping atomic counters back to their limits or
  setting up indirect dispatches or draws etc.
  
  BUILD_SETUP_... are enums for the various operations

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
  uint buildSetup;
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

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) coherent buffer buildBufferRW
{
  SceneBuilding buildRW;  
};

layout(scalar, binding = BINDINGS_TESSTABLE_UBO, set = 0) uniform tessTableBuffer
{
  TessellationTable tessTable;  
};

////////////////////////////////////////////

layout(local_size_x=1) in;

////////////////////////////////////////////

#include "build.glsl"

////////////////////////////////////////////

void main()
{  
  // special values for dispatch indirect setups
  if (push.buildSetup == BUILD_SETUP_CLASSIFY)
  {

    const uint maxEntries = MAX_VISIBLE_CLUSTERS;
    uint counter = buildRW.visibleClusterCounter;
    readback.numVisibleClusters = counter;
    
    counter = min(counter, maxEntries);

    buildRW.visibleClusterCounter  = counter;
    buildRW.dispatchClassify.gridX = counter;
    buildRW.dispatchClassify.gridY = 1;
    buildRW.dispatchClassify.gridZ = 1;

  }
  else if (push.buildSetup == BUILD_SETUP_DRAW_TESS)
  {

    const uint maxEntriesFull = MAX_VISIBLE_CLUSTERS;
    const uint maxEntriesPart = MAX_PART_TRIANGLES;
    uint counterFull = buildRW.fullClusterCounter;
    uint counterPart = buildRW_partTriangleCounter();
    
    readback.numFullClusters   = counterFull;
    readback.numPartTriangles  = counterPart;
    readback.numSplitTriangles = buildRW.splitWriteCounter;
    
    counterFull = min(counterFull, maxEntriesFull);
    counterPart = min(counterPart, maxEntriesPart);

    buildRW.drawFullClusters.first  = 0;
    buildRW.drawFullClusters.count  = counterFull;
#if TESS_RASTER_USE_BATCH
    buildRW.drawPartTriangles.first = 0;
    buildRW.drawPartTriangles.count = (counterPart + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;
    
    readback.numBlasClusters = counterFull;
#else
    readback.numBlasClusters = counterFull + counterPart;
    
    buildRW.drawPartTriangles.first = 0;
    buildRW.drawPartTriangles.count = counterPart;
#endif

  }
  else if (push.buildSetup == BUILD_SETUP_SPLIT)
  {

    buildRW.splitWriteCounter = buildRW.splitTriangleCounter;
    buildRW.partTriangleCounter = buildRW_partTriangleCounter();

  }
  else if (push.buildSetup == BUILD_SETUP_BUILD_BLAS)
  {

    const uint maxEntries = MAX_GENERATED_CLUSTERS;
    uint counterTemp = buildRW.tempInstantiateCounter;
  #if TESS_USE_TRANSIENTBUILDS
    uint counterTrans = buildRW.transBuildCounter;
  #endif
    
  #if TESS_USE_TRANSIENTBUILDS
    readback.numBlasClusters       = counterTemp + counterTrans;
    readback.numTransBuilds        = counterTrans;
  #else
    readback.numBlasClusters       = counterTemp;
  #endif
    readback.numTempInstantiations = counterTemp;
    readback.numGenDatas           = buildRW.genClusterDataCounter;
    readback.numGenVertices        = buildRW.genVertexCounter;
    readback.numBlasReservedSizes  = buildRW.numBlasReservedSizes;
    
    counterTemp  = min(maxEntries, counterTemp);
  #if TESS_USE_TRANSIENTBUILDS
    counterTrans = min(maxEntries, counterTemp + counterTrans) - counterTemp;
  #endif
    
    // used by vkCmdBuildClusterAccelerationStructureIndirectNVX instantiate
    buildRW.tempInstantiateCounter       = counterTemp;
    readback.numActualTempInstantiations = counterTemp;
    
    // template insert
    buildRW.dispatchBlasTempInsert.gridX = (counterTemp + CLUSTER_BLAS_INSERT_WORKGROUP - 1) / CLUSTER_BLAS_INSERT_WORKGROUP; 
    buildRW.dispatchBlasTempInsert.gridY = 1;
    buildRW.dispatchBlasTempInsert.gridZ = 1;

  #if TESS_USE_TRANSIENTBUILDS
    // used by vkCmdBuildClusterAccelerationStructureIndirectNVX build
    buildRW.transBuildCounter     = counterTrans;
    readback.numActualTransBuilds = counterTrans;
    
    // transient insert
    buildRW.dispatchBlasTransInsert.gridX = (counterTrans + CLUSTER_BLAS_INSERT_WORKGROUP - 1) / CLUSTER_BLAS_INSERT_WORKGROUP; 
    buildRW.dispatchBlasTransInsert.gridY = 1;
    buildRW.dispatchBlasTransInsert.gridZ = 1;
  #endif

  }
  else if (push.buildSetup == BUILD_SETUP_INSTANTIATE_TESS)
  {

    const uint maxEntriesPart = MAX_PART_TRIANGLES;
    uint counterPart = buildRW_partTriangleCounter();
    
  #if TESS_USE_TRANSIENTBUILDS
    uint counterPartTransient  = buildRW_partTriangleCounterTransient();
    readback.numPartTriangles  = counterPart + counterPartTransient;
    readback.numTransPartTriangles = counterPartTransient;
  #else
    readback.numPartTriangles  = counterPart;
  #endif
    readback.numSplitTriangles = buildRW.splitWriteCounter;
    
  #if TESS_USE_TRANSIENTBUILDS
    // we know the actual part triangles from this dedicated counter
    counterPart = buildRW.partTriangleCounter;
  #else
    counterPart = min(counterPart, maxEntriesPart);
    
    buildRW.partTriangleCounter = counterPart;
  #endif

    buildRW.dispatchTriangleInstantiate.gridX = (counterPart + TESS_INSTANTIATE_BATCHSIZE - 1) / TESS_INSTANTIATE_BATCHSIZE;
    buildRW.dispatchTriangleInstantiate.gridY = 1;
    buildRW.dispatchTriangleInstantiate.gridZ = 1;

  }
}