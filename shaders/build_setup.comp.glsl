/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#if USE_EXT_MESH_SHADER
    // EXT launch grids are limited to 16 bit per dimension and a total count.
    // The 2D grid can overshoot, the mesh/task shaders test against the clamped counters.
    counterFull = min(counterFull, MESHSHADER_MAX_WORKGROUPS);
  #if TESS_RASTER_USE_BATCH
    counterPart = min(counterPart, MESHSHADER_MAX_WORKGROUPS * SUBGROUP_SIZE);
  #else
    counterPart = min(counterPart, MESHSHADER_MAX_WORKGROUPS);
  #endif
    buildRW.fullClusterCounter  = counterFull;
    buildRW.partTriangleCounter = counterPart;

    uvec3 grid = fit16bitLaunchGrid(counterFull);
    buildRW.drawFullClustersEXT.gridX = grid.x;
    buildRW.drawFullClustersEXT.gridY = grid.y;
    buildRW.drawFullClustersEXT.gridZ = grid.z;
  #if TESS_RASTER_USE_BATCH
    grid = fit16bitLaunchGrid((counterPart + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE);
  #else
    grid = fit16bitLaunchGrid(counterPart);
  #endif
    buildRW.drawPartTrianglesEXT.gridX = grid.x;
    buildRW.drawPartTrianglesEXT.gridY = grid.y;
    buildRW.drawPartTrianglesEXT.gridZ = grid.z;
#endif

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
    uint count = min(buildRW.splitTriangleCounter, MAX_SPLIT_TRIANGLES);
    buildRW.splitWriteCounter    = count;
    buildRW.splitTriangleCounter = int(count);
    buildRW.partTriangleCounter  = buildRW_partTriangleCounterClamped();
  #if !TESS_USE_PERSISTENT_KERNEL
    buildRW.splitPassStart = 0;
    buildRW.splitPassEnd   = count;

    buildRW.dispatchTriangleSplit.gridX = (count + TRIANGLE_SPLIT_WORKGROUP - 1) / TRIANGLE_SPLIT_WORKGROUP;
    buildRW.dispatchTriangleSplit.gridY = 1;
    buildRW.dispatchTriangleSplit.gridZ = 1;

    // readback.debugA[0] = count;

  #endif
  }
  else if (push.buildSetup == BUILD_SETUP_SPLIT_PASS)
  {
  #if !TESS_USE_PERSISTENT_KERNEL
    uint pass = buildRW.splitPass + 1;
    buildRW.splitPass = pass;

    // begin at last end
    uint start = min(buildRW.splitPassEnd, MAX_SPLIT_TRIANGLES);
    // end at current write position
    uint end   = min(buildRW.splitWriteCounter, MAX_SPLIT_TRIANGLES);

    buildRW.splitPassStart = start;
    buildRW.splitPassEnd   = end;

    uint count = end - start;

    buildRW.dispatchTriangleSplit.gridX = (count + TRIANGLE_SPLIT_WORKGROUP - 1) / TRIANGLE_SPLIT_WORKGROUP;
    buildRW.dispatchTriangleSplit.gridY = 1;
    buildRW.dispatchTriangleSplit.gridZ = 1;

    // readback.debugA[pass] = count;
  #endif
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
    
  #if TESS_USE_LOWTESS_PARTS
    uint counterPartLowtess       = buildRW_partTriangleCounterLowtess();
    readback.numPartTriangles  = counterPart + counterPartLowtess;
    readback.numTransPartTriangles = counterPartLowtess;
  #else
    readback.numPartTriangles  = counterPart;
  #endif
    readback.numSplitTriangles = buildRW.splitWriteCounter;
    
  #if TESS_USE_LOWTESS_PARTS
    // we know the actual part triangles from this dedicated counter,
    // it was clamped against the information stored at the back of the array
    counterPart = min(buildRW.partTriangleCounter, buildRW_partTriangleCounterClamped());
  #else
    counterPart = min(counterPart, maxEntriesPart);
    
    buildRW.partTriangleCounter = counterPart;
  #endif

    buildRW.dispatchTriangleInstantiate.gridX = (counterPart + TESS_INSTANTIATE_BATCHSIZE - 1) / TESS_INSTANTIATE_BATCHSIZE;
    buildRW.dispatchTriangleInstantiate.gridY = 1;
    buildRW.dispatchTriangleInstantiate.gridZ = 1;

  }
}