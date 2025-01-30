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

  This mesh shader renders a tessellated triangle.

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
layout(max_vertices = TESSTABLE_MAX_VERTICES, max_primitives = TESSTABLE_MAX_TRIANGLES) out;
layout(triangles) out;

////////////////////////////////////////////

#include "tessellation.glsl"
#if TESS_USE_PN || DO_ANIMATION
#include "displacement.glsl"
#endif
////////////////////////////////////////////

void main()
{
  TessTriangleInfo tessInfo = build.partTriangles.d[gl_WorkGroupID.x];

  uint instanceID = tessInfo.cluster.instanceID;
  uint clusterID  = tessInfo.cluster.clusterID;
  uint triangleID = tessInfo.subTriangle.triangleID_config & 0xFFFF;
  uint cfg        = tessInfo.subTriangle.triangleID_config >> 16;
  
  TessTableEntry entry    = tessTable.entries.d[tess_getConfigIndex(cfg)];
  RenderInstance instance = instances[instanceID];
  Cluster cluster         = instance.clusters.d[clusterID];

  uint numVertices  = uint(entry.numVertices);
  uint numTriangles = uint(entry.numTriangles);

  if (gl_LocalInvocationID.x == 0){
    gl_PrimitiveCountNV = numTriangles;
    // just for stats
    atomicAdd(readback.numTotalTriangles, tess_getConfigTriangleCount(cfg));
  }

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
  
  uint partID = 0;
  
  {
    // get vertices
    [[unroll]] for (uint v = 0; v < 3; v++) {
      uint vtxTemp        = tessInfo.subTriangle.vtxEncoded[v];
      basePositions[v]    = oPositions.d[baseIndices[v]];
      baseNormals[v]      = normalize(oNormals.d[baseIndices[v]]);
      baseTexcoords[v]    = oTexcoords.d[baseIndices[v]];
      baseBarycentrics[v] = tess_decodeBarycentrics(vtxTemp);
      
      // just for debug coloring
      partID ^= (vtxTemp >> 20) | ((vtxTemp >> 4) & 0xFFF);
    }
  }
   
#if TESS_USE_PN
  DeformBasePN basePN;
  deform_setupPN(basePN, basePositions, baseNormals);
#endif

  for (uint vert = gl_LocalInvocationID.x; vert < numVertices; vert += MESHSHADER_WORKGROUP_SIZE)
  {
    vec3 vertexBarycentrics = tess_getConfigVertexBarycentrics(cfg, vert);
    
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

  for (uint tri = gl_LocalInvocationID.x; tri < numTriangles; tri += MESHSHADER_WORKGROUP_SIZE)
  {
    uvec3 indices = tess_getConfigTriangleVertices(cfg, tri);
    
    partID = view.visualize == VISUALIZE_TRIANGLES ? triLocal + 1 : partID;

    gl_PrimitiveIndicesNV[tri * 3 + 0] = indices.x;
    gl_PrimitiveIndicesNV[tri * 3 + 1] = indices.y;
    gl_PrimitiveIndicesNV[tri * 3 + 2] = indices.z;
    gl_MeshPrimitivesNV[tri].gl_PrimitiveID = int((triangleID & 0xFF) | ((partID | 1) << 8));
  }
}