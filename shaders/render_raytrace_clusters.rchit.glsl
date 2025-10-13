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
  
  This hit shader handles the shading of clusters in
  ray tracing. 
  
  Note the use of a new input: `gl_ClusterIDNV`

  In our tessellation system we might create four different kinds
  of CLAS clusters that can be part of the scene and their kind is stored
  in the top bits of gl_ClusterIDNV.

  - RT_CLUSTER_MODE_FULL_CLUSTER: 
    A full cluster of the original model.
  - RT_CLUSTER_MODE_SINGLE_TESSELLATED:
    A cluster that represents a single tessellated region with a triangle (part triangle)
  - RT_CLUSTER_MODE_1X_SUBSET_CLUSTER
    A cluster that is a subset of a non-tessellated cluster (result of TESS_USE_1X_TRANSIENTBUILDS) 
  - RT_CLUSTER_MODE_2X_BATCHED_TESSELLATED
    A cluster that contains a batch of low-tessellated triangles (result of TESS_USE_2X_TRANSIENTBUILDS) 
  
*/

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_buffer_reference2 : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_ray_tracing_position_fetch : require

// at the time of writing, no GLSL extension was available, we leverage 
// GL_EXT_spirv_intrinsics to hook up the new builtin.
#extension GL_EXT_spirv_intrinsics : require

// Note that `VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV::allowClusterAccelerationStructure` must
// be set to `VK_TRUE` to make this valid.
spirv_decorate(extensions = ["SPV_NV_cluster_acceleration_structure"], capabilities = [5437], 11, 5436) in int gl_ClusterIDNV_;

// While not required in this sample, as we use dedicated hit-shader for clusters,
// `int gl_ClusterIDNoneNV = -1;` can be used to dynamically detect regular hits.

#include "shaderio.h"

/////////////////////////////////

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

#if TESS_ACTIVE
layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};
layout(scalar, binding = BINDINGS_TESSTABLE_UBO, set = 0) uniform tessTableBuffer
{
  TessellationTable tessTable;  
};
#endif

layout(set = 0, binding = BINDINGS_TLAS) uniform accelerationStructureEXT asScene;


/////////////////////////////////

hitAttributeEXT vec2 barycentrics;

/////////////////////////////////

layout(location = 0) rayPayloadInEXT RayPayload rayHit;
layout(location = 1) rayPayloadEXT RayPayload rayHitAO;

/////////////////////////////////

#if TESS_ACTIVE
#include "tessellation.glsl"
#endif


#define SUPPORTS_RT 1

#include "render_shading.glsl"

/////////////////////////////////

void main()
{
  // get cluster ID
  uint clusterID  = gl_ClusterIDNV_;
  uint triangleID = gl_PrimitiveID;
  
#if TESS_ACTIVE
  uint mode = (clusterID >> 30);
  bool isSpecial        = mode != RT_CLUSTER_MODE_FULL_CLUSTER;
  bool isTessTriangle   = mode == RT_CLUSTER_MODE_SINGLE_TESSELLATED;
  bool isSimpleTriangle = mode == RT_CLUSTER_MODE_1X_SUBSET_CLUSTER;
  bool isMiniTriangle   = mode == RT_CLUSTER_MODE_2X_BATCHED_TESSELLATED;

  // remove top mode bits
  clusterID &= 0x3FFFFFFF;

  TessTriangleInfo tessInfo;
  uint partID        = view.visualize == VISUALIZE_TRIANGLES ? triangleID + 1 : 0;;
  uint subTriangleID = triangleID;
  uint cfg           = 0;
  if(isSpecial)
  {
    tessInfo   = build.partTriangles.d[clusterID];

  #if TESS_USE_2X_TRANSIENTBUILDS
    if (isMiniTriangle)
    {
      // remap triangle ID
      uint packedTriangleID = uint16s_in(build.transTriMappings).d[(clusterID * (TessTriangleInfo_size/2) + (ClusterInfo_size/2)) + triangleID];
      // cluster triangle
      triangleID    = packedTriangleID & 0xff;
      // tessellated config triangle
      subTriangleID = (packedTriangleID >> 8) & 4;
      // get tessellation levels for local config
      tessInfo.subTriangle.vtxEncoded.x = tess_encodeBarycentrics(0,0);
      tessInfo.subTriangle.vtxEncoded.y = tess_encodeBarycentrics(TESSTABLE_COORD_MAX,0);
      tessInfo.subTriangle.vtxEncoded.z = tess_encodeBarycentrics(0,TESSTABLE_COORD_MAX);
      // derive config from int factors
      uvec3 intFactors = uvec3(1 + ((packedTriangleID >> 12) & 1), 1 + ((packedTriangleID >> 13) & 1), 1 + (packedTriangleID >> 14));
      
      cfg = tess_getConfig(intFactors, tessInfo.subTriangle.vtxEncoded);
      
      isTessTriangle = true;
    }
    else
  #endif
  #if TESS_USE_1X_TRANSIENTBUILDS
    if (isSimpleTriangle)
    {
      // remap triangle ID
      triangleID = build.transTriMappings.d[clusterID * TessTriangleInfo_size + ClusterInfo_size + triangleID];
    }
    else
  #endif
    {
      triangleID = tessInfo.subTriangle.triangleID_config & 0xFFFF;
      cfg        = tessInfo.subTriangle.triangleID_config >> 16;
    }
    clusterID    = tessInfo.cluster.clusterID;
  }
#endif

  // Fetch cluster header
  Cluster cluster       = instances[gl_InstanceID].clusters.d[clusterID];

  // Fetch triangle
  uint8s_in indexBuffer = uint8s_in(instances[gl_InstanceID].clusterLocalTriangles);
  vec3s_in oPositions   = vec3s_in(instances[gl_InstanceID].positions);
  vec3s_in oNormals     = vec3s_in(instances[gl_InstanceID].normals);
  
  uvec3 baseIndices     = uvec3(indexBuffer.d[triangleID * 3 + 0 + cluster.firstLocalTriangle],
                                indexBuffer.d[triangleID * 3 + 1 + cluster.firstLocalTriangle],
                                indexBuffer.d[triangleID * 3 + 2 + cluster.firstLocalTriangle]) 
                          +  uint(cluster.firstLocalVertex);

  vec3 baryWeight = vec3((1.f - barycentrics[0] - barycentrics[1]), barycentrics[0], barycentrics[1]);
  vec3 baryWeightBase = baryWeight;

#if TESS_ACTIVE
  
  if(isTessTriangle)
  {
    vec3 baseBarycentrics[3];
    partID = 0;
    [[unroll]] for (uint v = 0; v < 3; v++) {
      uint vtxEncoded   = tessInfo.subTriangle.vtxEncoded[v];
      partID ^= (vtxEncoded >> 20) | ((vtxEncoded >> 4) & 0xFFF);
      baseBarycentrics[v] = tess_decodeBarycentrics(vtxEncoded);
    }
    
    uvec3 tessTriIndices = tess_getConfigTriangleVertices(cfg, subTriangleID);
    
    // barycentrics from micro-triangle to sub-triangle
    vec3 newBaryWeight = 
      tess_getConfigVertexBarycentrics(cfg, tessTriIndices.x) * baryWeight.x +
      tess_getConfigVertexBarycentrics(cfg, tessTriIndices.y) * baryWeight.y +
      tess_getConfigVertexBarycentrics(cfg, tessTriIndices.z) * baryWeight.z;
      
    // barycentrics from sub-triangle to base-triangle
    baryWeightBase = 
      baseBarycentrics[0] * newBaryWeight.x +
      baseBarycentrics[1] * newBaryWeight.y +
      baseBarycentrics[2] * newBaryWeight.z;
      
    partID = view.visualize == VISUALIZE_TRIANGLES ? subTriangleID * 3: partID;
      
    partID = triangleID | ((partID | 1) << 8);
  }
#endif

  vec3 oPos = baryWeight.x * gl_HitTriangleVertexPositionsEXT[0] + baryWeight.y * gl_HitTriangleVertexPositionsEXT[1] + baryWeight.z * gl_HitTriangleVertexPositionsEXT[2];    
  vec3 wPos = vec3(gl_ObjectToWorldEXT * vec4(oPos, 1.0));

  vec3 oNrm;
#if ALLOW_VERTEX_NORMALS
  if(view.facetShading != 0)
#endif
  {
    // Otherwise compute geometric normal
    vec3 e0 = gl_HitTriangleVertexPositionsEXT[1] - gl_HitTriangleVertexPositionsEXT[0];
    vec3 e1 = gl_HitTriangleVertexPositionsEXT[2] - gl_HitTriangleVertexPositionsEXT[0];
    oNrm    = normalize(cross(e0, e1));
  }
#if ALLOW_VERTEX_NORMALS
  else
  {
    vec3     normals[3];

    [[unroll]] for(uint32_t i = 0; i < 3; i++)
    {
      normals[i] = normalize(oNormals.d[baseIndices[i]]);
    }
    oNrm = baryWeightBase.x * normals[0] + baryWeightBase.y * normals[1] + baryWeightBase.z * normals[2];
  }
#endif

  vec3 wNrm = normalize(vec3(oNrm * gl_WorldToObjectEXT));
  if(view.flipWinding != 0)
  {
    wNrm = -wNrm;
  }
  
  uint clusterShadeID = clusterID;
  if (view.visualize == VISUALIZE_TRIANGLES
#if TESS_ACTIVE
   || (isTessTriangle && view.visualize == VISUALIZE_TESSELLATED_CLUSTER)
#endif
      )
  {
    clusterShadeID ^= partID;
  }


  vec4 shaded = vec4(1.f);
  {
    float ambientOcclusion =
        ambientOcclusion(wPos, wNrm, view.ambientOcclusionSamples, view.ambientOcclusionRadius * view.sceneSize);

    float sunContribution  = 1.0;
    vec3  directionToLight = view.skyParams.sunDirection;
    if(view.doShadow == 1)
      sunContribution = traceShadowRay(wPos, directionToLight);

    shaded = shading(gl_InstanceID, wPos, wNrm, clusterShadeID, sunContribution, ambientOcclusion);
  }
#if TESS_ACTIVE
  if (view.visualize == VISUALIZE_TESSELLATED_TRIANGLES)
  {
    if (isTessTriangle)
    {
      shaded *= vec4(0.5,1,0.5,1);
    }
  }
#endif

#if DEBUG_VISUALIZATION

  if(view.doWireframe != 0 || (view.visFilterInstanceID == gl_InstanceID && view.visFilterClusterID == clusterID))
  {
    vec3 derivativeTargetX = gl_WorldToObjectEXT * vec4(gl_WorldRayOriginEXT + rayHit.color.xyz, 1);
    vec3 derivativeDirX    = derivativeTargetX.xyz - gl_ObjectRayOriginEXT;
    vec3 derivativeX = intersectRayTriangle(gl_ObjectRayOriginEXT, derivativeDirX, gl_HitTriangleVertexPositionsEXT[0], gl_HitTriangleVertexPositionsEXT[1], gl_HitTriangleVertexPositionsEXT[2]);
    derivativeX = abs(derivativeX - baryWeight);


    vec3 derivativeTargetY = gl_WorldToObjectEXT * vec4(gl_WorldRayOriginEXT + rayHit.differentialY.xyz, 1);
    vec3 derivativeDirY    = derivativeTargetY.xyz - gl_ObjectRayOriginEXT;
    vec3 derivativeY = intersectRayTriangle(gl_ObjectRayOriginEXT, derivativeDirY, gl_HitTriangleVertexPositionsEXT[0], gl_HitTriangleVertexPositionsEXT[1], gl_HitTriangleVertexPositionsEXT[2]);
    derivativeY = abs(derivativeY - baryWeight);

    vec3 derivative = max(derivativeX, derivativeY);

    rayHit.color.xyz = addWireframe(shaded.xyz, baryWeight, true, derivative, view.wireColor);
  }
  else
#endif
  {
    rayHit.color.xyz = shaded.xyz;
  }


  if(gl_LaunchIDEXT.xy == view.mousePosition)
  {
    vec4  projected            = (view.viewProjMatrix * vec4(wPos, 1.f));
    float depth                = projected.z / projected.w;
    readback.clusterTriangleId = packPickingValue((clusterID << 8) | triangleID, depth);
    readback.instanceId        = packPickingValue(gl_InstanceID, depth);
  }


  vec4 projPos   = view.viewProjMatrix * vec4(wPos, 1.f);
  rayHit.color.w = projPos.z / projPos.w;
}