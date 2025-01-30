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
#ifndef _SHADERIO_BUILDING_H_
#define _SHADERIO_BUILDING_H_

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
#else

#ifndef MAX_VISIBLE_CLUSTERS
#define MAX_VISIBLE_CLUSTERS 1024
#endif

#ifndef MAX_PART_TRIANGLES
#define MAX_PART_TRIANGLES 1024
#endif

#ifndef MAX_SPLIT_TRIANGLES
#define MAX_SPLIT_TRIANGLES 1024
#endif

#ifndef MAX_GENERATED_CLUSTERS
#define MAX_GENERATED_CLUSTERS 1024
#endif

#ifndef MAX_GENERATED_CLUSTER_MEGS
#define MAX_GENERATED_CLUSTER_MEGS 1024
#endif

#ifndef MAX_GENERATED_VERTICES
#define MAX_GENERATED_VERTICES 1024
#endif


/////////////////////////////////////////

#define INSTANCE_FRUSTUM_BIT 1
#define INSTANCE_VISIBLE_BIT 2

/////////////////////////////////////////

#endif

// A renderable cluster
struct ClusterInfo
{
  uint32_t instanceID;
  uint32_t clusterID;
};
BUFFER_REF_DECLARE_ARRAY(ClusterInfos_inout, ClusterInfo, , 4);
BUFFER_REF_DECLARE_SIZE(ClusterInfo_size, ClusterInfo, 8);

// Sub-triangle within a tessellated triangle, which itself is tessellated
struct SubTriangleInfo
{
  // position within triangle
  // 3 barycentric coordinates for each sub-triangle vertex
  // each 2 x 16 bit uv
  uvec3    vtxEncoded;
  
  // tessellation configuration
  // triangle id within cluster 16 bit
  // config 16 bit: 4 bits per edge (3), 1 bit flipped
  uint32_t triangleID_config;
};

// A renderable sub-triangle
struct TessTriangleInfo
{
  ClusterInfo     cluster;
  SubTriangleInfo subTriangle;
};
BUFFER_REF_DECLARE_ARRAY(TessTriangleInfos_inout, TessTriangleInfo, , 8);
BUFFER_REF_DECLARE_ARRAY(TessTriangleInfos_coh_volatile, TessTriangleInfo, volatile coherent, 8);
BUFFER_REF_DECLARE_SIZE(TessTriangleInfo_size, TessTriangleInfo, 24);

// Indirect build information to build a CLAS using the new device-side acceleration structure function
struct ClasBuildInfo
{
  uint32_t clusterID;
  uint32_t clusterFlags;

#define ClasBuildInfo_packed_triangleCount 0 : 9
#define ClasBuildInfo_packed_vertexCount 9 : 9
#define ClasBuildInfo_packed_positionTruncateBitCount 18 : 6
#define ClasBuildInfo_packed_indexType 24 : 4
#define ClasBuildInfo_packed_opacityMicromapIndexType 28 : 4
  uint32_t packed;

  // struct VkClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV
  // {
  //   uint32_t geometryIndex : 24;
  //   uint32_t reserved : 5;
  //   uint32_t geometryFlags : 3;
  // };
  // VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV << 29
#define ClasGeometryFlag_OPAQUE_BIT_NV (4 << 29)
  uint32_t baseGeometryIndexAndFlags;

  uint16_t indexBufferStride;
  uint16_t vertexBufferStride;
  uint16_t geometryIndexAndFlagsBufferStride;
  uint16_t opacityMicromapIndexBufferStride;
  uint64_t indexBuffer;
  uint64_t vertexBuffer;
  uint64_t geometryIndexAndFlagsBuffer;
  uint64_t opacityMicromapArray;
  uint64_t opacityMicromapIndexBuffer;
};
BUFFER_REF_DECLARE_ARRAY(ClasBuildInfos_inout, ClasBuildInfo, , 8);

// Indirect build information to instantiate a template, which ressults in a CLAS, using the new device-side acceleration structure function
struct TemplateInstantiateInfo
{
  uint32_t clusterIdOffset;
  uint32_t geometryIndexOffset;
  uint64_t clusterTemplateAddress;
  uint64_t vertexBufferAddress;
  uint64_t vertexBufferStride;
};
BUFFER_REF_DECLARE_ARRAY(TemplateInstantiateInfos_inout, TemplateInstantiateInfo, , 16);

// Indirect build information to build a BLAS from an array of CLAS references
struct BlasBuildInfo
{
  // the number of CLAS that this BLAS references
  uint32_t clusterReferencesCount;
  // stride of array (typically 8 for 64-bit)
  uint32_t clusterReferencesStride;
  // start address of the array
  uint64_t clusterReferences;
};
BUFFER_REF_DECLARE_ARRAY(BlasBuildInfos_inout, BlasBuildInfo, , 16);

// The central structure that contains relevant information to
// perform the runtime tessellation and building of 
// all relevant clusters to be rendered in the current frame.
// (not optimally packed for cache efficiency but readability)
struct SceneBuilding
{
  vec3 viewPos;
  uint _pad;

  uint numRenderInstances;
  uint visibleClusterCounter;

  uint fullClusterCounter;
  uint partTriangleCounter;

  uint64_t dualPartTriangleCounter;  // dual sided counter version used in transient builds

  int  splitTriangleCounter;
  uint splitReadCounter;
  uint splitWriteCounter;

  uint     genVertexCounter;
  uint     genClusterCounter;
  uint64_t genClusterDataCounter;

  DispatchIndirectCommand dispatchClassify;

  // instance states store culling/visibility related information
  // result of instance classification
  BUFFER_REF(uint32s_inout) instanceStates;  

  // result of traversal / culling
  BUFFER_REF(ClusterInfos_inout) visibleClusters;

  // clusters to be rendered as is without subdivision
  BUFFER_REF(ClusterInfos_inout) fullClusters;
  // triangle to be split recursively (subdivision too big)
  BUFFER_REF(TessTriangleInfos_coh_volatile) splitTriangles;
  // sub-triangle to be rendered (subdivision fits in table)
  // or for transient builds the end of array contains meta information for triangle remapping as well
  BUFFER_REF(TessTriangleInfos_inout) partTriangles;
  
  // rasterization related
  //////////////////////////////////////////////////
  
  DrawMeshTasksIndirectCommandNV drawFullClusters;
  DrawMeshTasksIndirectCommandNV drawPartTriangles;
  
  // ray tracing focused
  //////////////////////////////////////////////////
  
  DispatchIndirectCommand dispatchClusterInstantiate;
  DispatchIndirectCommand dispatchTriangleInstantiate;
  DispatchIndirectCommand dispatchBlasTempInsert;
  DispatchIndirectCommand dispatchBlasTransInsert;
  
  uint positionTruncateBitCount;
  
  uint blasClusterCounter;
  uint tempInstantiateCounter;
  uint transBuildCounter;
  
  // precomputed worst-case CLAS sizes based on number of triangles per cluster
  BUFFER_REF(uint32s_in) basicClusterSizes;
  
  // generated clusters base address
  uint64_t genClusterData;
  // generated vertices for the CLAS build/template instantiation input
  BUFFER_REF(vec3s_inout) genVertices;

  // template instantiations
  // these buffers are effecitvely SoA, sized according to the max
  // number of CLAS to be built
  
  // which instance the instantiation belongs to
  BUFFER_REF(uint32s_inout) tempInstanceIDs;
  // indirect instantiation arguments
  BUFFER_REF(TemplateInstantiateInfos_inout) tempInstantiations;
  // after instantiation, contains CLAS addresses
  BUFFER_REF(uint64s_inout) tempClusterAddresses;
  // 
  BUFFER_REF(uint32s_inout) tempClusterSizes;

  // transient CLAS clusters
  // similar to above
  BUFFER_REF(uint32s_inout) transInstanceIDs;
  BUFFER_REF(ClasBuildInfos_inout) transBuilds;
  BUFFER_REF(uint64s_inout) transClusterAddresses;
  BUFFER_REF(uint32s_inout) transClusterSizes;

  // transient clusters contain a subset of the original cluster triangles,
  // need to map back the triangleID.
  // aliases with partTriangles, needed at render time
  BUFFER_REF(uint8s_inout) transTriMappings;
  // transient clusters also need have triangle index buffers for CLAS building
  // aliases with genVertices
  BUFFER_REF(uint8s_inout) transTriIndices;

  // blas
  // per instance
  BUFFER_REF(BlasBuildInfos_inout) blasBuildInfos;
  BUFFER_REF(uint32s_inout) blasBuildSizes;
  // split into per-instance regions
  BUFFER_REF(uint64s_inout) blasClusterAddresses;
  uint64_t blasBuildData;

  uint numBlasReservedSizes;
};


#ifdef __cplusplus
} // namespace shaderio
#endif

#endif
