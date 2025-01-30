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

#ifndef _SHADERIO_SCENE_H_
#define _SHADERIO_SCENE_H_

#include "shaderio_core.h"

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
#else

#ifndef CLUSTER_VERTEX_COUNT
#define CLUSTER_VERTEX_COUNT 32
#endif

#ifndef CLUSTER_TRIANGLE_COUNT
#define CLUSTER_TRIANGLE_COUNT 32
#endif

#ifndef TESSTABLE_LOOKUP_SIZE
#define TESSTABLE_LOOKUP_SIZE 16
#endif

#ifndef TESSTABLE_SIZE
#define TESSTABLE_SIZE 11
#endif

#ifndef TESSTABLE_MAX_TRIANGLES
#define TESSTABLE_MAX_TRIANGLES (TESSTABLE_SIZE * TESSTABLE_SIZE)
#endif

#ifndef TESSTABLE_MAX_VERTICES
#define TESSTABLE_MAX_VERTICES (((TESSTABLE_SIZE + 1) * (TESSTABLE_SIZE + 1 + 1)) / 2)
#endif

#ifndef TESS_RASTER_USE_BATCH
#define TESS_RASTER_USE_BATCH 1
#endif

#define TESS_RASTER_BATCH_VERTICES 96
#define TESS_RASTER_BATCH_TRIANGLES (TESSTABLE_MAX_TRIANGLES)

#ifndef TESSTABLE_COORD_MAX
#define TESSTABLE_COORD_MAX (1 << 15)
#endif

#endif

// A subdivision configuration (for example edge subdivisions: 1x2x3)
// refers to a range of triangles and vertices within
// the TessellationTable
struct TessTableEntry
{
  uint16_t firstTriangle;
  uint16_t firstVertex;
  uint16_t numTriangles;
  uint16_t numVertices;
};
BUFFER_REF_DECLARE_ARRAY(TessTableEntrys_in, TessTableEntry, readonly, 8);

// The tessellation table contains pre-computed subdivisions of triangles
struct TessellationTable
{
  // vertices are stored as uv coordinates in 16-bit
  BUFFER_REF(uint32s_in) vertices;
  // triangles as 3 x 8-bit indices with padding
  BUFFER_REF(uint32s_in) triangles;
  // for each tessellation configuration store the pre-computed subdivision information
  BUFFER_REF(TessTableEntrys_in) entries;
  // for each tessellation configuration provide pre-build templates to build CLAS from
  BUFFER_REF(uint64s_in) templateAddresses;
  // as well as their worst-case instantiation size, so we can estimate the size of the resulting CLAS
  BUFFER_REF(uint32s_in) templateInstantiationSizes;
};

struct BBox
{
  vec3  lo;
  vec3  hi;
  float shortestEdge;
  float longestEdge;
};
BUFFER_REF_DECLARE_ARRAY(BBoxes_inout, BBox, , 16);

// A cluster is made out of a set of triangles and vertices
// the offsets point into respective array of the geometry information
// within a RenderInstance
struct Cluster
{
  uint16_t numVertices;
  uint16_t numTriangles;
  // offset for first global triangle ID
  uint32_t firstTriangle;
  // offset into `RenderInstance::positions, normals, texcoords`
  uint32_t firstLocalVertex;
  // offset into `RenderInstance::clusterLocalTriangles`
  uint32_t firstLocalTriangle;
};
BUFFER_REF_DECLARE_ARRAY(Clusters_in, Cluster, readonly, 16);


// A renderable instance contains matrix, material and geometry information
struct RenderInstance
{
  // spatial information for the object
  mat4 worldMatrix;

  // The geometry information 
  // We avoided another indirection into a unique geometry array
  // and just inlined the data. In a scenario of millions of instances,
  // we would keep the instance more light-weight and use indirection.
  uint32_t geometryID;
  uint32_t numTriangles;
  uint32_t numVertices;
  uint32_t numClusters;

  // for dispacement texture usage
  int32_t displacementIndex;
  float   displacementScale;
  float   displacementOffset;
  float   _pad;

  // object space geometry bbox
  vec4 geoLo;
  vec4 geoHi;

  // vertices over all clusters
  BUFFER_REF(vec3s_in) positions;
  BUFFER_REF(vec3s_in) normals;
  BUFFER_REF(vec2s_in) texcoords;

  // the cluster headers
  BUFFER_REF(Clusters_in) clusters;
  // densely packed 8-bit triangle indices for the vertices within each cluster
  BUFFER_REF(uint8s_in) clusterLocalTriangles;
  // cluster bounding boxes for culling
  BUFFER_REF(BBoxes_inout) clusterBboxes;
  // pre-build cluster templates to speed up CLAS builds by instantiating the templates
  BUFFER_REF(uint64s_in) clusterTemplateAdresses;
  // as well as the instantiation sizes, so we can estimate the worst-case size of the CLAS
  // prior it being built.
  BUFFER_REF(uint32s_in) clusterTemplateInstantiatonSizes;
};

#ifdef __cplusplus
} // namespace shaderio
#endif

#endif
