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

  Utility functions for the runtime tessellation.

  Rely on the availability of `tessTable`

*/

bool tess_isValid(TessTriangleInfo info)
{
  return info.cluster.clusterID != ~0 &&
         info.cluster.instanceID != ~0 &&
         info.subTriangle.vtxEncoded.x != ~0 &&
         info.subTriangle.vtxEncoded.y != ~0 &&
         info.subTriangle.vtxEncoded.z != ~0 &&
         info.subTriangle.triangleID_config != ~0;
}

void tess_invalidate(out TessTriangleInfo info)
{
  info.cluster.clusterID = ~0;
  info.cluster.instanceID = ~0;
  info.subTriangle.vtxEncoded.x = ~0;
  info.subTriangle.vtxEncoded.y = ~0;
  info.subTriangle.vtxEncoded.z = ~0;
  info.subTriangle.triangleID_config = ~0;
}

uint tess_encodeBarycentrics(vec3 wuv)
{
  uvec3 intWuv = uvec3(wuv * vec3(TESSTABLE_COORD_MAX) + 0.5);
  if (intWuv.x > max(intWuv.y, intWuv.z))
    intWuv.x = TESSTABLE_COORD_MAX - intWuv.y - intWuv.z;
  else if (intWuv.y > intWuv.z)
    intWuv.y = TESSTABLE_COORD_MAX - intWuv.x - intWuv.z;
  else
    intWuv.z = TESSTABLE_COORD_MAX - intWuv.x - intWuv.y;
    
  return (intWuv.y | (intWuv.z << 16));
}

uint tess_encodeBarycentrics(uint u, uint v)
{
  return (u | (v << 16));
}

vec3 tess_decodeBarycentrics(uint vtx)
{
  uvec2 uv;
  uv.x = vtx & 0xFFFF;
  uv.y = vtx >> 16;
  vec3 wuv;
  wuv.y = float(vtx & 0xFFFF) / float(TESSTABLE_COORD_MAX);
  wuv.z = float(vtx >> 16)    / float(TESSTABLE_COORD_MAX);
  wuv.x = 1.0 - wuv.y - wuv.z;
  return wuv;
}

vec3 tess_getTessFactors(vec3 wPosA, vec3 wPosB, vec3 wPosC)
{
  // spherical distance to camera
  // use build.viewPos rather than view.viewPos to allow freezing
  vec3 segmentsPerLength;
  float distA = distance(wPosA,build.viewPos.xyz);
  float distB = distance(wPosB,build.viewPos.xyz);
  float distC = distance(wPosC,build.viewPos.xyz);

  // clamp distance to nearplane
  segmentsPerLength.x = 1.0f / max(view.nearPlane, min(distA,distB));
  segmentsPerLength.y = 1.0f / max(view.nearPlane, min(distB,distC));
  segmentsPerLength.z = 1.0f / max(view.nearPlane, min(distC,distA));

  vec3 edgeLengths;
  edgeLengths.x = distance(wPosA, wPosB);
  edgeLengths.y = distance(wPosB, wPosC);
  edgeLengths.z = distance(wPosC, wPosA);
  
  return clamp(edgeLengths * segmentsPerLength * view.viewportf.y * view.tessRate, vec3(1.0), vec3(1024 * 32));
}

vec3 tess_getSplitFactor(vec3 tessFactor)
{
  return clamp(tessFactor / float(TESSTABLE_SIZE), vec3(1), vec3(8));
}

uint tess_getConfigIndex(uint cfg)
{
  return cfg & (~(1 << 15));
}

uvec3 tess_getConfigFactors(uint cfg)
{
  uint idx = tess_getConfigIndex(cfg);
  idx += (1 + TESSTABLE_LOOKUP_SIZE + TESSTABLE_LOOKUP_SIZE * TESSTABLE_LOOKUP_SIZE);
  uint mask = TESSTABLE_LOOKUP_SIZE - 1;
  return uvec3(idx & mask, (idx / (TESSTABLE_LOOKUP_SIZE)) & mask,  (idx / (TESSTABLE_LOOKUP_SIZE * TESSTABLE_LOOKUP_SIZE)) & mask);
}

uint tess_getConfig(uvec3 intFactors, inout uvec3 triangleVertices)
{
  // sort largest first
  uint maxFactor = max(max(intFactors.x,intFactors.y),intFactors.z);
  
  if (maxFactor == intFactors.y)
  {
    intFactors = intFactors.yzx;
    triangleVertices = triangleVertices.yzx;
  }
  else if (maxFactor == intFactors.z)
  {
    intFactors = intFactors.zxy;
    triangleVertices = triangleVertices.zxy;
  }
  
  uint idx = intFactors.x + intFactors.y * TESSTABLE_LOOKUP_SIZE + intFactors.z * TESSTABLE_LOOKUP_SIZE * TESSTABLE_LOOKUP_SIZE
            - (1 + TESSTABLE_LOOKUP_SIZE + TESSTABLE_LOOKUP_SIZE * TESSTABLE_LOOKUP_SIZE);
  
  if (intFactors.z > intFactors.y)
  {
    idx |= 1 << 15;
  }
  
  return idx;
}

bool tess_isFlipped(uint cfg)
{
  return (cfg & (1 << 15)) != 0;
}

uint tess_getConfigTriangleCount(uint cfg)
{
  TessTableEntry entry = tessTable.entries.d[tess_getConfigIndex(cfg)];
  
  return uint(entry.numTriangles);
}

uint tess_getConfigVertexCount(uint cfg)
{
  TessTableEntry entry = tessTable.entries.d[tess_getConfigIndex(cfg)];
  
  return uint(entry.numVertices);
}

uvec3 tess_getConfigTriangleVertices(uint cfg, uint tri)
{
  TessTableEntry entry = tessTable.entries.d[tess_getConfigIndex(cfg)];
  
  uint triPacked = tessTable.triangles.d[entry.firstTriangle + tri];
  u8vec4 triIndices = unpack8(triPacked);
  uvec3  indices = uvec3(triIndices.xyz);
  if (tess_isFlipped(cfg))
    indices = indices.xzy;
    
  return indices;
}

vec3 tess_getConfigVertexBarycentrics(uint cfg, uint vert)
{
  TessTableEntry entry = tessTable.entries.d[tess_getConfigIndex(cfg)];
  
  uint vertexPacked = tessTable.vertices.d[entry.firstVertex + vert];
  
  vec3 wuv = tess_decodeBarycentrics(vertexPacked);
  if (tess_isFlipped(cfg))
    wuv = wuv.yxz;
  
  return wuv;
}

vec3 tess_interpolate(vec3 base[3], vec3 wuv)
{
  return base[0] * wuv.x + base[1] * wuv.y + base[2] * wuv.z;
}
vec2 tess_interpolate(vec2 base[3], vec3 wuv)
{
  return base[0] * wuv.x + base[1] * wuv.y + base[2] * wuv.z;
}
