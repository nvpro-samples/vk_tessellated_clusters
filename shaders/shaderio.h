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

#ifndef _SHADERIO_H_
#define _SHADERIO_H_

#include "shaderio_core.h"
#include "shaderio_scene.h"
#include "shaderio_building.h"
#include "nvshaders/sky_io.h.slang"

/////////////////////////////////////////

#define ALLOW_SHADING 1
#define ALLOW_VERTEX_NORMALS 1

/////////////////////////////////////////

#define VISUALIZE_NONE 0
#define VISUALIZE_CLUSTER 1
#define VISUALIZE_TESSELLATED_CLUSTER 2
#define VISUALIZE_TESSELLATED_TRIANGLES 3
#define VISUALIZE_TRIANGLES 4

/////////////////////////////////////////

// Single descriptor set is used and here we define the
// various binding slots.
// Note we may bind the same buffer as SSBO and UBO
// to leverage different caching strategies.

#define BINDINGS_FRAME_UBO 0
#define BINDINGS_TESSTABLE_UBO 1
#define BINDINGS_READBACK_SSBO 2
#define BINDINGS_RENDERINSTANCES_SSBO 3
#define BINDINGS_SCENEBUILDING_SSBO 4
#define BINDINGS_SCENEBUILDING_UBO 5
#define BINDINGS_HIZ_TEX 6
#define BINDINGS_TLAS 7
#define BINDINGS_RENDER_TARGET 8
#define BINDINGS_RAYTRACING_DEPTH 9
#define BINDINGS_DISPLACED_TEXTURES 10

/////////////////////////////////////////

// The `indirect_setup.comp.glsl` kernel can be run in different
// modes. The mode is passed as push_constant. The overhead
// of calling this single invocation shader is high enough that 
// optimizing it further isn't worth it and a dynamic branch is fine.

#define BUILD_SETUP_CLASSIFY 0
#define BUILD_SETUP_INSTANTIATE_TESS 1
#define BUILD_SETUP_DRAW_TESS 2
#define BUILD_SETUP_SPLIT 3
#define BUILD_SETUP_BUILD_BLAS 4

/////////////////////////////////////////

// dimensions of the various compute shader workgroups

#define INSTANCES_CLASSIFY_WORKGROUP            64
#define CLUSTERS_CULL_WORKGROUP                 64
#define CLUSTER_CLASSIFY_WORKGROUP              64
#define CLUSTER_BLAS_INSERT_WORKGROUP           64
#define CLUSTER_TEMPLATE_INSTANTIATE_WORKGROUP  64
#define TRIANGLE_SPLIT_WORKGROUP                64
#define BLAS_BUILD_SETUP_WORKGROUP              64

/////////////////////////////////////////

// In ray tracing when we hit a cluster, it might have different origin
// depending on the mode we need to fetch the cluster's data differently.
// We encode the mode into the top bits of gl_ClusterIDNV

// A full cluster of the original model.
#define RT_CLUSTER_MODE_FULL_CLUSTER 0
// A cluster that represents a single tessellated region with a triangle (part triangle)
#define RT_CLUSTER_MODE_SINGLE_TESSELLATED 1
// A cluster that is a subset of a non-tessellated cluster (result of TESS_USE_1X_TRANSIENTBUILDS) 
#define RT_CLUSTER_MODE_1X_SUBSET_CLUSTER 2
// A cluster that contains a batch of low-tessellated triangles (result of TESS_USE_2X_TRANSIENTBUILDS) 
#define RT_CLUSTER_MODE_2X_BATCHED_TESSELLATED 3

/////////////////////////////////////////

// various switches we can change in the UI
// or are part of the shader's individual configuration.

#ifndef DO_ANIMATION
#define DO_ANIMATION 1
#endif

#ifndef DO_CULLING
#define DO_CULLING 1
#endif

#ifndef HAS_DISPLACEMENT_TEXTURES
#define HAS_DISPLACEMENT_TEXTURES 1
#endif

#ifndef TARGETS_RASTERIZATION
#define TARGETS_RASTERIZATION 0
#endif

#define TARGETS_RAY_TRACING (!TARGETS_RASTERIZATION)

#ifndef TESS_ACTIVE
#define TESS_ACTIVE 1
#endif

#ifndef TESS_USE_INSTANCEBITS
#define TESS_USE_INSTANCEBITS 1
#endif

#ifndef TESS_USE_1X_TRANSIENTBUILDS
#define TESS_USE_1X_TRANSIENTBUILDS 0
#endif

#ifndef TESS_USE_2X_TRANSIENTBUILDS
#define TESS_USE_2X_TRANSIENTBUILDS 0
#endif

#define TESS_USE_TRANSIENTBUILDS                                                                                       \
  ((TESS_USE_1X_TRANSIENTBUILDS || TESS_USE_2X_TRANSIENTBUILDS) && !TARGETS_RASTERIZATION)

// must be <= SUBGROUP_SIZE

#ifndef TESS_INSTANTIATE_BATCHSIZE
#define TESS_INSTANTIATE_BATCHSIZE 32
#endif

#ifndef TESS_2X_MINI_BATCHSIZE
#define TESS_2X_MINI_BATCHSIZE 8
#endif

#define TESS_2X_MINI_TRIANGLES 4
#define TESS_2X_MINI_VERTICES  6

#if TESS_INSTANTIATE_BATCHSIZE > SUBGROUP_SIZE
#error "invalid TESS_INSTANTIATE_BATCHSIZE"
#endif

#ifndef TESS_USE_PN
#define TESS_USE_PN 0
#endif

/////////////////////////////////////////

#ifdef __cplusplus
namespace shaderio {
using namespace glm;

#endif

struct FrameConstants
{
  mat4 projMatrix;
  mat4 projMatrixI;

  mat4 viewProjMatrix;
  mat4 viewProjMatrixI;
  mat4 viewMatrix;
  mat4 viewMatrixI;
  vec4 viewPos;
  vec4 viewDir;
  vec4 viewPlane;

  mat4 skyProjMatrixI;

  ivec2 viewport;
  vec2  viewportf;

  vec2 viewPixelSize;
  vec2 viewClipSize;

  vec3  wLightPos;
  float tessRate;

  float displacementScale;
  float displacementOffset;
  float lightMixer;
  uint doShadow;

  vec3  wUpDir;
  float sceneSize;

  vec4 bgColor;

  float   lodScale;
  float   animationState;
  float   ambientOcclusionRadius;
  int32_t ambientOcclusionSamples;

  int32_t animationRippleEnabled;
  float   animationRippleFrequency;
  float   animationRippleAmplitude;
  float   animationRippleSpeed;

  uvec3 _pad;
  uint visualize;

  uint  doAnimation;
  uint  flipWinding;
  float nearPlane;
  float farPlane;

  vec4 hizSizeFactors;
  vec4 nearSizeFactors;

  float hizSizeMax;
  int   facetShading;
  int   supersample;
  uint  colorXor;

  uint  dbgUint;
  float dbgFloat;
  float time;
  uint  frame;

  uvec2 mousePosition;
  float wireThickness;
  float wireSmoothing;

  vec3 wireColor;
  uint wireStipple;

  vec3  wireBackfaceColor;
  float wireStippleRepeats;

  float wireStippleLength;
  uint  doWireframe;
  uint  visFilterInstanceID;
  uint  visFilterClusterID;

  SkySimpleParameters skyParams;
};

struct Readback
{
  uint numVisibleClusters;
  uint numFullClusters;

  uint numSplitTriangles;
  uint numPartTriangles;

  uint numTotalTriangles;
  uint numTempInstantiations;

  uint numGenVertices;
  uint numBlasClusters;

  uint numTransBuilds;
  uint numTransPartTriangles;

  uint numActualTransBuilds;
  uint numActualTempInstantiations;

  uint64_t numGenDatas;
  uint64_t numGenActualDatas;

  uint numBlasReservedSizes;
  uint numBlasActualSizes;

  uint64_t debugU64;

#ifndef __cplusplus
  uint64_t clusterTriangleId;
  uint64_t instanceId;
#else
  uint32_t clusterTriangleId;
  uint32_t _packedDepth0;

  uint32_t instanceId;
  uint32_t _packedDepth1;
#endif

  int  debugI;
  uint debugUI;
  uint debugF;

  uint debugA[64];
  uint debugB[64];
  uint debugC[64];
};


struct RayPayload
{
  // Ray gen writes the direction through the pixel at x+1 for ray differentials.
  // Closest hit returns the shaded color there.
  vec4 color;
#if DEBUG_VISUALIZATION
  // Ray direction through the pixel at y+1 for ray differentials
  vec4 differentialY;
#endif
};

#ifdef __cplusplus
}
#endif

#endif
