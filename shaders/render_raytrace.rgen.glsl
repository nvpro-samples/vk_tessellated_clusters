/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : enable
#include "shaderio.h"
#if USE_DLSS
#include "dlss_util.h"
#endif

//////////////////////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(set = 0, binding = BINDINGS_TLAS) uniform accelerationStructureEXT asScene;
layout(set = 0, binding = BINDINGS_RENDER_TARGET, rgba8) uniform image2D imgColor;

layout(set = 0, binding = BINDINGS_RAYTRACING_DEPTH, r32f) uniform image2D imgRaytracingDepth;

#if USE_DLSS
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssAlbedo, rgba8)            uniform image2D imgDlssAlbedo;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssSpecAlbedo, rgba16f)      uniform image2D imgDlssSpecAlbedo;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssNormalRoughness, rgba16f) uniform image2D imgDlssNormalRoughness;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssMotion, rg16f)            uniform image2D imgDlssMotion;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssSpecHitDist, r16f)        uniform image2D imgDlssSpecHitDist;
#endif

//////////////////////////////////////////////////////////////

layout(location = 0) rayPayloadEXT RayPayload rayHit;

//////////////////////////////////////////////////////////////

void main()
{
  // for writing debugging values to stats.debug etc.
  bool center = gl_LaunchIDEXT.xy == (gl_LaunchSizeEXT.xy / 2);

  vec2 pixelOffset = vec2(0.5);
#if USE_DLSS
  pixelOffset += view.jitter;
#endif

  ivec2 screen = ivec2(gl_LaunchIDEXT.xy);
  vec2  uv     = (vec2(gl_LaunchIDEXT.xy) + pixelOffset) / vec2(gl_LaunchSizeEXT.xy);


  vec2 d = uv * 2.0 - 1.0;


  vec4 origin    = view.viewMatrixI * vec4(0, 0, 0, 1);
  vec4 target    = normalize(view.projMatrixI * vec4(d.x, d.y, 1, 1));
  vec4 direction = normalize(view.viewMatrixI * vec4(target.xyz, 0));

  float tMin = view.nearPlane;
  float tMax = view.farPlane;

#if DEBUG_VISUALIZATION
  vec2 uvOffset            = (vec2(gl_LaunchIDEXT.xy) + pixelOffset + vec2(1.0, 1.0)) / vec2(gl_LaunchSizeEXT.xy);
  vec2 dOffset             = uvOffset * 2.0 - 1.0;
  vec4 targetOffsetX       = normalize(view.projMatrixI * vec4(dOffset.x, d.y, 1, 1));
  vec4 targetOffsetY       = normalize(view.projMatrixI * vec4(d.x, dOffset.y, 1, 1));
  vec4 directionOffsetX    = normalize(view.viewMatrixI * vec4(targetOffsetX.xyz, 0));
  vec4 directionOffsetY    = normalize(view.viewMatrixI * vec4(targetOffsetY.xyz, 0));
  rayHit.color.xyz         = directionOffsetX.xyz;
  rayHit.differentialY.xyz = directionOffsetY.xyz;
#endif

  traceRayEXT(asScene, gl_RayFlagsCullBackFacingTrianglesEXT, 0xff, 0, 0,  // hit offset, hit stride
              0,                                                           // miss offset
              origin.xyz, tMin, direction.xyz, tMax,
              0  // rayPayloadNV location qualifier
  );

  bool  hitValid = rayHit.color.w != 0;
  // reversed-Z, misses are at the far plane
  float hitDepth = hitValid ? rayHit.color.w : 0.0;

  {
    imageStore(imgColor, screen, vec4(rayHit.color.xyz, 1));
    imageStore(imgRaytracingDepth, screen, vec4(hitDepth, 0.f, 0.f, 0.f));
  }

#if USE_DLSS
  vec3 hitPos;
  if(hitValid)
  {
    vec4 wPos = view.viewProjMatrixI * vec4(d.x, d.y, hitDepth, 1);
    hitPos    = wPos.xyz / wPos.w;
  }
  else
  {
    hitPos = origin.xyz + direction.xyz * view.farPlane * 0.99f;
  }

#if DO_ANIMATION
  vec3 hitPosPrev = hitValid ? rayHit.dlssWorldPosPrev : hitPos;
  vec2 motionVec  = calculateMotionVector(hitPos, hitPosPrev, view.viewProjMatrixPrev, view.viewProjMatrix, view.viewportf);
#else
  vec2 motionVec = calculateMotionVector(hitPos, view.viewProjMatrixPrev, view.viewProjMatrix, view.viewportf);
#endif
  imageStore(imgDlssMotion, screen, vec4(motionVec.x, motionVec.y, 0, 0));
  imageStore(imgDlssSpecHitDist, screen, vec4(0));
  imageStore(imgDlssAlbedo, screen, rayHit.dlssAlbedo);
  imageStore(imgDlssSpecAlbedo, screen, vec4(rayHit.dlssSpecular, 1.0f));
  imageStore(imgDlssNormalRoughness, screen, rayHit.dlssNormalRoughness);
#endif
}