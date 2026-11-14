/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#extension GL_EXT_fragment_shader_barycentric : enable
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

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

layout(binding = BINDINGS_DISPLACED_TEXTURES) uniform sampler2D[] texturesMap;


///////////////////////////////////////////////////

#if USE_DLSS
#include "dlss_util.h"
#endif
#if USE_DLSS && DO_ANIMATION
#include "displacement.glsl"
#endif
#include "render_shading.glsl"

///////////////////////////////////////////////////

layout(location = 0) in Interpolants
{
#if ALLOW_SHADING
  vec3 wPos;
#if ALLOW_VERTEX_NORMALS
  vec3 wNormal;
#endif
#endif
  flat uint clusterID;
  flat uint instanceID;
}
IN;


///////////////////////////////////////////////////

layout(location = 0, index = 0) out vec4 out_Color;
#if USE_DLSS
layout(location = 1, index = 0) out vec2 out_DlssMotion;
#endif
layout(early_fragment_tests) in;

///////////////////////////////////////////////////


void main()
{
  vec3 wNormal;

#if ALLOW_SHADING
#if ALLOW_VERTEX_NORMALS
  if(view.facetShading != 0)
#endif
  {
    wNormal = -cross(dFdx(IN.wPos), dFdy(IN.wPos));
  }
#if ALLOW_VERTEX_NORMALS
  else
  {
    wNormal = IN.wNormal;
  }
#endif
#endif

  uint clusterShadeID = IN.clusterID;
  if(view.visualize == VISUALIZE_TRIANGLES || (view.visualize == VISUALIZE_TESSELLATED_CLUSTER && gl_PrimitiveID >= 256))
  {
    clusterShadeID ^= gl_PrimitiveID;
  }

  out_Color.w = 1.f;
#if ALLOW_SHADING && 1
  {
    const float overHeadLight = 1.0f;
    const float ambientLight  = 1.f;

    out_Color = shading(IN.instanceID, IN.wPos, wNormal, clusterShadeID, overHeadLight, ambientLight);
  }
#else
  {
    out_Color = clusterShading(clusterShadeID);
  }
#endif

  if(view.visualize == VISUALIZE_TESSELLATED_TRIANGLES)
  {
    bool isTessTriangle = gl_PrimitiveID >= 256;
    if(isTessTriangle)
    {
      out_Color.xyz *= vec3(0.5, 1, 0.5);
    }
  }

#if DEBUG_VISUALIZATION
  if(view.doWireframe != 0 || (view.visFilterInstanceID == IN.instanceID && view.visFilterClusterID == IN.clusterID))
  {
    out_Color.xyz = addWireframe(out_Color.xyz, gl_BaryCoordEXT, gl_FrontFacing, fwidthFine(gl_BaryCoordEXT), view.wireColor);
  }
#endif

#if USE_DLSS && DO_ANIMATION
  {
    RenderInstance instance = instances[IN.instanceID];
    vec3           oPos     = (instance.worldMatrixI * vec4(IN.wPos, 1)).xyz;
    vec3           wPosPrev = (instance.worldMatrix * vec4(rippleDeformPrev(oPos, IN.instanceID, instance.geoHi.w), 1)).xyz;
    out_DlssMotion = calculateMotionVector(IN.wPos, wPosPrev, view.viewProjMatrixPrev, view.viewProjMatrix, view.viewportf);
  }
#elif USE_DLSS
  out_DlssMotion = calculateMotionVector(IN.wPos, view.viewProjMatrixPrev, view.viewProjMatrix, view.viewportf);
#endif

  uvec2 pixelCoord = uvec2(gl_FragCoord.xy);
  if(pixelCoord == view.mousePosition)
  {
    uint32_t packedClusterTriangleId = (IN.clusterID << 8) | (gl_PrimitiveID & 0xFF);
    atomicMax(readback.clusterTriangleId, packPickingValue(packedClusterTriangleId, gl_FragCoord.z));
    atomicMax(readback.instanceId, packPickingValue(IN.instanceID, gl_FragCoord.z));
  }
}