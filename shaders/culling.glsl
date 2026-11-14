/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Utility code for frustum and occlusion culling of 
  bounding boxes
  
*/

const float c_epsilon    = 1.2e-07f;
const float c_depthNudge = 2.0/float(1<<24);

bool intersectSize(vec4 clipMin, vec4 clipMax)
{
  vec2 rect = clipMax.xy - clipMin.xy;
  vec2 clipThreshold = vec2(2.0) / viewLast.viewportf.xy;
  return any(greaterThan(rect,clipThreshold));
}

vec4 getClip(vec4 hPos, out bool valid) {
  valid = !(-c_epsilon < hPos.w && hPos.w < c_epsilon);
  return vec4(hPos.xyz / abs(hPos.w), hPos.w);
}

uint getCullBits(vec4 hPos)
{
  uint cullBits = 0;
  cullBits |= hPos.x < -hPos.w ?  1 : 0;
  cullBits |= hPos.x >  hPos.w ?  2 : 0;
  cullBits |= hPos.y < -hPos.w ?  4 : 0;
  cullBits |= hPos.y >  hPos.w ?  8 : 0;
  cullBits |= hPos.z <  0      ? 16 : 0;
  cullBits |= hPos.z >  hPos.w ? 32 : 0;
  cullBits |= hPos.w <= 0      ? 64 : 0; 
  return cullBits;
}

vec4 getBoxCorner(vec3 bboxMin, vec3 bboxMax, int n)
{
  bvec3 useMax = bvec3((n & 1) != 0, (n & 2) != 0, (n & 4) != 0);
  return vec4(mix(bboxMin, bboxMax, useMax),1);
}

bool intersectFrustum(vec3 bboxMin, vec3 bboxMax, mat4 worldTM, out vec4 oClipmin, out vec4 oClipmax, out bool oClipvalid)
{
  mat4 worldViewProjTM = viewLast.viewProjMatrix * worldTM;
  bool valid;
  // clipspace bbox
  vec4 hPos     = worldViewProjTM * getBoxCorner(bboxMin, bboxMax, 0);
  vec4 clip     = getClip(hPos, valid);
  uint bits     = getCullBits(hPos);
  vec4 clipMin  = clip;
  vec4 clipMax  = clip;
  bool clipValid = valid;
  
  [[unroll]]
  for (int n = 1; n < 8; n++){
    hPos  = worldViewProjTM * getBoxCorner(bboxMin, bboxMax, n);
    clip  = getClip(hPos, valid);
    bits &= getCullBits(hPos);
    // TODO instead of loop unroll manually to do independent paired min/max to allow
    // instruction parallelism
    clipMin = min(clipMin,clip);
    clipMax = max(clipMax,clip);

    clipValid = clipValid && valid;
  }
  
  oClipvalid = clipValid;
  oClipmin = vec4(clamp(clipMin.xy, vec2(-1), vec2(1)), clipMin.zw);
  oClipmax = vec4(clamp(clipMax.xy, vec2(-1), vec2(1)), clipMax.zw);

  //return true;
  return bits == 0;
}

bool intersectHiz(vec4 clipMin, vec4 clipMax)
{
  clipMin.xy = clipMin.xy * 0.5 + 0.5;
  clipMax.xy = clipMax.xy * 0.5 + 0.5;
  
  clipMin.xy *= viewLast.hizSizeFactors.xy;
  clipMax.xy *= viewLast.hizSizeFactors.xy;
   
  clipMin.xy = min(clipMin.xy, viewLast.hizSizeFactors.zw);
  clipMax.xy = min(clipMax.xy, viewLast.hizSizeFactors.zw);
  
  vec2  size = (clipMax.xy - clipMin.xy);
  float maxsize = max(size.x, size.y) * viewLast.hizSizeMax;
  float miplevel = ceil(log2(maxsize));

  // center at most on the mip's last valid texel, so the 2x2 footprint stays on texels the build wrote
  float mipsize = viewLast.hizSizeMax * exp2(-max(miplevel, 0.0));
  vec2  uv      = min((clipMin.xy + clipMax.xy) * 0.5, (ceil(viewLast.hizSizeFactors.zw * mipsize) - 1.0) / mipsize);

  float depth = textureLod(texHizFar, uv, miplevel).r;
  // reversed-Z, the hiz stores the farthest depth, which is the smallest value
  bool result = clipMax.z >= depth - c_depthNudge;

  return result;
}