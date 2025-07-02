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

#extension GL_EXT_fragment_shader_barycentric : enable

uint murmurHash(uint idx)
{
  uint m = 0x5bd1e995;
  uint r = 24;

  uint h = 64684;
  uint k = idx;

  k *= m;
  k ^= (k >> r);
  k *= m;
  h *= m;
  h ^= k;

  return h;
}

vec4 clusterShading(uint clusterID)
{
  return vec4(unpackUnorm4x8(murmurHash(clusterID ^ view.colorXor)).xyz * 0.9 + 0.1, 1);
}

vec4 shading(uint instanceID, vec3 wPos, vec3 wNormal, uint clusterID, float overheadLight, float ambientOcclusion)
{
  const vec3 sunColor       = vec3(0.99f, 1.f, 0.71f);
  const vec3 skyColor       = view.skyParams.skyColor;
  const vec3 groundColor    = view.skyParams.groundColor;
  vec3       materialAlbedo = vec3(0.8f);
  vec4       color          = vec4(0.f);

  if((view.visualize == VISUALIZE_CLUSTER || view.visualize == VISUALIZE_TESSELLATED_CLUSTER || view.visualize == VISUALIZE_TRIANGLES)  && clusterID != 0)
  {
    materialAlbedo = clusterShading(clusterID).xyz * 0.5 + 0.3;
  }

  vec3 normal  = normalize(wNormal.xyz);
  vec3 wEyePos = vec3(view.viewMatrixI[3].x, view.viewMatrixI[3].y, view.viewMatrixI[3].z);
  vec3 eyeDir  = normalize(wEyePos.xyz - wPos.xyz);

  // Ambient
  float ambientIntensity = 1.f;
  vec3  ambientLighting  = ambientOcclusion * materialAlbedo* ambientIntensity
                         * mix(groundColor, skyColor, dot(normal, view.wUpDir.xyz) * 0.5 + 0.5) ;

  // Light mixer
  float lightMixer             = view.lightMixer;
  float flashlightIntensity    = 1.0f - lightMixer;
  float overheadLightIntensity = lightMixer;

  // Flashlight
  vec3  flashlightLighting  = vec3(0.f);
  {
    // Use a flashlight intensity similar to the sky color for average luminance consistency
    flashlightIntensity *= max(skyColor.x, max(skyColor.y, skyColor.z));
    vec3  lightDir     = normalize(view.wLightPos.xyz - wPos.xyz);
    vec3  reflDir      = normalize(-reflect(lightDir, normal));
    float bsdf         = abs(dot(normal, lightDir)) + pow(max(0, dot(reflDir, eyeDir)), 16) * 0.3;
    flashlightLighting = flashlightIntensity * materialAlbedo * bsdf;
  }

  // Overhead light
  vec3 overheadLightColor = view.skyParams.lightRadiance;
  vec3  overheadLighting       = vec3(overheadLightIntensity * overheadLight * overheadLightColor);
  {
    vec3 lightDir = normalize(view.skyParams.sunDirection);
    vec3 reflDir  = normalize(-reflect(lightDir, normal));
    float diffuse    = max(0, dot(normal, lightDir));
    float specular   = pow(max(0, dot(reflDir, eyeDir)), 16) * 0.3;
    float bsdf       = diffuse + specular;
    overheadLighting = overheadLighting * materialAlbedo * bsdf;
  }

  color.xyz = overheadLighting + flashlightLighting + ambientLighting;
  color.w   = 1.0;
  return color;
}

uint64_t packPickingValue(uint32_t v, float z)
{
  z         = 1.f - clamp(z, 0.f, 1.f);
  uint bits = floatBitsToUint(z);
  bits ^= (int(bits) >> 31) | 0x80000000u;
  uint64_t value = (uint64_t(bits) << 32) | uint64_t(v);
  return value;
}


// Return the width [0..1] for which the line should be displayed or not
float getLineWidth(in vec3 deltas, in float thickness, in float smoothing, in vec3 barys)
{
  barys         = smoothstep(deltas * (thickness), deltas * (thickness + smoothing), barys);
  float minBary = min(barys.x, min(barys.y, barys.z));
  return 1.0 - minBary;
}

// Position along the edge [0..1]
float edgePosition(vec3 barycentrics)
{
  return max(barycentrics.z, max(barycentrics.y, barycentrics.x));
}

// Return 0 or 1 if edgePos should be diplayed or not
float stipple(in float stippleRepeats, in float stippleLength, in float edgePos)
{
  float offset = 1.0 / stippleRepeats;
  offset *= 0.5 * stippleLength;
  float pattern = fract((edgePos + offset) * stippleRepeats);
  return 1.0 - step(stippleLength, pattern);
}


vec3 addWireframe(vec3 color, vec3 barycentrics, bool frontFacing, vec3 barycentricsDerivatives, vec3 wireColor)
{
  float oThickness    = view.wireThickness * 0.5;
  float thickness     = oThickness * 0.5;  // Thickness for both side of the edge, must be divided by 2
  float smoothing     = oThickness * view.wireSmoothing;  // Could be thickness
  bool  enableStipple = (view.wireStipple == 1);

  // Uniform position on the edge [0, 1]
  float edgePos = edgePosition(barycentrics);

  if(!frontFacing)
  {
    enableStipple = true;  // Forcing backface to always stipple the line
    wireColor     = view.wireBackfaceColor;
  }


  // fwidth ? return the sum of the absolute value of derivatives in x and y
  //          which makes the width in screen space
  vec3 deltas = barycentricsDerivatives;  //fwidthFine(barycentrics);

  // Get the wireframe line width
  float lineWidth = getLineWidth(deltas, thickness, smoothing, barycentrics);

  // [optional]
  if(enableStipple)
  {
    float stippleFact = stipple(view.wireStippleRepeats, view.wireStippleLength, edgePos);
    lineWidth *= stippleFact;  // 0 or 1
  }

  // Final color
  return mix(color, wireColor, lineWidth);
}

#if SUPPORTS_RT == 1

uint wangHash(uint seed)
{
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9;
  seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d;
  seed = seed ^ (seed >> 15);
  return seed;
}

//-----------------------------------------------------------------------
// https://www.pcg-random.org/
//-----------------------------------------------------------------------
uint pcg(inout uint state)
{
  uint prev = state * 747796405u + 2891336453u;
  uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
  state     = prev;
  return (word >> 22u) ^ word;
}

//-----------------------------------------------------------------------
// Generate a random float in [0, 1) given the previous RNG state
//-----------------------------------------------------------------------
float rand(inout uint seed)
{
  uint r = pcg(seed);
  return float(r) * (1.F / float(0xffffffffu));
}
// Generate an arbitrary orthonormal basis from a normal vector
void computeDefaultBasis(const vec3 z, out vec3 x, out vec3 y)
{
  const float yz = -z.y * z.z;
  y = normalize(((abs(z.z) > 0.99999f) ? vec3(-z.x * z.y, 1.0f - z.y * z.y, yz) : vec3(-z.x * z.z, yz, 1.0f - z.z * z.z)));

  x = cross(y, z);
}
#ifndef M_PI
#define M_PI 3.141592653589
#endif
float ambientOcclusion(vec3 wPos, vec3 wNormal, uint32_t sampleCount, float radius)
{
  uint32_t seed = wangHash(gl_LaunchIDEXT.x) ^ wangHash(gl_LaunchIDEXT.y);
  vec3     z    = wNormal;
  vec3     x, y;
  computeDefaultBasis(z, x, y);

  uint32_t occlusion = 0u;

  for(uint32_t i = 0; i < sampleCount; i++)
  {
    float r1 = 2 * M_PI * rand(seed);
    float r2 = rand(seed);
    float sq = sqrt(1.0 - r2);

    vec3 wDirection  = vec3(cos(r1) * sq, sin(r1) * sq, sqrt(r2));
    wDirection       = wDirection.x * x + wDirection.y * y + wDirection.z * z;
    rayHitAO.color.w = 1.f;
    uint mask        = 0xFF;
    traceRayEXT(asScene, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                mask /*0xFF*/, 0, 0, 1, wPos, 1e-4f, wDirection, radius, 1);
    if(rayHitAO.color.w > 0.f)
    {
      occlusion++;
    }
  }
  float linearAo = float(sampleCount - occlusion) / float(sampleCount);
  return max(0.2f, linearAo* linearAo);
}

float overheadLightingContribution(vec3 wPos, vec3 wNormal, vec3 wShadowDir, bool doShadow)
{
  const float minValue = 0.f;
  if(!doShadow)
    return 0.f;

  float nDotDir = clamp(dot(wNormal, -wShadowDir), 0.f, 1.f);
  if(nDotDir <= minValue)
  {
    return minValue;
  }

  vec3 wDirection = -wShadowDir;

  rayHitAO.color.w = 1.f;
  uint mask        = 0xFF;
  traceRayEXT(asScene, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
              mask /*0xFF*/, 0, 0, 1, wPos, 0.001f, wDirection, 10000000, 1);

  return (rayHitAO.color.w > 0.f) ? minValue : 1.f;
}

// Returns 0.0 if there is a hit along the light direction and 1.0, if nothing was hit
float traceShadowRay(vec3 wPos, vec3 wDirection)
{
  rayHitAO.color.w = 1.f;
  uint  mask       = 0xFF;
  uint  flags      = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;
  float minT       = 0.001f;
  float maxT       = 10000000.0f;
  traceRayEXT(asScene, flags, mask, 0, 0, 1, wPos, minT, wDirection, maxT, 1);

  return (rayHitAO.color.w > 0.f) ? 0.0F : 1.0f;
}

float determinant(vec3 a, vec3 b, vec3 c)
{
  return dot(cross(a, b), c);
}

vec3 intersectRayTriangle(vec3 origin, vec3 direction, vec3 v0, vec3 v1, vec3 v2)
{
  // Edge vectors
  vec3 e1 = v1 - v0;
  vec3 e2 = v2 - v0;

  // Plane normal (cross product of edge vectors)
  vec3 planeNormal = cross(e1, e2);

  // Check for parallelism (ray parallel to the plane)
  float nDotDir = dot(planeNormal, direction);
  //if (abs(nDotDir) < 0.001f) {
  //  return vec3(-1.0); // Return negative values to indicate no intersection
  //}

  // Distance from ray origin to the plane
  float t = dot(planeNormal, v0 - origin) / nDotDir;

  // Check if intersection is behind the ray origin (negative t means no intersection)
  //if (t <= 0.0) {
  //  return vec3(-1.0); // Return negative values to indicate no intersection
  //}

  // Intersection point
  vec3 p = origin + t * direction;

  // Compute barycentric coordinates using determinant
  vec3  temp = p - v0;
  float det  = determinant(e1, e2, planeNormal);
  float u    = dot(cross(temp, e2), planeNormal) / det;
  float v    = dot(cross(e1, temp), planeNormal) / det;
  float w    = 1.0 - u - v;

  return vec3(w, u, v);
}

ivec2 objectToPixel(vec3 objectPos)
{
  vec3 wObjectPos = gl_ObjectToWorldEXT * vec4(objectPos, 1.f);

  vec4 pPos = view.viewProjMatrix * vec4(wObjectPos, 1.f);

  pPos /= pPos.w;
  pPos.xy = pPos.xy * vec2(0.5f) + vec2(0.5f);
  pPos.xy *= vec2(gl_LaunchSizeEXT.xy);
  return ivec2(pPos.xy);
}



#endif  // SUPPORTS_RT