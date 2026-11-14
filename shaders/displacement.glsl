/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Utility code to setup the PN-triangle displacement 
  as well as the procedural deformation effect.
  
*/

struct DeformBasePN {
  vec3 vB030;
  vec3 vB003;
  vec3 vB300;
  vec3 vB021;
  vec3 vB012;
  vec3 vB102;
  vec3 vB201;
  vec3 vB210;
  vec3 vB120;
  vec3 vB111;
};

vec3 deform_projectToPlane(vec3 p, vec3 plane, vec3 planeNormal)
{
  vec3 delta = p - plane;
  vec3 deltaProjected = dot(delta, planeNormal) * planeNormal;
  return (p - deltaProjected);
}

void deform_setupPN(inout DeformBasePN basis, vec3 verts[3], vec3 normals[3])
{
  // https://alex.vlachos.com/graphics/CurvedPNTriangles.pdf
  // https://ogldev.org/www/tutorial31/tutorial31.html

  basis.vB030 = verts[0];
  basis.vB003 = verts[1];
  basis.vB300 = verts[2];

  vec3 edgeB300 = basis.vB003 - basis.vB030;
  vec3 edgeB030 = basis.vB300 - basis.vB003;
  vec3 edgeB003 = basis.vB030 - basis.vB300;

  basis.vB021 = basis.vB030 + edgeB300 / 3.0;
  basis.vB012 = basis.vB030 + edgeB300 * 2.0 / 3.0;
  basis.vB102 = basis.vB003 + edgeB030 / 3.0;
  basis.vB201 = basis.vB003 + edgeB030 * 2.0 / 3.0;
  basis.vB210 = basis.vB300 + edgeB003 / 3.0;
  basis.vB120 = basis.vB300 + edgeB003 * 2.0 / 3.0;

  basis.vB021 = deform_projectToPlane(basis.vB021, basis.vB030, normals[0]);
  basis.vB012 = deform_projectToPlane(basis.vB012, basis.vB003, normals[1]);
  basis.vB102 = deform_projectToPlane(basis.vB102, basis.vB003, normals[1]);
  basis.vB201 = deform_projectToPlane(basis.vB201, basis.vB300, normals[2]);
  basis.vB210 = deform_projectToPlane(basis.vB210, basis.vB300, normals[2]);
  basis.vB120 = deform_projectToPlane(basis.vB120, basis.vB030, normals[0]);

  vec3 vCenter = (basis.vB003 + basis.vB030 + basis.vB300) / 3.0;
  basis.vB111  = (basis.vB021 + basis.vB012 + basis.vB102 + basis.vB201 + basis.vB210 + basis.vB120) / 6.0;
  basis.vB111 += (basis.vB111 - vCenter) / 2.0;
}

vec3 deform_getPN(inout DeformBasePN basis, vec3 bary)
{
  float u = bary.x;
  float v = bary.y;
  float w = bary.z;

  float uPow3 = pow(u, 3);
  float vPow3 = pow(v, 3);
  float wPow3 = pow(w, 3);
  float uPow2 = pow(u, 2);
  float vPow2 = pow(v, 2);
  float wPow2 = pow(w, 2);

  vec3 pnPos =  basis.vB300 * wPow3 +
    basis.vB030 * uPow3 +
    basis.vB003 * vPow3 +
    basis.vB210 * 3.0 * wPow2 * u +
    basis.vB120 * 3.0 * w * uPow2 +
    basis.vB201 * 3.0 * wPow2 * v +
    basis.vB021 * 3.0 * uPow2 * v +
    basis.vB102 * 3.0 * w * vPow2 +
    basis.vB012 * 3.0 * u * vPow2 +
    basis.vB111 * 6.0 * w * u * v;

  return pnPos;
}

vec3 rippleDeformAt(vec3 originalVertex, uint seed, float geometrySize, float animationState)
{
  vec3 newVertex = originalVertex;

  float maxCoord = max(abs(originalVertex.x), max(abs(originalVertex.y), abs(originalVertex.z)));

  float frequency = view.animationRippleFrequency / geometrySize;

  vec3 wave = vec3(sin(maxCoord * frequency + seed + animationState * view.animationRippleSpeed),
                   cos(maxCoord * frequency * 3 + seed + animationState * view.animationRippleSpeed),
                   sin(maxCoord * frequency * 1.2f + seed + animationState * view.animationRippleSpeed));
  newVertex += (normalize(originalVertex.zyx)) * (view.animationRippleAmplitude * geometrySize * wave);
  
  return newVertex;
}

vec3 rippleDeform(vec3 originalVertex, uint seed, float geometrySize)
{
  return rippleDeformAt(originalVertex, seed, geometrySize, view.animationState);
}

// Returns where the rippled vertex was in the previous frame. Its original position is no longer known,
// two fixed-point iterations recover it well enough to evaluate the ripple at both points in time.
vec3 rippleDeformPrev(vec3 deformedVertex, uint seed, float geometrySize)
{
  vec3 originalVertex = deformedVertex;
  [[unroll]] for(uint i = 0; i < 2; i++)
  {
    originalVertex = deformedVertex - (rippleDeformAt(originalVertex, seed, geometrySize, view.animationState) - originalVertex);
  }

  return deformedVertex + rippleDeformAt(originalVertex, seed, geometrySize, view.animationStatePrev)
         - rippleDeformAt(originalVertex, seed, geometrySize, view.animationState);
}
