/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Utilities for the "combined template" optimization, which is the alternative to
  the transient CLAS builds. See `cluster_classify_combined.comp.glsl`.

  We pre-build two families of cluster templates at init time:

  - "1x" family: a template that stores `n` independent, non-tessellated triangles.
    `3 * n` vertices and `n` triangles, the triangle indices are simply `0,1,2, 3,4,5, ...`
    The triangle at index `i` therefore always uses the vertices `i * 3 + 0/1/2`.

  - "2x" family: a template that stores `n` independent 2x2x2 patches.
    `TESS_2X_VERTICES * n` vertices and `TESS_2X_TRIANGLES * n` triangles.
    The patch at index `i` uses the vertices `i * 6 + 0..5` and the triangles `i * 4 + 0..3`.

  A patch stores the three corners of the base triangle, plus the mid point of each
  of its three edges:

              C (2)
             / \
            /   \
       (5) x-----x (4)          vertex 3 = mid point of edge 0 (A to B)
          / \   / \             vertex 4 = mid point of edge 1 (B to C)
         /   \ /   \            vertex 5 = mid point of edge 2 (C to A)
        A-----x-----B
       (0)   (3)   (1)

  All lower tessellation configurations (1x1x1, 2x1x1, 2x2x1) are expressed by
  *collapsing* the mid point of an edge with factor 1 onto the edge's first corner.
  The triangles that reference the collapsed vertex become degenerate (zero area) and
  will never be hit by a ray, while the remaining triangles still cover the base
  triangle exactly. For example with 1x1x1 all three mid points collapse onto the
  corners, and only the center triangle (3,4,5) -> (A,B,C) remains non-degenerate.

  This way a single template topology serves all configs <= 2x2x2, at the cost of
  building a few degenerate triangles.

  Requires `tessTable` to be available for the template lookups.
*/

//////////////////////////////////////////////////////////////////////////
// template lookups

// templates that store 1 to TESS_COMBINED_1X_BATCHSIZE independent triangles
uint tessc_get1xTemplateIndex(uint numTriangles)
{
  return numTriangles - 1;
}

// templates that store 1 to TESS_COMBINED_2X_BATCHSIZE independent 2x2x2 patches
uint tessc_get2xTemplateIndex(uint numPatches)
{
  return TESS_COMBINED_1X_BATCHSIZE + numPatches - 1;
}

//////////////////////////////////////////////////////////////////////////
// 2x2x2 patch layout
// must stay in sync with `TessellationTable::initCombinedTemplates`

// the four triangles of a patch, all wound like the base triangle
const uvec3 TESSC_PATCH_TRIANGLES[TESS_2X_TRIANGLES] = {
  uvec3(0, 3, 5),
  uvec3(3, 1, 4),
  uvec3(5, 4, 2),
  uvec3(3, 4, 5),
};

uvec3 tessc_getPatchTriangleVertices(uint tri)
{
  return TESSC_PATCH_TRIANGLES[tri];
}

vec3 tessc_getCornerBarycentrics(uint corner)
{
  return vec3(corner == 0 ? 1 : 0, corner == 1 ? 1 : 0, corner == 2 ? 1 : 0);
}

// Returns the barycentric coordinate of a patch vertex within the base triangle.
// `factors` are the per-edge tessellation factors (each must be 1 or 2),
// edge N goes from corner N to corner (N+1) % 3.
vec3 tessc_getPatchVertexBarycentrics(uint vert, uvec3 factors)
{
  if(vert < 3)
  {
    return tessc_getCornerBarycentrics(vert);
  }

  uint edge = vert - 3;

  // collapse the mid point onto the edge's first corner if the edge isn't subdivided
  if(factors[edge] == 1)
  {
    return tessc_getCornerBarycentrics(edge);
  }

  return (tessc_getCornerBarycentrics(edge) + tessc_getCornerBarycentrics((edge + 1) % 3)) * 0.5;
}

//////////////////////////////////////////////////////////////////////////
// per-patch information
// stored as 16-bit at the back of `build.partTriangles`

// 8 bit: triangle index within the original cluster
// 3 bit: the per-edge tessellation factors (0 means 1 segment, 1 means 2 segments)
uint tessc_packPatchMeta(uint triangleID, uvec3 factors)
{
  return triangleID | (((factors.x - 1) | ((factors.y - 1) << 1) | ((factors.z - 1) << 2)) << 8);
}

uint tessc_unpackPatchTriangleID(uint packed)
{
  return packed & 0xFF;
}

uvec3 tessc_unpackPatchFactors(uint packed)
{
  return uvec3(1 + ((packed >> 8) & 1), 1 + ((packed >> 9) & 1), 1 + ((packed >> 10) & 1));
}
