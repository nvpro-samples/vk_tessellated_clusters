/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================
  
  This compute shader computes the tessellation rate for every triangle
  in a cluster and then classifies its further processing.
  
  It is the variant without any optimization for the lowly tessellated
  triangles, and the only variant that rasterization uses. For ray tracing
  there is two alternatives that optimize those triangles, and to keep the
  amount of permutations per file manageable they live in their own files:
  `cluster_classify_transient_lowtess.comp.glsl` and
  `cluster_classify_combined_lowtess.comp.glsl`

  A single workgroup represents one clusters. Threads may operate
  on vertex or triangle level.
  
  At first all base vertices of the cluster need to be transformed.
  We skip doing actual displacement for computing the tessellation
  factors for performance reasons, that means we assume triangles will not be
  displaced too much over all.
  
  There is multiple possible outcomes for the tessellation factors:
  - no tessellation required: we emit the cluster as "full cluster"
    append to `build.fullClusters`
  
  - all remaining triangles
    - if tessellation factors fit within the tessellation table
      append to `build.partTriangles`
    - else we need to split the triangle recursively
      append to `build.splitTriangles`

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
#extension GL_EXT_nonuniform_qualifier : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#extension GL_NV_shader_subgroup_partitioned : require

#include "shaderio.h"

layout(push_constant) uniform pushData
{
  uint instanceID;
} push;

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

layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;  
};

layout(scalar, binding = BINDINGS_TESSTABLE_UBO, set = 0) uniform tessTableBuffer
{
  TessellationTable tessTable;  
};

#if HAS_DISPLACEMENT_TEXTURES
layout(binding = BINDINGS_DISPLACED_TEXTURES, set = 0) uniform sampler2D displacementTextures[];
#endif

////////////////////////////////////////////

layout(local_size_x=CLUSTER_CLASSIFY_WORKGROUP) in;

////////////////////////////////////////////

#include "build.glsl"
#include "tessellation.glsl"
#if TESS_USE_PN || DO_ANIMATION
#include "displacement.glsl"
#endif

////////////////////////////////////////////

const uint CLUSTER_VERTEX_ITERATIONS = ((CLUSTER_VERTEX_COUNT + CLUSTER_CLASSIFY_WORKGROUP - 1) / CLUSTER_CLASSIFY_WORKGROUP);
const uint CLUSTER_TRIANGLE_ITERATIONS = ((CLUSTER_TRIANGLE_COUNT + CLUSTER_CLASSIFY_WORKGROUP - 1) / CLUSTER_CLASSIFY_WORKGROUP);

const uint SUBGROUP_COUNT = CLUSTER_CLASSIFY_WORKGROUP / SUBGROUP_SIZE;

shared vec3       s_vertices[CLUSTER_VERTEX_COUNT];
shared uvec3      s_factors[CLUSTER_TRIANGLE_COUNT];
shared uint       s_exchange[max(2,SUBGROUP_COUNT)];

////////////////////////////////////////////

void main()
{
  // retrieve all information about this cluster

  ClusterInfo cinfo = build.visibleClusters.d[gl_WorkGroupID.x];

  uint instanceID = cinfo.instanceID;
  uint clusterID  = cinfo.clusterID;

  RenderInstance instance = instances[instanceID];
  Cluster cluster         = instance.clusters.d[clusterID];
  
  uint instanceState = build.instanceStates.d[instanceID];

  uint vertMax = cluster.numVertices  - 1;
  uint triMax  = cluster.numTriangles - 1;
  uint numVertices  = cluster.numVertices;
  uint numTriangles = cluster.numTriangles;

  vec3s_in  oPositions     = vec3s_in(instance.positions);
  vec3s_in  oNormals       = vec3s_in(instance.normals);
  vec2s_in  oTexcoords     = vec2s_in(instance.texcoords);
  uint8s_in localTriangles = uint8s_in(instance.clusterLocalTriangles);

  mat4 worldMatrix   = instance.worldMatrix;
  
  
  // load all cluster vertices into shared memory
  // note, we are conserving shared memory and are not transforming the vertices here
  // because we later need object space vertices
  [[unroll]] for(uint i = 0; i < uint(CLUSTER_VERTEX_ITERATIONS); i++)
  {
    uint vert        = gl_LocalInvocationID.x + i * CLUSTER_CLASSIFY_WORKGROUP;
    uint vertLoad    = min(vert, vertMax);
    
    uint vertexIndex = cluster.firstLocalVertex + vertLoad;
    
    vec3 oPos = oPositions.d[vertexIndex];
    
    // for performance we base tessellation factors on the undisplaced triangle
    
    if (vert == vertLoad)
    {
      s_vertices[vert] = oPos.xyz;
    }
  }
  
  memoryBarrierShared();
  barrier();
  
  // this is the number of triangles that require no tessellation
  uint simpleCount = 0;
  
#if DO_CULLING && TARGETS_RAY_TRACING
  // This is a simple heuristic to reduce load for ray tracing, this is a crude switch that can cause
  // visible artifacts, alternatives would be fading out tessellation factors over time.
  
  // instance not directly visible, disable tessellation
  if ((instanceState & INSTANCE_VISIBLE_BIT) == 0)
  {
    simpleCount = gl_SubgroupID == 0 ? numTriangles : 0;
  }
  else
#endif
  {
    // compute the tessellation factors for all triangles based
    // on the vertices in shared memory
    [[unroll]] for(uint i = 0; i < uint(CLUSTER_TRIANGLE_ITERATIONS); i++)
    {
      uint tri     = gl_LocalInvocationID.x + i * CLUSTER_CLASSIFY_WORKGROUP;
      uint triLoad = min(tri, triMax);

      uvec3 indices = uvec3(localTriangles.d[cluster.firstLocalTriangle + triLoad * 3 + 0],
                            localTriangles.d[cluster.firstLocalTriangle + triLoad * 3 + 1],
                            localTriangles.d[cluster.firstLocalTriangle + triLoad * 3 + 2]);
      
      uvec3 factors = uvec3(1,1,1);
      if (tri == triLoad)
      {
        factors = uvec3(tess_getTessFactors((worldMatrix * vec4(s_vertices[indices.x],1)).xyz, 
                                            (worldMatrix * vec4(s_vertices[indices.y],1)).xyz,
                                            (worldMatrix * vec4(s_vertices[indices.z],1)).xyz));
      
        s_factors[tri] = factors;
      }
      
      uint maxFactor = max(max(factors.x,factors.y),factors.z);
      simpleCount   += subgroupBallotBitCount(subgroupBallot(maxFactor == 1 && tri == triLoad));
    }
  }
  
  // processing was spread across multiple subgroups, each computed their own indepdent set
  // of triangles, hence we need to communicate this via shared memory
  if (gl_SubgroupInvocationID == 0){
    s_exchange[gl_SubgroupID] = simpleCount;
  }
  
  memoryBarrierShared();
  barrier();
  
  // let's compute the full number of simple triangles
  simpleCount = subgroupAdd(gl_SubgroupInvocationID < SUBGROUP_COUNT ? s_exchange[gl_SubgroupInvocationID] : 0);

  // all subgroups have read `s_exchange` at this point, from here on we re-use it
  // to communicate the storage offsets, hence we must not overwrite it too early.
  barrier();

  // depending on the compile flags we optimize building
  // clusters with low tessellation rates.
  
  if (simpleCount == numTriangles)
  {
    // we only need a single thread to perform the logistical setup for this cluster
  
    if (gl_LocalInvocationID.x == 0)
    {
    #if TARGETS_RASTERIZATION
    
      // Straight forward in rasterization we treat the cluster where
      // all triangles are no-tessellated (simpleCount == numTriangles) as full cluster.
      // There is also no need to store the vertices, given we can compute them on the fly
      // in the mesh shaders.

      uint fullOffset = atomicAdd(buildRW.fullClusterCounter, 1);
      build.fullClusters.d[fullOffset] = cinfo;
      
    #else
    
      // in ray tracing we setup the template instantiation for the entire cluster
      
    #if USE_RENDER_STATS
      atomicAdd(readback.numFullClusters, 1);
    #endif
      
      uint     genOffset = atomicAdd(buildRW.genClusterCounter, 1);
      uint64_t clasDataSize;
      uint     vertexSize = numVertices;
      
      // let's get the CLAS size estimate for the cluster
      {
        // When full we can use the worst-case instantiation value that we stored during template generation
        // with the geometry. These will actually be quite close to the real thing given templates
        // were able to use topology information
        clasDataSize = uint64_t(instance.clusterTemplateInstantiatonSizes.d[clusterID]);
      }
    
      // grab storage offsets for the clas data and the vertices    
      uint64_t dataOffset    = atomicAdd(buildRW.genClusterDataCounter, clasDataSize);
      uint     vertexOffset  = atomicAdd(buildRW.genVertexCounter, vertexSize);
      
      // test offsets against our pre-allocated limits
      if (  (vertexOffset + vertexSize > MAX_GENERATED_VERTICES) 
            || (genOffset + 1 > MAX_GENERATED_CLUSTERS)
            || (dataOffset + clasDataSize > (uint64_t(MAX_GENERATED_CLUSTER_MEGS) * 1024 * 1024)) 
        )
      {
        vertexOffset = ~0;
      }
      
      s_exchange[0] = vertexOffset;
      
      if (vertexOffset != ~0)
      {
        // we have enough space and can perform the template instantiation
        
        {
          // full cluster sets up template instantiation
          // get storage offset (at this point it's guaranteed to be valid)
          uint tempOffset = atomicAdd(buildRW.tempInstantiateCounter, 1);
          
          // set up instantiation parameters
          TemplateInstantiateInfo tempInfo;
          
          // the template's clusterID is already set appropriately
          tempInfo.clusterIdOffset        = 0;
          tempInfo.geometryIndexOffset    = 0;
          tempInfo.clusterTemplateAddress = instance.clusterTemplateAdresses.d[clusterID];
          // vertices are provided through a memory region that we fill every frame
          tempInfo.vertexBufferAddress    = uint64_t(build.genVertices) + uint64_t(vertexOffset * 4 * 3);
          tempInfo.vertexBufferStride     = 4 * 3;
          
          build.tempInstantiations.d[tempOffset]  = tempInfo;
          // need to know which instance the instantiated CLAS belongs to, so we can insert it into its BLAS
          build.tempInstanceIDs.d[tempOffset]     = instanceID;
          // we instantiate in explicit mode, so we provide the CLAS destination address
          build.tempClusterAddresses.d[tempOffset] = uint64_t(build.genClusterData) + dataOffset;
        #if TESS_GENERATE_HISTOGRAM
          atomicAdd(readback.histogram[numTriangles], 1);
        #endif
        }
      
        // increment the instance's BLAS number of CLAS it will reference
        // we need this to later build per-BLAS arrays of references
        atomicAdd(build.blasBuildInfos.d[instanceID].clusterReferencesCount, 1);
      #if USE_RENDER_STATS
        atomicAdd(readback.numTotalTriangles, uint(simpleCount));
      #endif
      }
      
    #endif
    }
  #if TARGETS_RAY_TRACING
    
    memoryBarrierShared();
    barrier();
    
    // So far we configured on a single thread what we want to instantiate on a cluster
    // level. We still have to store the actual vertex data that those structs referenced.
    
    // export vertices
    
    uint vertexOffset = s_exchange[0];
    
    // we used this as marker whether we have storage space at all
    if (vertexOffset != ~0)
    {
      for(uint vert = gl_LocalInvocationID.x; vert < numVertices; vert += CLUSTER_CLASSIFY_WORKGROUP)
      {
        // apply animation, displacement etc.
        uint vertexIndex = cluster.firstLocalVertex + vert;
        // vertex already was loaded in shared memory
        vec3 oPos = s_vertices[vert];
        
      #if HAS_DISPLACEMENT_TEXTURES
        if (instance.displacementIndex >= 0)
        {
          vec3 oNormal = oNormals.d[vertexIndex];
          vec2  uv     = oTexcoords.d[vertexIndex];
          float height = texture(displacementTextures[nonuniformEXT(instance.displacementIndex)], uv).r;
          height = (height * instance.displacementScale * view.displacementScale) + instance.displacementOffset + view.displacementOffset;
          oPos += normalize(oNormal) * height;
        }
      #endif
      
      #if DO_ANIMATION
        oPos = rippleDeform(oPos, instanceID, instance.geoHi.w);
      #endif
      
        build.genVertices.d[vert + vertexOffset] = oPos;
      }
    }
  #endif
  }
  
  // Up until this point we looked at the simple (non-tessellated) triangles of a cluster.
  // Now let's see about those that actually need tessellation (if there is any)
  
  if (simpleCount != numTriangles)
  {
    
    // let's iterate over all triangles in the cluster
  
  #if 1
    // bug WAR
    uint numTriSubgroups = (numTriangles + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;
    for (uint triSubgroup = gl_SubgroupID; triSubgroup < numTriSubgroups; triSubgroup += SUBGROUP_COUNT)
    {
      uint tri = triSubgroup * SUBGROUP_SIZE + gl_SubgroupInvocationID;
      bool valid = tri < numTriangles;
  #else
    // FIXME compiler bug with subgroup intrinsics
    for(uint tri = gl_LocalInvocationID.x; tri < numTriangles; tri += CLUSTER_CLASSIFY_WORKGROUP)
    {
      bool valid = true;
  #endif
    
      // get the edge tessellation factors
      uvec3 factors = valid ? s_factors[tri] : uvec3(0);
      
      uint maxFactor = max(max(factors.x,factors.y),factors.z);
      
      // classify the triangle's tessellation state 
      bool requiresSplitTess = maxFactor >  TESSTABLE_SIZE;
      bool requiresPartTess  = maxFactor <= TESSTABLE_SIZE;
      if (!valid){
        requiresSplitTess = false;
        requiresPartTess  = false;
      }
    
      // let's get storage offsets for the split or partial triangles
    #if 1
      uvec4 voteSplit = subgroupBallot(requiresSplitTess);
      uvec4 votePart  = subgroupBallot(requiresPartTess);
      
      uint offsetSplit = 0;
      uint offsetPart  = 0;
      
      if (subgroupElect())
      {
        offsetSplit = atomicAdd(buildRW.splitTriangleCounter, int(subgroupBallotBitCount(voteSplit)));
        offsetPart  = build_atomicAdd_partTriangleCounter(subgroupBallotBitCount(votePart));
      }
      
      offsetSplit = subgroupBroadcastFirst(offsetSplit);
      offsetPart  = subgroupBroadcastFirst(offsetPart);
      
      offsetSplit += subgroupBallotExclusiveBitCount(voteSplit);
      offsetPart  += subgroupBallotExclusiveBitCount(votePart);
    #else
    
      uint offsetSplit = atomicAdd(buildRW.splitTriangleCounter, int(requiresSplitTess ? 1 : 0));
      uint offsetPart  = build_atomicAdd_partTriangleCounter(requiresPartTess ? 1 : 0);
    
    #endif
    
      // prepare the information for a tessellated triangle
    
      TessTriangleInfo tessInfo;
      
      tessInfo.cluster     = cinfo;
      tessInfo.subTriangle = SubTriangleInfo(uvec3(~0),~0);
      // by default it covers the entire triangle
      tessInfo.subTriangle.vtxEncoded.x = tess_encodeBarycentrics(0,0);
      tessInfo.subTriangle.vtxEncoded.y = tess_encodeBarycentrics(TESSTABLE_COORD_MAX,0);
      tessInfo.subTriangle.vtxEncoded.z = tess_encodeBarycentrics(0,TESSTABLE_COORD_MAX);
      tessInfo.subTriangle.triangleID_config = tri;
      
      uint cfg;
      
      if (requiresSplitTess && offsetSplit < MAX_SPLIT_TRIANGLES) 
      {
        // insert the triangle for recursive splitting if there is space
        // append to `build.splitTriangles`

        // adjust for split factor
        factors.xyz = tess_getSplitFactor(factors.xyz);

        cfg = tess_getConfig(factors, tessInfo.subTriangle.vtxEncoded);
        tessInfo.subTriangle.triangleID_config |= cfg << 16;

        // no need to use coherent writes here, hence we wrap access to build.splitTriangles
        TessTriangleInfos_inout splitTriangles = TessTriangleInfos_inout(build.splitTriangles);
        
        splitTriangles.d[offsetSplit] = tessInfo;
      }
      else if (requiresPartTess && offsetPart < MAX_PART_TRIANGLES) 
      {
        // insert the triangle into list of renderable partial triangles
        // append to `build.partTriangles`
      
        // compute the internal tessellation table config 
        cfg = tess_getConfig(factors, tessInfo.subTriangle.vtxEncoded);
        tessInfo.subTriangle.triangleID_config |= cfg << 16;
        
        build.partTriangles.d[offsetPart]  = tessInfo;
      }
    }
  }
}