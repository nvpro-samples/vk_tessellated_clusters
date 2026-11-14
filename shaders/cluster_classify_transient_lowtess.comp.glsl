/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================
  
  This compute shader computes the tessellation rate for every triangle
  in a cluster and then classifies its further processing.
  
  It is the ray tracing variant that optimizes the lowly tessellated triangles
  through transient CLAS builds. `cluster_classify.comp.glsl` is the same shader
  without these optimizations, and `cluster_classify_combined_lowtess.comp.glsl`
  is the alternative that uses combined templates instead. Diffing the files
  against each other shows what an optimization adds.

  A single workgroup represents one clusters. Threads may operate
  on vertex or triangle level.
  
  At first all base vertices of the cluster need to be transformed.
  We skip doing actual displacement for computing the tessellation
  factors for performance reasons, that means we assume triangles will not be
  displaced too much over all.
  
  There is multiple possible outcomes for the tessellation factors:
  - no tessellation required: we instantiate the cluster's template
    append to `build.tempInstantiations`

  - if TESS_USE_1X_TRANSIENTBUILDS is active, we will build a transient CLAS of all
    all triangles that have no tessellation. We use the original cluster vertex
    indices to benefit from connectivity among those triangles.
    append to `build.transBuilds`

  - if TESS_USE_2X_TRANSIENTBUILDS is active, we will build multiple transient CLAS
    collecting a few low-tesselled triangles (up to 2x2) with their own set of
    vertices and triangles.
    append to `build.transBuilds`

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
  
#if DO_CULLING
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

        // transient build require the triangle vertex indices again
        // let's pack them along with the factors
        // we get 3x ( 24 bit: edge factor, 8 bit: triangle vertex index)
        s_factors[tri] = factors | (indices << 24);
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

  bool forceTransient = false;
  
  // if TESS_USE_1X_TRANSIENTBUILDS is active and there is more than  
  // threshold many of simple triangles in a cluster, 
  // we use a transient build for this cluster.
  const uint transientSimpleThreshold = 1;

  // depending on the compile flags we optimize building
  // clusters with low tessellation rates.

#if TESS_USE_1X_TRANSIENTBUILDS
  if (simpleCount == numTriangles || simpleCount > transientSimpleThreshold)
#else
  if (simpleCount == numTriangles)
#endif
  {
    // we only need a single thread to perform the logistical setup for this cluster

    if (gl_LocalInvocationID.x == 0)
    {
      // we might enter this branch either when we are actually a full non-tessellated cluster
      // or if we have some non-tessellated triangles (but not all).
      
      // full cluster CLAS are instantiated via template instantiations
      // partials are going through transient CLAS builds
    
      bool isFull = simpleCount == numTriangles && !forceTransient;
    
    #if USE_RENDER_STATS
      atomicAdd(readback.numFullClusters, isFull ? 1 : 0);
    #endif
      
      // both template instantiation and transient build count against same upper limit
      uint     genOffset = atomicAdd(buildRW.genClusterCounter, 1);
      uint64_t clasDataSize;
      uint     partOffset = 0;
      uint     partSize = 0;
      // In this code path we always store all the cluster vertices, even if we only
      // end up using a subset of the triangles. This allows us to keep the original
      // triangle vertex indices and means we benefit from connectivity among the triangles.
      uint     vertexSize = numVertices;
      
      // let's get the CLAS size estimate for the cluster
    #if TESS_USE_1X_TRANSIENTBUILDS
      if (isFull)
    #endif
      {
        // When full we can use the worst-case instantiation value that we stored during template generation
        // with the geometry. These will actually be quite close to the real thing given templates
        // were able to use topology information
        clasDataSize = uint64_t(instance.clusterTemplateInstantiatonSizes.d[clusterID]);
      }
    #if TESS_USE_1X_TRANSIENTBUILDS
      else
      {
        // If not full, we use some pre-computed sizes as well, these will be worse given
        // they cannot account for the topology of the triangles in the cluster. 
        // however, overall not so many clusters will be typically built like this.
        clasDataSize  = uint64_t(build.basicClusterSizes.d[simpleCount]);
        // We need to increase the vertexSize as we append the triangle indices for the 
        // transient build after the cluster vertex storage (avoids managing two separate allocations).
        // Convert size from uint8 array to fp32 x 3
        vertexSize    += (simpleCount * 3 + 11) / 12;
        // Transient built CLAS require a mapping table of the CLAS triangle to the original cluster triangle,
        // because they only store a subset of triangles. We also need to store some other information with them.
        // Just like with the vertex & indieces data, to avoid managing separate storage,
        // we just grab more space in the `build.partTriangles` array and therefore need to adjust the number
        // of array elements.
        partSize      = (ClusterInfo_size + simpleCount + TessTriangleInfo_size - 1) / TessTriangleInfo_size;
        // get a storage offset for these slots
        partOffset    = build_atomicAdd_partTriangleCounterLowtess(partSize);
      }
    #endif
    
      // grab storage offsets for the clas data and the vertices    
      uint64_t dataOffset    = atomicAdd(buildRW.genClusterDataCounter, clasDataSize);
      uint     vertexOffset  = atomicAdd(buildRW.genVertexCounter, vertexSize);
      
      // test offsets against our pre-allocated limits
      if (  (vertexOffset + vertexSize > MAX_GENERATED_VERTICES) 
            || (genOffset + 1 > MAX_GENERATED_CLUSTERS)
            || (dataOffset + clasDataSize > (uint64_t(MAX_GENERATED_CLUSTER_MEGS) * 1024 * 1024)) 
#if TESS_USE_1X_TRANSIENTBUILDS
            || (partOffset + partSize > MAX_PART_TRIANGLES)
#endif
        )
      {
        vertexOffset = ~0;
      }
      
      s_exchange[0] = vertexOffset;
      s_exchange[1] = partOffset;
      
      if (vertexOffset != ~0)
      {
        // we have enough space and can perform the template instantiation or the transient cluster build
        
      #if TESS_USE_1X_TRANSIENTBUILDS
        if (isFull)
      #endif
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
      #if TESS_USE_1X_TRANSIENTBUILDS
        else
        {
          // transient builds sets up CLAS build        
          // get storage offset (at this point it's guaranteed to be valid)
          uint transOffset = atomicAdd(buildRW.transBuildCounter, 1);
          
          ClasBuildInfo buildInfo;
          // given we have multiple kinds of clusters (full, partial etc.) we need to tag the
          // clusterID so that the hit shader will later know where to get information from.
          buildInfo.clusterID    = (RT_CLUSTER_MODE_1X_SUBSET_CLUSTER << 30) | partOffset;
          buildInfo.clusterFlags = 0;
          
          buildInfo.packed = 0;
          buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_triangleCount, simpleCount);
          buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_vertexCount, numVertices);
          buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_positionTruncateBitCount, build.positionTruncateBitCount);
          buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_indexType, 1);
          
          buildInfo.baseGeometryIndexAndFlags = ClasGeometryFlag_OPAQUE_BIT_NV;
          
          buildInfo.indexBufferStride                 = uint16_t(1);
          buildInfo.vertexBufferStride                = uint16_t(4 * 3);
          buildInfo.geometryIndexAndFlagsBufferStride = uint16_t(0);
          buildInfo.opacityMicromapIndexBufferStride  = uint16_t(0);
      
          buildInfo.vertexBuffer = uint64_t(build.genVertices) + uint64_t(vertexOffset * 4 * 3);
          // here we put the indices of the triangles after the vertices
          buildInfo.indexBuffer  = buildInfo.vertexBuffer  + uint64_t(numVertices * 4 * 3); // indices after vertices
          
          buildInfo.geometryIndexAndFlagsBuffer = 0;
          buildInfo.opacityMicromapArray        = 0;
          buildInfo.opacityMicromapIndexBuffer  = 0;
          
          build.transBuilds.d[transOffset]          = buildInfo;
          // need to know which instance the instantiated CLAS belongs to, so we can insert it into its BLAS
          build.transInstanceIDs.d[transOffset]     = instanceID;
          // we build the CLAS in explicit mode, so we provide the CLAS destination address
          build.transClusterAddresses.d[transOffset] = uint64_t(build.genClusterData) + dataOffset;
          
          // for now just export the crucial header
          // but we will later append the triangle mapping to this partOffset
          build.partTriangles.d[partOffset].cluster = cinfo;
        #if TESS_GENERATE_HISTOGRAM
          atomicAdd(readback.histogram[simpleCount], 1);
        #endif
        }
      #endif
      
        // increment the instance's BLAS number of CLAS it will reference
        // we need this to later build per-BLAS arrays of references
        atomicAdd(build.blasBuildInfos.d[instanceID].clusterReferencesCount, 1);
      #if USE_RENDER_STATS
        atomicAdd(readback.numTotalTriangles, uint(simpleCount));
      #endif
      }
    }

    memoryBarrierShared();
    barrier();
    
    // So far we configured on a single thread what we want to instantiate or build on a cluster
    // level. We still have to store the actual vertex & index data that those structs referenced.
    
    // export vertices & indces
    
    uint vertexOffset = s_exchange[0];
    uint partOffset   = s_exchange[1];
    
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
    #if TESS_USE_1X_TRANSIENTBUILDS
      
      // When we are doing a transient build, rather than a full template instantion, we 
      // also need to export the triangle data
      //
      // As the packing of triangle data is serial, and we want to avoid cross subgroup synchronization
      // we do this on the first subgroup only.
      
      if ((forceTransient || (simpleCount > transientSimpleThreshold && simpleCount != numTriangles)) && gl_SubgroupID == 0)
      {    
        // convert to byte offsets
        // indices come after vertices
        uint indexOffset      = (vertexOffset + numVertices) * 4 * 3;
        // triangle mappings come after ClusterInfo stored within `build.partTriangles`
        uint triMappingOffset = partOffset * TessTriangleInfo_size + ClusterInfo_size;
        
        // export simple triangles in a single subgroup for easier in-order processing
        uint outOffset = 0;
        for (uint tri = gl_SubgroupInvocationID; tri < numTriangles; tri += SUBGROUP_SIZE)
        {
          uvec3 factors = s_factors[tri];
          uvec3 indices = factors >> 24;
          factors &= 0x00FFFFFF;
          
          uint maxFactor    = max(max(factors.x,factors.y),factors.z);
          bool isSimple     = maxFactor == 1;
          uvec4 voteSimple  = subgroupBallot(isSimple);
          uint offsetSimple = subgroupBallotExclusiveBitCount(voteSimple);
          
          uint triOffset = outOffset + offsetSimple;
          
          if (isSimple)
          {
            // mapping transient cluster triangle to original triangle
            // `build.lowtessTriMappings` aliases with build.genVertices
            build.lowtessTriMappings.d[triMappingOffset + triOffset] = uint8_t(tri);
            // triangle indices
            // `build.transTriIndices` aliases with `build.partTriangles`
            build.transTriIndices.d[indexOffset + triOffset * 3 + 0] = uint8_t(indices.x);
            build.transTriIndices.d[indexOffset + triOffset * 3 + 1] = uint8_t(indices.y);
            build.transTriIndices.d[indexOffset + triOffset * 3 + 2] = uint8_t(indices.z);
          }
          
          outOffset += subgroupBallotBitCount(voteSimple);
        }
      }
    #endif
    }
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
      // for transient builds we also need the original triangle indices
      uvec3 indices = factors >> 24;
      factors &= 0x00FFFFFF;
      
      uint maxFactor = max(max(factors.x,factors.y),factors.z);
      
      // classify the triangle's tessellation state 
      bool requiresNoTess    = maxFactor == 1;
      bool requiresLow2xTess = maxFactor <= 2;
      bool requiresSplitTess = maxFactor >  TESSTABLE_SIZE;
      bool requiresPartTess  = maxFactor <= TESSTABLE_SIZE;
    #if TESS_USE_1X_TRANSIENTBUILDS
      if (simpleCount > transientSimpleThreshold && requiresNoTess)
      {
        // this triangle was already handled in the 1X_TRANSIENTBUILDS
        // hence ensure it isn't processed further
        requiresPartTess = false;
        requiresLow2xTess = false;
      }
    #endif
    #if TESS_USE_2X_TRANSIENTBUILDS
      // only those triangles not fitting in the 2X transient builds need
      // full partial triangle emit
      requiresPartTess = requiresPartTess && !requiresLow2xTess;
    #endif
      if (!valid){
        requiresLow2xTess = false;
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
    #if TESS_USE_2X_TRANSIENTBUILDS
      else if (requiresLow2xTess)
      {
        cfg = tess_getConfig(factors, tessInfo.subTriangle.vtxEncoded);
      }
      
      if(subgroupAll(!requiresLow2xTess)) continue;
      
      // Now comes the fun part.
      // In this optimization we pack all tessellated triangles
      // with max edge factor <= 2 into a transient CLAS build.
    
    // The batch size is clamped against the device's cluster limits on the host, which also
    // raises `VkClusterAccelerationStructureTriangleClusterInputNV::maxCluster*Count` and the
    // `basicClusterSizes` table accordingly. The batching itself happens within a subgroup.
    #if TESS_TRANSIENT_2X_BATCHSIZE > SUBGROUP_SIZE
      #error "TESS_TRANSIENT_2X_BATCHSIZE must be <= SUBGROUP_SIZE"
    #endif
    #if (TESS_TRANSIENT_2X_BATCHSIZE & (TESS_TRANSIENT_2X_BATCHSIZE - 1)) != 0
      #error "TESS_TRANSIENT_2X_BATCHSIZE must be a power of two"
    #endif
      
      const uint batchSize = TESS_TRANSIENT_2X_BATCHSIZE;
      
      // we pack up to batchSize tessellated base triangles within a subgroup into one
      // transient build
      
      uvec4 voteBatch   = subgroupBallot(requiresLow2xTess);
      uint  offsetBatch = subgroupBallotExclusiveBitCount(voteBatch);
      uint  relBatch    = (offsetBatch & (batchSize-1));

      bool  batchStart      = requiresLow2xTess && (relBatch == 0);
      uvec4 voteBatchStart  = subgroupBallot(batchStart);
      
      uint  batchIdx         = requiresLow2xTess ? (offsetBatch / batchSize) : ~0;
      // find threads that belong to the same batch
      uvec4 voteSameBatch    = subgroupPartitionNV(batchIdx);
      // find last thread within a batch
      uint  lastSameBatchIdx = subgroupBallotFindMSB(voteSameBatch);
      // The number of base triangles that the batch we belong to actually contains. Only the last
      // batch of a subgroup is partially filled, but reserving space for a full batch would waste
      // a lot of the generated vertices, hence we size everything by the actual content.
      uint  numPatches       = subgroupBallotBitCount(voteSameBatch);
      // worst case triangles and vertices for those patches
      uint  batchTriangles   = numPatches * TESS_2X_TRIANGLES;
      uint  batchVertices    = numPatches * TESS_2X_VERTICES;
      
      // acquire transient      
      uint     transVertexOffset = ~0;
      uint     transPartOffset   = 0;
      // the first thread of the batch we belong to
      uint     startIdx          = subgroupBallotFindMSB(voteBatchStart & gl_SubgroupLeMask);
      
      // prefix sum over the number of tessellated triangles we will generate within group
      uint numTris = requiresLow2xTess ? tess_getConfigTriangleCount(cfg) : 0;
      uint numTrisInclusive = subgroupInclusiveAdd(numTris);
      // the number of triangles up to the last thread in a batch
      uint lastBatchTris = subgroupShuffle(numTrisInclusive, lastSameBatchIdx);
      
      if (batchStart) {
        // the first thread in a batch drives the logic of the transient CLAS build setup,
        // it makes all the allocation decisions and fills the build info.
        
        // the allocation is based on the worst case size of the patches in this batch
        uint64_t transDataSize   = uint64_t(build.basicClusterSizes.d[batchTriangles]);
        // append indices after vertices
        uint     transVertexSize = batchVertices + (batchTriangles * 3 + 11) / 12;
        // for these batches triangles we need to store 16-bit mapping table hence * 2
        uint     transPartSize   = (ClusterInfo_size + batchTriangles * 2 + TessTriangleInfo_size - 1) / TessTriangleInfo_size;
        
        uint transGenOffset      = atomicAdd(buildRW.genClusterCounter, 1);
        transPartOffset          = build_atomicAdd_partTriangleCounterLowtess(transPartSize);
        uint64_t transDataOffset = atomicAdd(buildRW.genClusterDataCounter, transDataSize);
        transVertexOffset        = atomicAdd(buildRW.genVertexCounter, transVertexSize);
        
        if (  (transVertexOffset + transVertexSize > MAX_GENERATED_VERTICES) 
              || (transGenOffset + 1 > MAX_GENERATED_CLUSTERS)
              || (transDataOffset + transDataSize > (uint64_t(MAX_GENERATED_CLUSTER_MEGS) * 1024 * 1024)) 
              || (transPartOffset + transPartSize > MAX_PART_TRIANGLES)
            )
        {
          transVertexOffset = ~0;
        }
        else {
          uint transOffset = atomicAdd(buildRW.transBuildCounter, 1);
          
          ClasBuildInfo buildInfo;
          // yet another cluster type in our system, hit shader needs to know this
          buildInfo.clusterID    = (RT_CLUSTER_MODE_2X_BATCHED_TESSELLATED << 30) | transPartOffset;
          buildInfo.clusterFlags = 0;
          
          // actual number of triangles this batch will generate is:
          // numTrisInclusive of last thread in batch - numTrisExclusive of first thread in batch
          // given we only store incluse this is the math required
          uint numBatchTris = (lastBatchTris - (numTrisInclusive - numTris));
          
          // for vertices we don't care about details and use worst-case. That gives
          // each base triangle up to 6 vertices for its tessellated triangles.
          
          buildInfo.packed = 0;
          buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_triangleCount, numBatchTris);
          buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_vertexCount, batchVertices);
          buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_positionTruncateBitCount, build.positionTruncateBitCount);
          buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_indexType, 1);
          
          buildInfo.baseGeometryIndexAndFlags = ClasGeometryFlag_OPAQUE_BIT_NV;
          
          buildInfo.indexBufferStride                 = uint16_t(1);
          buildInfo.vertexBufferStride                = uint16_t(4 * 3);
          buildInfo.geometryIndexAndFlagsBufferStride = uint16_t(0);
          buildInfo.opacityMicromapIndexBufferStride  = uint16_t(0);
      
          buildInfo.vertexBuffer = uint64_t(build.genVertices) + uint64_t(transVertexOffset * 4 * 3);
          buildInfo.indexBuffer  = buildInfo.vertexBuffer + uint64_t(batchVertices * 4 * 3); // indices after vertices
          
          buildInfo.geometryIndexAndFlagsBuffer = 0;
          buildInfo.opacityMicromapArray        = 0;
          buildInfo.opacityMicromapIndexBuffer  = 0;
          
          build.transBuilds.d[transOffset]          = buildInfo;
          build.transInstanceIDs.d[transOffset]     = instanceID;
          build.transClusterAddresses.d[transOffset] = uint64_t(build.genClusterData) + transDataOffset;
          
          build.partTriangles.d[transPartOffset].cluster = cinfo;
        #if TESS_GENERATE_HISTOGRAM
          atomicAdd(readback.histogram[numBatchTris], 1);
        #endif
          
          atomicAdd(build.blasBuildInfos.d[instanceID].clusterReferencesCount, 1);
        #if USE_RENDER_STATS
          atomicAdd(readback.numTotalTriangles, uint(numBatchTris));
        #endif
        }
      }
      
      if (requiresLow2xTess)
      {
        // Now we go wide with our threads again.
        // Each thread will generate the tessellated triangles & vertices
        // for the base triangle it represents.
        
        // And get the batch information from the first thread in a batch we belong to
      
        transPartOffset   = subgroupShuffle(transPartOffset,   startIdx);
        transVertexOffset = subgroupShuffle(transVertexOffset, startIdx);
        
        // we rebase our local tessellated triangles against the number of triangles
        // up to the point the batch starts
        uint firstTris    = subgroupShuffle(numTrisInclusive - numTris, startIdx);
        
        if (transVertexOffset != ~0)
        {
          // do the rebase of our local triangles
          uint baseTris      = numTrisInclusive - numTris - firstTris;
          // pack the tessellation factors for the hit-shader
          uint packedFactors = (factors.x-1) | ((factors.y-1) << 1) | ((factors.z-1) << 2);
          
          // let's compute the actual tessellation of this base triangle
          vec3 baseBarycentrics[3];
          [[unroll]] for (uint v = 0; v < 3; v++) {
            uint vtxEncoded     = tessInfo.subTriangle.vtxEncoded[v];
            baseBarycentrics[v] = tess_decodeBarycentrics(vtxEncoded);
          }

          mat3 worldMatrixIT = transpose(inverse(mat3(worldMatrix)));
          
          uvec3 baseIndices = indices + uint(cluster.firstLocalVertex);
          
          vec3 basePositions[3];
          vec3 baseNormals[3];
          vec2 baseTexcoords[3];
          
          {
            [[unroll]] for (uint v = 0; v < 3; v++) {
              basePositions[v]  = oPositions.d[baseIndices[v]];
              baseNormals[v]    = normalize(oNormals.d[baseIndices[v]]);
              baseTexcoords[v]  = oTexcoords.d[baseIndices[v]];
            }
          }
          
        #if TESS_USE_PN
          DeformBasePN basePN;
          deform_setupPN(basePN, basePositions, baseNormals);
        #endif
        
        
          // compute and store 3 to 6 vertices depending on tessellation config
          // into `build.genVertices`
          uint numVertices = tess_getConfigVertexCount(cfg);
          
          for (uint vert = 0; vert < numVertices; vert++)
          {
            vec3 vertexBarycentrics = tess_getConfigVertexBarycentrics(cfg, vert);
            
            vertexBarycentrics = tess_interpolate(baseBarycentrics, vertexBarycentrics);

          #if TESS_USE_PN
            vec3 oPos    = deform_getPN(basePN, vertexBarycentrics);
          #else
            vec3 oPos    = tess_interpolate(basePositions, vertexBarycentrics);
          #endif
          
          #if HAS_DISPLACEMENT_TEXTURES
            if (instance.displacementIndex >= 0)
            {
              vec3 oNormal = tess_interpolate(baseNormals, vertexBarycentrics);
              vec2  uv     = tess_interpolate(baseTexcoords, vertexBarycentrics);
              float height = texture(displacementTextures[nonuniformEXT(instance.displacementIndex)], uv).r;
              height = (height * instance.displacementScale * view.displacementScale) + instance.displacementOffset + view.displacementOffset;
              oPos += normalize(oNormal) * height;
            }
          #endif
          
          #if DO_ANIMATION
            oPos = rippleDeform(oPos, instanceID, instance.geoHi.w);
          #endif
            
            build.genVertices.d[vert + transVertexOffset + relBatch * TESS_2X_VERTICES] = oPos;
          }
          
          // now store the tessellated triangles
          
          // convert to u8 offset, triangle indices start after vertices
          uint indexOffset = (transVertexOffset + batchVertices) * 4 * 3;
          // convert to u16 offset, triangle mapping starts after ClusterInfo_size
          uint triMappingOffset = (transPartOffset * (TessTriangleInfo_size/2) + (ClusterInfo_size/2));
          
          for (uint i = 0; i < numTris; i++)
          {
            uint triOffset = baseTris + i;
            
            // triangle mapping
            // aliases with `build.partTriangles`
            uint16s_inout(build.lowtessTriMappings).d[triMappingOffset + triOffset] = uint16_t(tri | (i << 8) | (packedFactors << 12));
            
            uvec3 cfgIndices = tess_getConfigTriangleVertices(cfg, i);
            cfgIndices += relBatch * TESS_2X_VERTICES;
          
            // triangle indices
            // aliases with `build.genVertices`
            build.transTriIndices.d[indexOffset + triOffset * 3 + 0] = uint8_t(cfgIndices.x);
            build.transTriIndices.d[indexOffset + triOffset * 3 + 1] = uint8_t(cfgIndices.y);
            build.transTriIndices.d[indexOffset + triOffset * 3 + 2] = uint8_t(cfgIndices.z);
          }
        }
      }
    #endif
    }
  }
}