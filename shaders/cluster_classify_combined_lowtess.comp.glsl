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
  through combined templates. `cluster_classify.comp.glsl` is the same shader
  without these optimizations, and `cluster_classify_transient_lowtess.comp.glsl`
  is the alternative that uses transient CLAS builds instead. Diffing the files
  against each other shows what an optimization adds.
  
  The transient builds have to provide vertices *and* triangle indices, and requires a second
  cluster operation next to the template instantiation. Here we instead instantiate templates
  that we pre-generated with the topology of a batch of independent triangles, or a batch of
  independent 2x2x2 patches (see `tessellation_combined.glsl`). That way everything stays a
  template instantiation and we don't have to generate index data. The trade-off is that the
  vertices of a batch cannot be shared among its triangles.

  A single workgroup represents one clusters. Threads may operate
  on vertex or triangle level.
  
  There is multiple possible outcomes for the tessellation factors:
  - no tessellation required for the entire cluster: we instantiate the cluster's own template
    append to `build.tempInstantiations`
  
  - if TESS_USE_1X_COMBINEDTEMPLATES is active, all triangles of the cluster that don't need
    tessellation are batched into instantiations of the "1x" templates, which store up to
    TESS_COMBINED_1X_BATCHSIZE independent triangles.
    append to `build.tempInstantiations`
  
  - if TESS_USE_2X_COMBINEDTEMPLATES is active, all triangles with tessellation factors <= 2
    are batched into instantiations of the "2x" templates, which store up to
    TESS_COMBINED_2X_BATCHSIZE independent 2x2x2 patches.
    append to `build.tempInstantiations`
      
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
#include "tessellation_combined.glsl"
#if TESS_USE_PN || DO_ANIMATION
#include "displacement.glsl"
#endif

////////////////////////////////////////////

const uint CLUSTER_VERTEX_ITERATIONS = ((CLUSTER_VERTEX_COUNT + CLUSTER_CLASSIFY_WORKGROUP - 1) / CLUSTER_CLASSIFY_WORKGROUP);
const uint CLUSTER_TRIANGLE_ITERATIONS = ((CLUSTER_TRIANGLE_COUNT + CLUSTER_CLASSIFY_WORKGROUP - 1) / CLUSTER_CLASSIFY_WORKGROUP);

const uint SUBGROUP_COUNT = CLUSTER_CLASSIFY_WORKGROUP / SUBGROUP_SIZE;

// number of `build.partTriangles` slots a completely filled "1x" batch needs for its mapping table
const uint COMBINED_1X_BATCH_PARTS = (ClusterInfo_size + TESS_COMBINED_1X_BATCHSIZE + TessTriangleInfo_size - 1) / TessTriangleInfo_size;

shared vec3       s_vertices[CLUSTER_VERTEX_COUNT];
shared uvec3      s_factors[CLUSTER_TRIANGLE_COUNT];
shared uint       s_exchange[max(3,SUBGROUP_COUNT)];

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
      
        // we always need the triangle's vertex indices again later,
        // so let's pack them along with the factors
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

  // if TESS_USE_1X_COMBINEDTEMPLATES is active and there is more than
  // threshold many simple triangles in a cluster, we batch them into
  // instantiations of the "1x" combined templates.
  const uint combinedSimpleThreshold = 1;
  
  // both conditions are uniform across the workgroup
  bool doFullCluster = simpleCount == numTriangles;
#if TESS_USE_1X_COMBINEDTEMPLATES
  bool doBatched1X   = !doFullCluster && simpleCount > combinedSimpleThreshold;
#else
  const bool doBatched1X = false;
#endif

  if (doFullCluster || doBatched1X)
  {
    // The displaced vertices of the cluster are needed in both cases, either as input for the
    // instantiation of the cluster's own template, or as source for the batched
    // "1x" template instantiations, which duplicate them per triangle.
  
    if (doFullCluster)
    {
      // we only need a single thread to perform the logistical setup for this cluster
      if (gl_LocalInvocationID.x == 0)
      {
      #if USE_RENDER_STATS
        atomicAdd(readback.numFullClusters, 1);
      #endif
  
        // When full we can use the worst-case instantiation value that we stored during template
        // generation with the geometry. These will actually be quite close to the real thing given
        // templates were able to use topology information
        uint64_t clasDataSize = uint64_t(instance.clusterTemplateInstantiatonSizes.d[clusterID]);
    
        // grab storage offsets for the clas data and the vertices
        uint     genOffset    = atomicAdd(buildRW.genClusterCounter, 1);
        uint64_t dataOffset   = atomicAdd(buildRW.genClusterDataCounter, clasDataSize);
        uint     vertexOffset = atomicAdd(buildRW.genVertexCounter, numVertices);

        // test offsets against our pre-allocated limits
        if (  (vertexOffset + numVertices > MAX_GENERATED_VERTICES)
              || (genOffset + 1 > MAX_GENERATED_CLUSTERS)
              || (dataOffset + clasDataSize > (uint64_t(MAX_GENERATED_CLUSTER_MEGS) * 1024 * 1024))
           )
        {
          vertexOffset = ~0;
        }
        else
        {
          // we have enough space and can perform the template instantiation
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
          
          build.tempInstantiations.d[tempOffset]   = tempInfo;
          // need to know which instance the instantiated CLAS belongs to, so we can insert it into its BLAS
          build.tempInstanceIDs.d[tempOffset]      = instanceID;
          // we instantiate in explicit mode, so we provide the CLAS destination address
          build.tempClusterAddresses.d[tempOffset] = uint64_t(build.genClusterData) + dataOffset;

          // increment the instance's BLAS number of CLAS it will reference
          // we need this to later build per-BLAS arrays of references
          atomicAdd(build.blasBuildInfos.d[instanceID].clusterReferencesCount, 1);
        #if USE_RENDER_STATS
          atomicAdd(readback.numTotalTriangles, numTriangles);
        #endif
        #if TESS_GENERATE_HISTOGRAM
          atomicAdd(readback.histogram[numTriangles], 1);
        #endif
        }

        s_exchange[0] = vertexOffset;
      }
    }
  #if TESS_USE_1X_COMBINEDTEMPLATES
    else if (gl_SubgroupID == 0)
    {
      // We batch the cluster's simple triangles into as many "1x" template instantiations
      // as required, each is set up by its own thread of the first subgroup.
      // All batches but the last are completely filled, therefore we can allocate the storage
      // of all batches in one go and derive each batch's offsets from its index.

      uint numBatches = (simpleCount + TESS_COMBINED_1X_BATCHSIZE - 1) / TESS_COMBINED_1X_BATCHSIZE;
      uint lastCount  = simpleCount - (numBatches - 1) * TESS_COMBINED_1X_BATCHSIZE;

      // we can pick the templates that match the number of triangles exactly
      uint fullTemplateIndex = tessc_get1xTemplateIndex(TESS_COMBINED_1X_BATCHSIZE);
      uint lastTemplateIndex = tessc_get1xTemplateIndex(lastCount);

      uint64_t fullDataSize = uint64_t(tessTable.combinedTemplateInstantiationSizes.d[fullTemplateIndex]);
      uint64_t lastDataSize = uint64_t(tessTable.combinedTemplateInstantiationSizes.d[lastTemplateIndex]);
      // The instantiated CLAS contains a subset of the original cluster's triangles, therefore we
      // require a mapping table of the CLAS triangle to the original cluster triangle.
      // To avoid managing separate storage we grab slots at the back of the `build.partTriangles`
      // array and store `uint instanceID; uint clusterID; uint8_t triMappings[];` there.
      uint     lastPartSize = (ClusterInfo_size + lastCount + TessTriangleInfo_size - 1) / TessTriangleInfo_size;

      // grab storage offsets for the clas data, the vertices and the mapping tables of all batches,
      // the templates store 3 independent vertices per triangle
      uint  baseGenOffset    = 0;
      uvec2 baseDataOffset   = uvec2(0);
      uint  baseVertexOffset = 0;
      uint  basePartOffset   = 0;
      if (subgroupElect())
      {
        baseGenOffset    = atomicAdd(buildRW.genClusterCounter, numBatches);
        baseDataOffset   = unpackUint2x32(atomicAdd(buildRW.genClusterDataCounter, uint64_t(numBatches - 1) * fullDataSize + lastDataSize));
        baseVertexOffset = atomicAdd(buildRW.genVertexCounter, simpleCount * 3);
        basePartOffset   = build_atomicAdd_partTriangleCounterLowtess((numBatches - 1) * COMBINED_1X_BATCH_PARTS + lastPartSize);
      }
      baseGenOffset    = subgroupBroadcastFirst(baseGenOffset);
      baseDataOffset   = subgroupBroadcastFirst(baseDataOffset);
      baseVertexOffset = subgroupBroadcastFirst(baseVertexOffset);
      basePartOffset   = subgroupBroadcastFirst(basePartOffset);

      // The offsets grow with the batch index, so the batches that fail the limits are always the
      // last ones. That lets us communicate the number of valid batches rather than per-batch state.
      uint numValidBatches = 0;

      for (uint batchFirst = 0; batchFirst < numBatches; batchFirst += SUBGROUP_SIZE)
      {
        uint batch  = batchFirst + gl_SubgroupInvocationID;
        bool isLast = batch == numBatches - 1;

        uint     batchCount    = isLast ? lastCount : TESS_COMBINED_1X_BATCHSIZE;
        uint     templateIndex = isLast ? lastTemplateIndex : fullTemplateIndex;
        uint64_t clasDataSize  = isLast ? lastDataSize : fullDataSize;
        uint     vertexSize    = batchCount * 3;
        uint     partSize      = isLast ? lastPartSize : COMBINED_1X_BATCH_PARTS;

        // all preceding batches are completely filled
        uint     genOffset    = baseGenOffset + batch;
        uint64_t dataOffset   = packUint2x32(baseDataOffset) + uint64_t(batch) * fullDataSize;
        uint     vertexOffset = baseVertexOffset + batch * TESS_COMBINED_1X_BATCHSIZE * 3;
        uint     partOffset   = basePartOffset + batch * COMBINED_1X_BATCH_PARTS;

        bool isValid = batch < numBatches;
        if (  (vertexOffset + vertexSize > MAX_GENERATED_VERTICES)
              || (genOffset + 1 > MAX_GENERATED_CLUSTERS)
              || (dataOffset + clasDataSize > (uint64_t(MAX_GENERATED_CLUSTER_MEGS) * 1024 * 1024))
              || (partOffset + partSize > MAX_PART_TRIANGLES)
           )
        {
          isValid = false;
        }

        uvec4 voteValid  = subgroupBallot(isValid);
        uint  tempOffset = 0;
        if (subgroupElect())
        {
          tempOffset = atomicAdd(buildRW.tempInstantiateCounter, subgroupBallotBitCount(voteValid));
        }
        tempOffset = subgroupBroadcastFirst(tempOffset) + subgroupBallotExclusiveBitCount(voteValid);

        numValidBatches += subgroupBallotBitCount(voteValid);

        if (isValid)
        {
          TemplateInstantiateInfo tempInfo;
          // given we have multiple kinds of clusters (full, partial etc.) we need to tag the
          // clusterID so that the hit shader will later know where to get information from.
          // the combined template's own clusterID is zero
          tempInfo.clusterIdOffset        = (RT_CLUSTER_MODE_1X_SUBSET_CLUSTER << 30) | partOffset;
          tempInfo.geometryIndexOffset    = 0;
          tempInfo.clusterTemplateAddress = tessTable.combinedTemplateAddresses.d[templateIndex];
          tempInfo.vertexBufferAddress    = uint64_t(build.genVertices) + uint64_t(vertexOffset * 4 * 3);
          tempInfo.vertexBufferStride     = 4 * 3;
          
          build.tempInstantiations.d[tempOffset]   = tempInfo;
          build.tempInstanceIDs.d[tempOffset]      = instanceID;
          build.tempClusterAddresses.d[tempOffset] = uint64_t(build.genClusterData) + dataOffset;
          
          // for now just export the crucial header
          // but we will later append the triangle mapping to this partOffset
          build.partTriangles.d[partOffset].cluster = cinfo;
        #if TESS_GENERATE_HISTOGRAM
          atomicAdd(readback.histogram[batchCount], 1);
        #endif
        }
      }

      if (subgroupElect())
      {
        atomicAdd(build.blasBuildInfos.d[instanceID].clusterReferencesCount, numValidBatches);
      #if USE_RENDER_STATS
        atomicAdd(readback.numTotalTriangles, min(simpleCount, numValidBatches * TESS_COMBINED_1X_BATCHSIZE));
      #endif

        s_exchange[0] = baseVertexOffset;
        s_exchange[1] = basePartOffset;
        s_exchange[2] = numValidBatches;
      }
    }
  #endif
    
    memoryBarrierShared();
    barrier();
    
    // So far we configured what we want to instantiate on a cluster level.
    // We still have to store the actual vertex data that those structs referenced.
    
    // apply animation, displacement etc. and keep the result in shared memory,
    // the batched instantiations may read a vertex multiple times.
    for(uint vert = gl_LocalInvocationID.x; vert < numVertices; vert += CLUSTER_CLASSIFY_WORKGROUP)
    {
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

      s_vertices[vert] = oPos;
    }

    memoryBarrierShared();
    barrier();

    if (doFullCluster)
    {
      // the cluster's own template uses the original vertices as is
      uint vertexOffset = s_exchange[0];

      // we used this as marker whether we have storage space at all
      if (vertexOffset != ~0)
      {
        for(uint vert = gl_LocalInvocationID.x; vert < numVertices; vert += CLUSTER_CLASSIFY_WORKGROUP)
        {
          build.genVertices.d[vert + vertexOffset] = s_vertices[vert];
        }
      }
    }
  #if TESS_USE_1X_COMBINEDTEMPLATES
    else
    {
      // The batched templates store independent triangles, which means we have to
      // duplicate the vertices per triangle. There is no index data to be provided.
      //
      // As the packing of the triangles is serial, and we want to avoid cross subgroup
      // synchronization, we do this on the first subgroup only.
        
      if (gl_SubgroupID == 0)
      {
        uint baseVertexOffset = s_exchange[0];
        uint basePartOffset   = s_exchange[1];
        uint numValidBatches  = s_exchange[2];

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
          
          if (isSimple)
          {
            // the index of this triangle among all simple triangles of the cluster
            uint triIndex = outOffset + offsetSimple;
            // which batch it belongs to and which slot it uses within the batch
            uint batch    = triIndex / TESS_COMBINED_1X_BATCHSIZE;
            uint slot     = triIndex % TESS_COMBINED_1X_BATCHSIZE;

            if (batch < numValidBatches)
            {
              // the template's triangle `slot` always uses the vertices `slot * 3 + 0/1/2`,
              // and all preceding batches are completely filled
              uint vertexOffset = baseVertexOffset + triIndex * 3;

              build.genVertices.d[vertexOffset + 0] = s_vertices[indices.x];
              build.genVertices.d[vertexOffset + 1] = s_vertices[indices.y];
              build.genVertices.d[vertexOffset + 2] = s_vertices[indices.z];

              // mapping instantiated cluster triangle to original triangle
              // `build.lowtessTriMappings` aliases with `build.partTriangles`
              uint triMappingOffset = (basePartOffset + batch * COMBINED_1X_BATCH_PARTS) * TessTriangleInfo_size + ClusterInfo_size;
              build.lowtessTriMappings.d[triMappingOffset + slot] = uint8_t(tri);
            }
          }
          
          outOffset += subgroupBallotBitCount(voteSimple);
        }
      }
    }
  #endif
  }
  
  // Up until this point we looked at the simple (non-tessellated) triangles of a cluster.
  // Now let's see about those that actually need tessellation (if there is any)
  
  if (simpleCount != numTriangles)
  {
    // let's iterate over all triangles in the cluster
  
    // bug WAR
    uint numTriSubgroups = (numTriangles + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;
    for (uint triSubgroup = gl_SubgroupID; triSubgroup < numTriSubgroups; triSubgroup += SUBGROUP_COUNT)
    {
      uint tri = triSubgroup * SUBGROUP_SIZE + gl_SubgroupInvocationID;
      bool valid = tri < numTriangles;
    
      // get the edge tessellation factors and the original triangle indices
      uvec3 factors = valid ? s_factors[tri] : uvec3(0);
      uvec3 indices = factors >> 24;
      factors &= 0x00FFFFFF;
      
      uint maxFactor = max(max(factors.x,factors.y),factors.z);
      
      // classify the triangle's tessellation state 
      bool requiresNoTess    = maxFactor == 1;
      bool requiresLow2xTess = maxFactor <= 2;
      bool requiresSplitTess = maxFactor >  TESSTABLE_SIZE;
      bool requiresPartTess  = maxFactor <= TESSTABLE_SIZE;
    #if TESS_USE_1X_COMBINEDTEMPLATES
      if (doBatched1X && requiresNoTess)
      {
        // this triangle was already handled by the "1x" combined templates
        // hence ensure it isn't processed further
        requiresPartTess = false;
        requiresLow2xTess = false;
      }
    #endif
    #if TESS_USE_2X_COMBINEDTEMPLATES
      // only those triangles not fitting in the "2x" combined templates need
      // full partial triangle emit
      requiresPartTess = requiresPartTess && !requiresLow2xTess;
    #endif
      if (!valid){
        requiresLow2xTess = false;
        requiresSplitTess = false;
        requiresPartTess  = false;
      }
    
      // let's get storage offsets for the split or partial triangles
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

    #if TESS_USE_2X_COMBINEDTEMPLATES
      
      if(subgroupAll(!requiresLow2xTess)) continue;
      
      // Now comes the fun part.
      // In this optimization we pack all triangles with max edge factor <= 2 into
      // instantiations of the "2x" combined templates. Unlike with the tessellation table
      // we don't need a dedicated config per factor permutation. A lower configuration is
      // achieved by collapsing vertices within the patch, see `tessellation_combined.glsl`.
    
    // The batch sizes are clamped against the hardware's cluster limits on the host,
    // which also raises `VkClusterAccelerationStructureTriangleClusterInputNV::maxCluster*Count`
    // accordingly. The batching itself happens within a single subgroup.
    #if TESS_COMBINED_2X_BATCHSIZE > SUBGROUP_SIZE
      #error "TESS_COMBINED_2X_BATCHSIZE must be <= SUBGROUP_SIZE"
    #endif
    #if (TESS_COMBINED_2X_BATCHSIZE & (TESS_COMBINED_2X_BATCHSIZE - 1)) != 0
      #error "TESS_COMBINED_2X_BATCHSIZE must be a power of two"
    #endif
      
      const uint batchSize = TESS_COMBINED_2X_BATCHSIZE;
      
      // we pack up to batchSize tessellated base triangles within a subgroup into one
      // template instantiation
      
      uvec4 voteBatch   = subgroupBallot(requiresLow2xTess);
      uint  offsetBatch = subgroupBallotExclusiveBitCount(voteBatch);
      uint  relBatch    = (offsetBatch & (batchSize-1));

      bool  batchStart      = requiresLow2xTess && (relBatch == 0);
      uvec4 voteBatchStart  = subgroupBallot(batchStart);
      
      uint  batchIdx      = requiresLow2xTess ? (offsetBatch / batchSize) : ~0;
      // find threads that belong to the same batch
      uvec4 voteSameBatch = subgroupPartitionNV(batchIdx);
      // the number of patches that the batch we belong to will contain
      uint  numPatches    = subgroupBallotBitCount(voteSameBatch);
      
      // acquire storage for the instantiation
      uint batchVertexOffset = ~0;
      uint batchPartOffset   = 0;
      // the first thread of the batch we belong to
      uint startIdx          = subgroupBallotFindMSB(voteBatchStart & gl_SubgroupLeMask);
      
      if (batchStart) {
        // the first thread in a batch drives the logic of the template instantiation setup,
        // it makes all the allocation decisions and fills the instantiation info.
        
        // we can pick the template that matches the number of patches exactly
        uint templateIndex = tessc_get2xTemplateIndex(numPatches);
        
        uint64_t batchDataSize   = uint64_t(tessTable.combinedTemplateInstantiationSizes.d[templateIndex]);
        uint     batchVertexSize = numPatches * TESS_2X_VERTICES;
        // for these batched triangles we need to store a 16-bit mapping per patch
        uint     batchPartSize   = (ClusterInfo_size + numPatches * 2 + TessTriangleInfo_size - 1) / TessTriangleInfo_size;
        
        uint     batchGenOffset  = atomicAdd(buildRW.genClusterCounter, 1);
        batchPartOffset          = build_atomicAdd_partTriangleCounterLowtess(batchPartSize);
        uint64_t batchDataOffset = atomicAdd(buildRW.genClusterDataCounter, batchDataSize);
        batchVertexOffset        = atomicAdd(buildRW.genVertexCounter, batchVertexSize);

        if (  (batchVertexOffset + batchVertexSize > MAX_GENERATED_VERTICES)
              || (batchGenOffset + 1 > MAX_GENERATED_CLUSTERS)
              || (batchDataOffset + batchDataSize > (uint64_t(MAX_GENERATED_CLUSTER_MEGS) * 1024 * 1024))
              || (batchPartOffset + batchPartSize > MAX_PART_TRIANGLES)
            )
        {
          batchVertexOffset = ~0;
        }
        else {
          uint tempOffset = atomicAdd(buildRW.tempInstantiateCounter, 1);
          
          TemplateInstantiateInfo tempInfo;
          // yet another cluster type in our system, hit shader needs to know this
          tempInfo.clusterIdOffset        = (RT_CLUSTER_MODE_2X_BATCHED_TESSELLATED << 30) | batchPartOffset;
          tempInfo.geometryIndexOffset    = 0;
          tempInfo.clusterTemplateAddress = tessTable.combinedTemplateAddresses.d[templateIndex];
          tempInfo.vertexBufferAddress    = uint64_t(build.genVertices) + uint64_t(batchVertexOffset * 4 * 3);
          tempInfo.vertexBufferStride     = 4 * 3;
          
          build.tempInstantiations.d[tempOffset]   = tempInfo;
          build.tempInstanceIDs.d[tempOffset]      = instanceID;
          build.tempClusterAddresses.d[tempOffset] = uint64_t(build.genClusterData) + batchDataOffset;
          
          build.partTriangles.d[batchPartOffset].cluster = cinfo;
          
          // note that this count includes the degenerate triangles of the patches
          uint numBatchTris = numPatches * TESS_2X_TRIANGLES;
        #if TESS_GENERATE_HISTOGRAM
          atomicAdd(readback.histogram[numBatchTris], 1);
        #endif
          
          atomicAdd(build.blasBuildInfos.d[instanceID].clusterReferencesCount, 1);
        #if USE_RENDER_STATS
          atomicAdd(readback.numTotalTriangles, numBatchTris);
        #endif
        }
      }
      
      if (requiresLow2xTess)
      {
        // Now we go wide with our threads again.
        // Each thread will generate the patch vertices for the base triangle it represents.
        
        // And get the batch information from the first thread in a batch we belong to
      
        batchPartOffset   = subgroupShuffle(batchPartOffset,   startIdx);
        batchVertexOffset = subgroupShuffle(batchVertexOffset, startIdx);
        
        if (batchVertexOffset != ~0)
        {
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
        
          // Compute and store the patch's vertices into `build.genVertices`.
          // Note that within the classification a triangle always covers the entire base
          // triangle, hence the patch barycentrics can be used directly.
          // The vertices of collapsed edges are computed from the very same barycentric
          // coordinate as the corner they collapse to, which guarantees the resulting
          // triangles are truly degenerate.
        
          for (uint vert = 0; vert < TESS_2X_VERTICES; vert++)
          {
            vec3 vertexBarycentrics = tessc_getPatchVertexBarycentrics(vert, factors);

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
            
            build.genVertices.d[batchVertexOffset + relBatch * TESS_2X_VERTICES + vert] = oPos;
          }
          
          // Store which original triangle and which tessellation factors this patch used.
          // The hit shader needs this to compute the barycentric coordinates.
          // convert to u16 offset, the patch mappings start after ClusterInfo_size
          uint triMappingOffset = (batchPartOffset * (TessTriangleInfo_size/2) + (ClusterInfo_size/2));
          // aliases with `build.partTriangles`
          uint16s_inout(build.lowtessTriMappings).d[triMappingOffset + relBatch] = uint16_t(tessc_packPatchMeta(tri, factors));
        }
      }
    #endif
    }
  }
}
