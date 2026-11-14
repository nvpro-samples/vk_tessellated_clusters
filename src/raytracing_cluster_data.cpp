/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <volk.h>

#include <nvutils/parallel_work.hpp>

#include "raytracing_cluster_data.hpp"

namespace tessellatedclusters {


void RayTracingClusterData::initRayTracingTemplates(Resources& res, Scene& scene, const RendererConfig& config)
{
  m_geometryTemplates.resize(scene.m_geometries.size());

  // generate templates for every geometry
  // and figure out the instantiation size for every cluster

  // slightly lower totals because we do one geometry at a time for template builds.
  VkClusterAccelerationStructureTriangleClusterInputNV templateTriangleInput = {
      VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
  templateTriangleInput.vertexFormat                = VK_FORMAT_R32G32B32_SFLOAT;
  templateTriangleInput.maxClusterTriangleCount     = scene.m_maxClusterTriangles;
  templateTriangleInput.maxClusterVertexCount       = scene.m_maxClusterVertices;
  templateTriangleInput.maxTotalTriangleCount       = scene.m_maxPerGeometryTriangles;
  templateTriangleInput.maxTotalVertexCount         = scene.m_maxPerGeometryClusterVertices;
  templateTriangleInput.minPositionTruncateBitCount = config.positionTruncateBits;

  // following operations are done per cluster in advance
  // we use implicit queries to query scratch size of all
  VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
  inputs.maxAccelerationStructureCount             = scene.m_maxPerGeometryClusters;
  inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
  inputs.opMode                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
  inputs.opInput.pTriangleClusters = &templateTriangleInput;
  inputs.flags                     = config.templateBuildFlags;
  VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);

  VkDeviceSize tempScratchSize = sizesInfo.buildScratchSize;
  inputs.opType                = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
  inputs.opMode                = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
  inputs.flags                 = config.templateInstantiateFlags;
  vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
  tempScratchSize = std::max(tempScratchSize, sizesInfo.buildScratchSize);

  nvvk::Buffer scratchBuffer;
  res.m_allocator.createBuffer(scratchBuffer, tempScratchSize,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                               VMA_MEMORY_USAGE_AUTO, {}, m_scratchAlignment);

  std::vector<VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV> templateInfos(scene.m_maxPerGeometryClusters);
  std::vector<VkClusterAccelerationStructureInstantiateClusterInfoNV> instantiateInfos(scene.m_maxPerGeometryClusters);

  nvvk::Buffer templateInfosBuffer;
  res.m_allocator.createBuffer(templateInfosBuffer,
                               sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV) * templateInfos.size(),
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  nvvk::Buffer instantiateInfosBuffer;
  res.m_allocator.createBuffer(instantiateInfosBuffer,
                               sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV) * instantiateInfos.size(),
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  nvvk::Buffer sizesBuffer;
  res.m_allocator.createBuffer(sizesBuffer, sizeof(uint32_t) * instantiateInfos.size(),
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                   | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  nvvk::Buffer dstAddressesBuffer;
  res.m_allocator.createBuffer(dstAddressesBuffer, sizeof(uint64_t) * instantiateInfos.size(),
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                   | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);


  // 32 byte alignment requirement for bbox
  struct TemplateBbox
  {
    shaderio::BBox bbox;
  };

  nvvk::Buffer bboxesBuffer;

  res.m_allocator.createBuffer(bboxesBuffer, sizeof(TemplateBbox) * instantiateInfos.size(),
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  for(size_t g = 0; g < scene.m_geometries.size(); g++)
  {
    GeometryTemplate&      geometryTemplate = m_geometryTemplates[g];
    const Scene::Geometry& geometry         = scene.m_geometries[g];

    uint32_t numClusters = uint32_t(geometry.clusters.size());

    float bloatSize = glm::length(geometry.bbox.hi - geometry.bbox.lo) * config.templateBBoxBloat;

    for(uint32_t c = 0; c < numClusters; c++)
    {
      const shaderio::Cluster&                                          cluster = geometry.clusters[c];
      VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV& templateInfo =
          ((VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV*)templateInfosBuffer.mapping)[c];

      // add bloat to original bbox

      TemplateBbox& tempBbox = ((TemplateBbox*)bboxesBuffer.mapping)[c];

      shaderio::BBox clusterBbox = geometry.clusterBboxes[c];
      clusterBbox.lo -= bloatSize;
      clusterBbox.hi += bloatSize;

      tempBbox.bbox = clusterBbox;

      templateInfo = {0};

      templateInfo.clusterID     = c;
      templateInfo.vertexCount   = cluster.numVertices;
      templateInfo.triangleCount = cluster.numTriangles;

      templateInfo.baseGeometryIndexAndGeometryFlags.geometryFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV;

      templateInfo.indexBuffer = geometry.clusterLocalTrianglesBuffer.address + (sizeof(uint8_t) * cluster.firstLocalTriangle);
      templateInfo.indexBufferStride = sizeof(uint8_t);
      templateInfo.indexType         = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;

      templateInfo.vertexBuffer       = geometry.positionsBuffer.address + sizeof(glm::vec3) * cluster.firstLocalVertex;
      templateInfo.vertexBufferStride = sizeof(glm::vec3);
      templateInfo.positionTruncateBitCount = config.positionTruncateBits;

      templateInfo.instantiationBoundingBoxLimit =
          config.templateBBoxBloat < 0 ? 0 : bboxesBuffer.address + sizeof(TemplateBbox) * c;
    }

    // actual count of current geometry
    inputs.maxAccelerationStructureCount = numClusters;

    VkCommandBuffer cmd;
    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    cmdInfo.srcInfosArray.deviceAddress     = templateInfosBuffer.address;
    cmdInfo.srcInfosArray.size              = templateInfosBuffer.bufferSize;
    cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV);
    cmdInfo.dstSizesArray.deviceAddress     = sizesBuffer.address;
    cmdInfo.dstSizesArray.size              = sizesBuffer.bufferSize;
    cmdInfo.dstSizesArray.stride            = sizeof(uint32_t);
    cmdInfo.dstAddressesArray.deviceAddress = dstAddressesBuffer.address;
    cmdInfo.dstAddressesArray.size          = dstAddressesBuffer.bufferSize;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);
    cmdInfo.scratchData                     = scratchBuffer.address;

    // query size of templates
    inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
    inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
    inputs.flags  = config.templateBuildFlags;

    cmd = res.createTempCmdBuffer();

    cmdInfo.input = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    res.tempSyncSubmit(cmd);

    // compute template buffer sizes

    uint32_t buildSum = 0;
    for(uint32_t c = 0; c < numClusters; c++)
    {
      buildSum += ((const uint32_t*)sizesBuffer.mapping)[c];
    }
    // allocate outputs and setup dst addresses
    res.m_allocator.createBuffer(geometryTemplate.templateData, buildSum, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
    m_resourceUsageInfo.rtTemplateMemBytes += geometryTemplate.templateData.bufferSize;

    res.m_allocator.createBuffer(geometryTemplate.templateInstantiationSizes, sizeof(uint32_t) * numClusters,
                                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceUsageInfo.operationsMemBytes += geometryTemplate.templateInstantiationSizes.bufferSize;

    res.m_allocator.createBuffer(geometryTemplate.templateAddresses, sizeof(uint64_t) * numClusters,
                                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceUsageInfo.operationsMemBytes += geometryTemplate.templateAddresses.bufferSize;

    {
      uint64_t* dstAddresses = ((uint64_t*)dstAddressesBuffer.mapping);
      buildSum               = 0;
      for(uint32_t c = 0; c < numClusters; c++)
      {
        dstAddresses[c] = geometryTemplate.templateData.address + buildSum;
        buildSum += ((const uint32_t*)sizesBuffer.mapping)[c];
      }

      // build explicit
      inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;

      cmd = res.createTempCmdBuffer();

      cmdInfo.input = inputs;
      vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

      res.tempSyncSubmit(cmd);
    }

    // now compute instantiation sizes
    for(uint32_t c = 0; c < numClusters; c++)
    {
      uint64_t* dstAddresses = ((uint64_t*)dstAddressesBuffer.mapping);

      const shaderio::Cluster&                                cluster = geometry.clusters[c];
      VkClusterAccelerationStructureInstantiateClusterInfoNV& instantiationInfo =
          ((VkClusterAccelerationStructureInstantiateClusterInfoNV*)instantiateInfosBuffer.mapping)[c];

      instantiationInfo.clusterIdOffset        = 0;
      instantiationInfo.clusterTemplateAddress = dstAddresses[c];
      instantiationInfo.geometryIndexOffset    = 0;
      // leave vertices off given we are looking for worst case instantiation size, not actual
      instantiationInfo.vertexBuffer.startAddress  = 0;
      instantiationInfo.vertexBuffer.strideInBytes = 0;
    }

    // query size of instantiations
    inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
    inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
    inputs.flags  = 0;

    cmdInfo.srcInfosArray.deviceAddress     = instantiateInfosBuffer.address;
    cmdInfo.srcInfosArray.size              = instantiateInfosBuffer.bufferSize;
    cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV);
    cmdInfo.dstAddressesArray.deviceAddress = 0;
    cmdInfo.dstAddressesArray.size          = 0;
    cmdInfo.dstAddressesArray.stride        = 0;

    cmd = res.createTempCmdBuffer();

    cmdInfo.input = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    // barrier
    VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memBarrier.srcAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    memBarrier.dstAccessMask   = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         1, &memBarrier, 0, nullptr, 0, nullptr);

    // copy dst addresses
    VkBufferCopy region;
    region.dstOffset = 0;
    region.srcOffset = 0;
    region.size      = sizeof(uint64_t) * numClusters;
    vkCmdCopyBuffer(cmd, dstAddressesBuffer.buffer, geometryTemplate.templateAddresses.buffer, 1, &region);
    region.size = sizeof(uint32_t) * numClusters;
    vkCmdCopyBuffer(cmd, sizesBuffer.buffer, geometryTemplate.templateInstantiationSizes.buffer, 1, &region);

    res.tempSyncSubmit(cmd);
  }

  // delete temp resources
  res.m_allocator.destroyBuffer(scratchBuffer);
  res.m_allocator.destroyBuffer(templateInfosBuffer);
  res.m_allocator.destroyBuffer(instantiateInfosBuffer);
  res.m_allocator.destroyBuffer(sizesBuffer);
  res.m_allocator.destroyBuffer(dstAddressesBuffer);
  res.m_allocator.destroyBuffer(bboxesBuffer);
}

void RayTracingClusterData::initRayTracingInstantiations(Resources& res, Scene& scene, const RendererConfig& config)
{
  {
    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputs.maxAccelerationStructureCount             = m_numTotalClusters;
    inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
    inputs.opMode                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    inputs.opInput.pTriangleClusters = &m_clusterTriangleInput;
    inputs.flags                     = 0;
    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
    m_scratchSize = std::max(m_scratchSize, sizesInfo.buildScratchSize);

    m_clusterDataSize = sizesInfo.accelerationStructureSize;
  }
}

bool RayTracingClusterData::init(Resources& res, Scene& scene, const RendererConfig& config, TessellationTable* tessTable, uint32_t transient2xBatchSize)
{
  VkPhysicalDeviceProperties2                              props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV clusterProps = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};
  props2.pNext = &clusterProps;
  vkGetPhysicalDeviceProperties2(res.m_physicalDevice, &props2);

  m_scratchAlignment = clusterProps.clusterScratchByteAlignment;

  // used for cluster builds or instantiations
  // which do entire scene at once
  m_clusterTriangleInput              = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
  m_clusterTriangleInput.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  m_clusterTriangleInput.maxClusterTriangleCount       = scene.m_maxClusterTriangles;
  m_clusterTriangleInput.maxClusterVertexCount         = scene.m_maxClusterVertices;
  m_clusterTriangleInput.maxTotalTriangleCount         = 0;
  m_clusterTriangleInput.maxTotalVertexCount           = 0;
  m_clusterTriangleInput.minPositionTruncateBitCount   = config.positionTruncateBits;
  m_clusterTriangleInput.maxClusterUniqueGeometryCount = 1;

  m_numTotalClusters = (1u << config.numVisibleClusterBits);

  if(tessTable)
  {
    m_numTotalClusters += (1 << config.numPartTriangleBits);
    m_clusterTriangleInput.maxClusterTriangleCount =
        std::max(m_clusterTriangleInput.maxClusterTriangleCount, tessTable->m_maxTriangles);
    m_clusterTriangleInput.maxClusterVertexCount = std::max(m_clusterTriangleInput.maxClusterVertexCount, tessTable->m_maxVertices);

    // the combined templates can be bigger than a scene cluster or a tessellation table entry
    m_clusterTriangleInput.maxClusterTriangleCount =
        std::max(m_clusterTriangleInput.maxClusterTriangleCount, tessTable->m_combinedMaxTriangles);
    m_clusterTriangleInput.maxClusterVertexCount =
        std::max(m_clusterTriangleInput.maxClusterVertexCount, tessTable->m_combinedMaxVertices);
  }

  // same for the batches of the transient builds
  m_transient2xBatchTriangles = transient2xBatchSize * TESS_2X_TRIANGLES;
  m_transient2xBatchVertices  = transient2xBatchSize * TESS_2X_VERTICES;

  m_clusterTriangleInput.maxClusterTriangleCount =
      std::max(m_clusterTriangleInput.maxClusterTriangleCount, m_transient2xBatchTriangles);
  m_clusterTriangleInput.maxClusterVertexCount = std::max(m_clusterTriangleInput.maxClusterVertexCount, m_transient2xBatchVertices);

  m_clusterTriangleInput.maxTotalVertexCount   = (1u << config.numGeneratedVerticesBits);
  m_clusterTriangleInput.maxTotalTriangleCount = m_numTotalClusters * m_clusterTriangleInput.maxClusterTriangleCount;

  initRayTracingTemplates(res, scene, config);
  initRayTracingInstantiations(res, scene, config);
  initRayTracingBlas(res, scene, config, tessTable ? m_numTotalClusters : scene.m_maxPerGeometryClusters);

  // The per-frame CLAS instantiations and transient CLAS builds are not separated by a barrier and may overlap,
  // hence each uses its own scratch range. The BLAS build comes after a barrier and re-uses the buffer's start.
  // Query with the same explicit mode that the per-frame operations use.
  VkDeviceSize instantiateScratchSize;
  {
    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputs.maxAccelerationStructureCount             = m_numTotalClusters;
    inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
    inputs.opMode                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opInput.pTriangleClusters = &m_clusterTriangleInput;
    inputs.flags                     = config.templateInstantiateFlags;
    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);

    instantiateScratchSize = sizesInfo.buildScratchSize;
  }

  VkDeviceSize scratchBufferSize = std::max(m_scratchSize, instantiateScratchSize);
  if(config.useTransientBuilds())
  {
    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputs.maxAccelerationStructureCount             = m_numTotalClusters;
    inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    inputs.opMode                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opInput.pTriangleClusters = &m_clusterTriangleInput;
    inputs.flags                     = config.clusterBuildFlags;
    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);

    m_scratchTransientOffset = nvutils::align_up(instantiateScratchSize, m_scratchAlignment);
    scratchBufferSize        = std::max(scratchBufferSize, m_scratchTransientOffset + sizesInfo.buildScratchSize);
  }

  res.m_allocator.createBuffer(m_scratchBuffer, scratchBufferSize,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                   | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                               VMA_MEMORY_USAGE_AUTO, {}, m_scratchAlignment);
  m_resourceUsageInfo.operationsMemBytes += m_scratchBuffer.bufferSize;

  if(config.useTransientBuilds())
  {
    // The transient builds look up their worst-case size by the number of triangles they will
    // contain. The "1x" builds stay within a scene cluster, but a "2x" batch can be bigger than
    // that, hence the table has to cover the batch as well.
    const uint32_t maxSizeTriangles = std::max(scene.m_config.clusterTriangles, m_transient2xBatchTriangles);

    m_maxClusterSizes.resize(maxSizeTriangles + 1, 0);

    for(uint32_t t = 1; t <= maxSizeTriangles; t++)
    {
      // A "1x" build re-uses the cluster's vertices, a "2x" batch needs
      // TESS_2X_VERTICES per TESS_2X_TRIANGLES. Take whichever is worse, otherwise we would
      // under-estimate the size and the CLAS would not fit into what we reserved for it.
      uint32_t maxVertices =
          std::max(scene.m_maxClusterVertices, (t * TESS_2X_VERTICES + TESS_2X_TRIANGLES - 1) / TESS_2X_TRIANGLES);

      VkClusterAccelerationStructureTriangleClusterInputNV triangleInput = {
          VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
      triangleInput.vertexFormat                  = VK_FORMAT_R32G32B32_SFLOAT;
      triangleInput.maxClusterTriangleCount       = t;
      triangleInput.maxClusterVertexCount         = maxVertices;
      triangleInput.maxTotalTriangleCount         = t;
      triangleInput.maxTotalVertexCount           = maxVertices;
      triangleInput.minPositionTruncateBitCount   = config.positionTruncateBits;
      triangleInput.maxClusterUniqueGeometryCount = 1;

      VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
      inputs.maxAccelerationStructureCount = 1;
      inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
      inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
      inputs.opInput.pTriangleClusters     = &triangleInput;
      inputs.flags                         = 0;
      VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
      vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);

      m_maxClusterSizes[t] = uint32_t(sizesInfo.accelerationStructureSize);
    }
  }

  return (m_blasDataSize < VkDeviceSize(0xF0000000));
}

void RayTracingClusterData::deinit(Resources& res)
{
  for(auto& it : m_geometryTemplates)
  {
    res.m_allocator.destroyBuffer(it.templateData);
    res.m_allocator.destroyBuffer(it.templateAddresses);
    res.m_allocator.destroyBuffer(it.templateInstantiationSizes);
  }

  res.m_allocator.destroyBuffer(m_scratchBuffer);
}

void RayTracingClusterData::initRayTracingBlas(Resources& res, Scene& scene, const RendererConfig& config, uint32_t maxPerGeometryClusters)
{
  // BLAS space requirement (implicit)
  // the size of the generated blas is dynamic, need to query prebuild info.
  {
    uint32_t numInstances = (uint32_t)m_renderInstances.size();

    m_clusterBlasInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
    m_clusterBlasInput.maxClusterCountPerAccelerationStructure = maxPerGeometryClusters;
    m_clusterBlasInput.maxTotalClusterCount                    = m_numTotalClusters;

    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputs.maxAccelerationStructureCount             = numInstances;
    inputs.opMode                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    inputs.opType                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    inputs.opInput.pClustersBottomLevel = &m_clusterBlasInput;
    inputs.flags                        = config.clusterBlasFlags;

    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
    m_scratchSize = std::max(m_scratchSize, sizesInfo.buildScratchSize);

    m_blasDataSize = sizesInfo.accelerationStructureSize;
  }
}
}  // namespace tessellatedclusters
