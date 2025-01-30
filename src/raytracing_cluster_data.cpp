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

#include <nvh/parallel_work.hpp>

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
  templateTriangleInput.maxClusterTriangleCount     = scene.m_config.clusterTriangles;
  templateTriangleInput.maxClusterVertexCount       = scene.m_config.clusterVertices;
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

  RBuffer scratchBuffer = res.createBuffer(tempScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  std::vector<VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV> templateInfos(scene.m_maxPerGeometryClusters);
  std::vector<VkClusterAccelerationStructureInstantiateClusterInfoNV> instantiateInfos(scene.m_maxPerGeometryClusters);

  RBuffer templateInfosBuffer =
      res.createBuffer(sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV) * templateInfos.size(),
                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  RBuffer instantiateInfosBuffer =
      res.createBuffer(sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV) * instantiateInfos.size(),
                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  RBuffer sizesBuffer = res.createBuffer(sizeof(uint32_t) * instantiateInfos.size(),
                                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  RBuffer dstAddressesBuffer =
      res.createBuffer(sizeof(uint64_t) * instantiateInfos.size(),
                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);


  // 32 byte alignment requirement for bbox
  struct TemplateBbox
  {
    shaderio::BBox bbox;
  };

  RBuffer bboxesBuffer = res.createBuffer(sizeof(TemplateBbox) * instantiateInfos.size(),
                                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

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
    cmdInfo.srcInfosArray.size              = templateInfosBuffer.info.range;
    cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV);
    cmdInfo.dstSizesArray.deviceAddress     = sizesBuffer.address;
    cmdInfo.dstSizesArray.size              = sizesBuffer.info.range;
    cmdInfo.dstSizesArray.stride            = sizeof(uint32_t);
    cmdInfo.dstAddressesArray.deviceAddress = dstAddressesBuffer.address;
    cmdInfo.dstAddressesArray.size          = dstAddressesBuffer.info.range;
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
    geometryTemplate.templateData = res.createBuffer(buildSum, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
    m_resourceUsageInfo.rtTemplateMemBytes += geometryTemplate.templateData.info.range;

    geometryTemplate.templateInstantiationSizes =
        res.createBuffer(sizeof(uint32_t) * numClusters, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                                             | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceUsageInfo.operationsMemBytes += geometryTemplate.templateInstantiationSizes.info.range;
    geometryTemplate.templateAddresses =
        res.createBuffer(sizeof(uint64_t) * numClusters, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                                             | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceUsageInfo.operationsMemBytes += geometryTemplate.templateAddresses.info.range;

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
    cmdInfo.srcInfosArray.size              = instantiateInfosBuffer.info.range;
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
  res.destroy(scratchBuffer);
  res.destroy(templateInfosBuffer);
  res.destroy(instantiateInfosBuffer);
  res.destroy(sizesBuffer);
  res.destroy(dstAddressesBuffer);
  res.destroy(bboxesBuffer);
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

bool RayTracingClusterData::init(Resources& res, Scene& scene, const RendererConfig& config, TessellationTable* tessTable)
{
  VkPhysicalDeviceProperties2                              props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV clusterProps = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};
  props2.pNext = &clusterProps;
  vkGetPhysicalDeviceProperties2(res.m_physical, &props2);

  // used for cluster builds or instantiations
  // which do entire scene at once
  m_clusterTriangleInput              = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
  m_clusterTriangleInput.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  m_clusterTriangleInput.maxClusterTriangleCount       = scene.m_config.clusterTriangles;
  m_clusterTriangleInput.maxClusterVertexCount         = scene.m_config.clusterVertices;
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
  }

  m_clusterTriangleInput.maxTotalVertexCount   = (1u << config.numGeneratedVerticesBits);
  m_clusterTriangleInput.maxTotalTriangleCount = m_numTotalClusters * m_clusterTriangleInput.maxClusterTriangleCount;

  initRayTracingTemplates(res, scene, config);
  initRayTracingInstantiations(res, scene, config);
  initRayTracingBlas(res, scene, config, tessTable ? m_numTotalClusters : scene.m_maxPerGeometryClusters);

  {
    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputs.maxAccelerationStructureCount             = m_numTotalClusters;
    inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    inputs.opMode                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    inputs.opInput.pTriangleClusters = &m_clusterTriangleInput;
    inputs.flags                     = config.clusterBuildFlags;
    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);

    m_scratchSize = std::max(m_scratchSize, sizesInfo.buildScratchSize);
  }

  VkDeviceSize scratchBufferActualSize = m_scratchSize;
  if(config.transientClusters1X || m_config.transientClusters2X)
  {
    // we overlap two operations when building transient clusters, ensure m_scratchSize is properly aligned
    m_scratchSize = nvh::align_up(m_scratchSize, VkDeviceSize(clusterProps.clusterScratchByteAlignment));
    // and then double the size we request (this is very conservative)
    scratchBufferActualSize = m_scratchSize * 2;
  }

  m_scratchBuffer = res.createBuffer(scratchBufferActualSize,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                         | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.operationsMemBytes += m_scratchBuffer.info.range;

  {
    m_maxClusterSizes.resize(scene.m_config.clusterTriangles + 1, 0);

    for(uint32_t t = 1; t <= scene.m_config.clusterTriangles; t++)
    {
      VkClusterAccelerationStructureTriangleClusterInputNV triangleInput = {
          VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
      triangleInput.vertexFormat                  = VK_FORMAT_R32G32B32_SFLOAT;
      triangleInput.maxClusterTriangleCount       = t;
      triangleInput.maxClusterVertexCount         = scene.m_config.clusterVertices;
      triangleInput.maxTotalTriangleCount         = t;
      triangleInput.maxTotalVertexCount           = scene.m_config.clusterVertices;
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
    res.destroy(it.templateData);
    res.destroy(it.templateAddresses);
    res.destroy(it.templateInstantiationSizes);
  }

  res.destroy(m_scratchBuffer);
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
