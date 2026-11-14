/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if 1
// by default we use a table that doesn't have dependencies on Unreal Engine license terms.
#include "tessellation_table_nv_raw.hpp"
#else
// the Unreal Engine derived version of the table can be found here:
// https://github.com/nvpro-samples/vk_tessellated_clusters/blob/92cc4aac8a556e3fde31619230dc33eec360baf1/thirdparty/epicgames_tessellation/tessellation_table_epicgames_raw.hpp
#include <epicgames_tessellation/tessellation_table_epicgames_raw.hpp>
#endif

#include <glm/glm.hpp>

#include "tessellation_table.hpp"
#include "../shaders/shaderio.h"

namespace tessellatedclusters {

void TessellationTable::init(Resources& res, bool withTemplates, uint32_t templatePositionTruncateBitCount, uint32_t combined1xBatchSize, uint32_t combined2xBatchSize)
{
  m_combined1xBatchSize = combined1xBatchSize;
  m_combined2xBatchSize = combined2xBatchSize;

  m_maxSize        = tessellation_table::max_edge_segments;
  m_maxSizeConfigs = 1 << (glm::bitCount(m_maxSize) == 1 ? glm::findMSB(m_maxSize) : glm::findMSB(m_maxSize) + 1);
  m_numConfigs     = m_maxSizeConfigs * m_maxSizeConfigs * m_maxSizeConfigs;

  res.m_allocator.createBuffer(m_vertices, sizeof(uint32_t) * tessellation_table::max_vertices, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  res.m_allocator.createBuffer(m_indices, sizeof(uint32_t) * tessellation_table::max_triangles, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  res.m_allocator.createBuffer(m_configs, sizeof(uint16_t) * 4 * m_numConfigs, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  res.m_allocator.createBuffer(m_ubo, sizeof(shaderio::TessellationTable), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

  VkCommandBuffer cmd = res.createTempCmdBuffer();

  res.simpleUploadBuffer(m_vertices, tessellation_table::vertices);
  res.simpleUploadBuffer(m_indices, tessellation_table::triangles);

  std::vector<ConfigEntry> lookupEntries(m_numConfigs);
  const ConfigEntry*       origEntries = (const ConfigEntry*)tessellation_table::configs;

  // raw order storage
  uint32_t configIdx = 0;
  for(uint32_t x = 1; x <= m_maxSize; x++)
  {
    for(uint32_t y = 1; y <= x; y++)
    {
      for(uint32_t z = 1; z <= y; z++, configIdx++)
      {
        assert(configIdx < tessellation_table::max_configs);
        {
          uint32_t lookUpIdx = getLookupIndex(x, y, z);
          assert(lookUpIdx < m_numConfigs);
          lookupEntries[lookUpIdx] = origEntries[configIdx];
        }

        if(z != y && x > 1)
        {
          uint32_t lookUpIdx = getLookupIndex(x, z, y);
          assert(lookUpIdx < m_numConfigs);
          lookupEntries[lookUpIdx] = origEntries[configIdx];
        }

        m_maxTriangles = std::max(m_maxTriangles, uint32_t(origEntries[configIdx].numTriangles));
        m_maxVertices  = std::max(m_maxVertices, uint32_t(origEntries[configIdx].numVertices));
      }
    }
  }

  res.simpleUploadBuffer(m_configs, lookupEntries.data());

  if(withTemplates)
  {
    initTemplates(res, templatePositionTruncateBitCount);

    if(m_combined1xBatchSize || m_combined2xBatchSize)
    {
      initCombinedTemplates(res, templatePositionTruncateBitCount);
    }
  }

  {
    shaderio::TessellationTable shaderData;
    shaderData.entries                            = m_configs.address;
    shaderData.vertices                           = m_vertices.address;
    shaderData.triangles                          = m_indices.address;
    shaderData.templateAddresses                  = m_templateAddresses.address;
    shaderData.templateInstantiationSizes         = m_templateInstantiationSizes.address;
    shaderData.combinedTemplateAddresses          = m_combinedTemplateAddresses.address;
    shaderData.combinedTemplateInstantiationSizes = m_combinedTemplateInstantiationSizes.address;

    res.simpleUploadBuffer(m_ubo, &shaderData);
  }
}

void TessellationTable::initTemplates(Resources& res, uint32_t positionTruncateBitCount)
{
  // * 2 for flipped version

  nvvk::Buffer vertexBuffer;
  res.m_allocator.createBuffer(vertexBuffer, sizeof(glm::vec3) * tessellation_table::max_vertices * 2,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  nvvk::Buffer indexBuffer;
  res.m_allocator.createBuffer(indexBuffer, sizeof(uint8_t) * tessellation_table::max_triangles * 3 * 2,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  {
    glm::vec3* vertices = (glm::vec3*)vertexBuffer.mapping;
    for(uint32_t v = 0; v < tessellation_table::max_vertices; v++)
    {
      uint32_t  vtx         = tessellation_table::vertices[v];
      glm::vec3 barycentric = glm::vec3(float(vtx & 0xFFFF) / float(1 << 15), float(vtx >> 16) / float(1 << 15), 0.0f);
      vertices[v]           = barycentric;

      // flipped
      barycentric.x                                  = 1.0f - barycentric.x - barycentric.y;
      vertices[v + tessellation_table::max_vertices] = barycentric;
    }
  }

  {
    // two sets for winding flip
    uint8_t* indices = (uint8_t*)indexBuffer.mapping;
    uint32_t tOffset = tessellation_table::max_triangles;

    for(uint32_t t = 0; t < tessellation_table::max_triangles; t++)
    {
      union
      {
        uint32_t raw32;
        uint8_t  triIndices[4];
      };

      raw32 = tessellation_table::triangles[t];

      indices[t * 3 + 0] = triIndices[0];
      indices[t * 3 + 1] = triIndices[1];
      indices[t * 3 + 2] = triIndices[2];

      indices[(t + tOffset) * 3 + 0] = triIndices[0];
      indices[(t + tOffset) * 3 + 1] = triIndices[2];
      indices[(t + tOffset) * 3 + 2] = triIndices[1];
    }
  }


  res.m_allocator.createBuffer(m_templateInstantiationSizes, sizeof(uint32_t) * m_numConfigs,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  res.m_allocator.createBuffer(m_templateAddresses, sizeof(uint64_t) * m_numConfigs,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  nvvk::Buffer templateSizesStageBuffer;
  res.m_allocator.createBuffer(templateSizesStageBuffer, sizeof(uint32_t) * m_numConfigs,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                   | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  nvvk::Buffer templateAddressesStageBuffer;
  res.m_allocator.createBuffer(templateAddressesStageBuffer, sizeof(uint64_t) * m_numConfigs,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                   | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  // * 2 because of flipped winding

  nvvk::Buffer templateInstInfosBuffer;
  res.m_allocator.createBuffer(templateInstInfosBuffer,
                               sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV) * tessellation_table::max_configs * 2,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
  nvvk::Buffer templateInfosBuffer;
  res.m_allocator.createBuffer(templateInfosBuffer,
                               sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV)
                                   * tessellation_table::max_configs * 2,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
  nvvk::Buffer templateSizesBuffer;
  res.m_allocator.createBuffer(templateSizesBuffer, sizeof(uint32_t) * tessellation_table::max_configs * 2,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  nvvk::Buffer templateAddressesBuffer;
  res.m_allocator.createBuffer(templateAddressesBuffer, sizeof(uint64_t) * tessellation_table::max_configs * 2,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  {
    auto* templateInfos = (VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV*)templateInfosBuffer.mapping;
    auto* templateSizes     = (uint32_t*)templateInfosBuffer.mapping;
    auto* templateAddresses = (uint64_t*)templateAddressesBuffer.mapping;

    const ConfigEntry* origEntries  = (const ConfigEntry*)tessellation_table::configs;
    uint32_t           configOffset = tessellation_table::max_configs;

    for(uint32_t configIdx = 0; configIdx < tessellation_table::max_configs; configIdx++)
    {
      VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV& templateInfo = templateInfos[configIdx];

      const ConfigEntry& entry = origEntries[configIdx];

      templateInfo = {0};

      templateInfo.vertexCount   = entry.numVertices;
      templateInfo.triangleCount = entry.numTriangles;

      templateInfo.baseGeometryIndexAndGeometryFlags.geometryFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV;

      templateInfo.indexBuffer       = indexBuffer.address + entry.firstTriangle * 3;
      templateInfo.indexBufferStride = 1;
      templateInfo.indexType         = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;

      templateInfo.vertexBuffer             = vertexBuffer.address + sizeof(glm::vec3) * entry.firstVertex;
      templateInfo.vertexBufferStride       = uint32_t(sizeof(glm::vec3));
      templateInfo.positionTruncateBitCount = positionTruncateBitCount;

      // flipped version
      templateInfos[configIdx + configOffset] = templateInfo;
      templateInfos[configIdx + configOffset].vertexBuffer += sizeof(float) * 3 * tessellation_table::max_vertices;
      templateInfos[configIdx + configOffset].indexBuffer += sizeof(uint8_t) * 3 * tessellation_table::max_triangles;
    }
  }

  nvvk::Buffer scratchBuffer;

  {
    // slightly lower totals because we do one geometry at a time for template builds.
    VkClusterAccelerationStructureTriangleClusterInputNV templateTriangleInput = {
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
    templateTriangleInput.vertexFormat                  = VK_FORMAT_R32G32B32_SFLOAT;
    templateTriangleInput.maxClusterTriangleCount       = m_maxTriangles;
    templateTriangleInput.maxClusterVertexCount         = m_maxVertices;
    templateTriangleInput.maxTotalTriangleCount         = m_maxTriangles * tessellation_table::max_configs * 2;
    templateTriangleInput.maxTotalVertexCount           = m_maxVertices * tessellation_table::max_configs * 2;
    templateTriangleInput.minPositionTruncateBitCount   = 0;
    templateTriangleInput.maxClusterUniqueGeometryCount = 1;

    // following operations are done per cluster in advance
    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputs.maxAccelerationStructureCount             = tessellation_table::max_configs * 2;
    inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
    inputs.opMode                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opInput.pTriangleClusters = &templateTriangleInput;
    inputs.flags                     = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);

    res.m_allocator.createBuffer(scratchBuffer, sizesInfo.buildScratchSize,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

    // first pass query sizes

    VkCommandBuffer cmd;
    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    cmdInfo.srcInfosArray.deviceAddress = templateInfosBuffer.address;
    cmdInfo.srcInfosArray.size          = templateInfosBuffer.bufferSize;
    cmdInfo.srcInfosArray.stride        = sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV);
    cmdInfo.dstSizesArray.deviceAddress = templateSizesBuffer.address;
    cmdInfo.dstSizesArray.size          = templateSizesBuffer.bufferSize;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);
    cmdInfo.scratchData                 = scratchBuffer.address;

    // query size of templates
    inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
    inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;

    cmd = res.createTempCmdBuffer();

    cmdInfo.input = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    res.tempSyncSubmit(cmd);

    // compute output sizes and addresses
    VkDeviceSize templateDataSize = 0;
    for(uint32_t t = 0; t < inputs.maxAccelerationStructureCount; t++)
    {
      templateDataSize += ((uint32_t*)templateSizesBuffer.mapping)[t];
    }

    res.m_allocator.createBuffer(m_templateData, templateDataSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

    templateDataSize = 0;
    // prepare explicit address
    for(uint32_t t = 0; t < inputs.maxAccelerationStructureCount; t++)
    {
      ((uint64_t*)templateAddressesBuffer.mapping)[t] = m_templateData.address + templateDataSize;

      templateDataSize += ((uint32_t*)templateSizesBuffer.mapping)[t];
    }

    // second pass fill explicit
    inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;

    cmdInfo.dstAddressesArray.deviceAddress = templateAddressesBuffer.address;
    cmdInfo.dstAddressesArray.size          = templateAddressesBuffer.bufferSize;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

    cmd = res.createTempCmdBuffer();

    cmdInfo.input = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    res.tempSyncSubmit(cmd);

    // third pass get instantiation sizes
    inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
    inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;

    cmdInfo.srcInfosArray.deviceAddress     = templateInstInfosBuffer.address;
    cmdInfo.srcInfosArray.size              = templateInstInfosBuffer.bufferSize;
    cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV);
    cmdInfo.dstAddressesArray.deviceAddress = 0;
    cmdInfo.dstAddressesArray.size          = 0;
    cmdInfo.dstAddressesArray.stride        = 0;

    {
      auto* templateInstInfos = (VkClusterAccelerationStructureInstantiateClusterInfoNV*)templateInstInfosBuffer.mapping;
      uint32_t configOffset = tessellation_table::max_configs;
      uint32_t configIdx    = 0;
      for(uint32_t configIdx = 0; configIdx < tessellation_table::max_configs * 2; configIdx++)
      {
        templateInstInfos[configIdx]                        = {0};
        templateInstInfos[configIdx].clusterTemplateAddress = ((uint64_t*)templateAddressesBuffer.mapping)[configIdx];
      }
    }
    cmd = res.createTempCmdBuffer();

    cmdInfo.input = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    res.tempSyncSubmit(cmd);

    memset(templateAddressesStageBuffer.mapping, 0, sizeof(uint64_t) * m_numConfigs);
    memset(templateSizesStageBuffer.mapping, 0, sizeof(uint32_t) * m_numConfigs);

    // fill final lookup tables
    uint32_t configOffset = tessellation_table::max_configs;
    uint32_t configIdx    = 0;
    for(uint32_t x = 1; x <= m_maxSize; x++)
    {
      for(uint32_t y = 1; y <= x; y++)
      {
        for(uint32_t z = 1; z <= y; z++, configIdx++)
        {
          // two versions at a time, account for flipped templates

          {
            uint32_t lookUpIdx = getLookupIndex(x, y, z);

            ((uint64_t*)templateAddressesStageBuffer.mapping)[lookUpIdx] = ((uint64_t*)templateAddressesBuffer.mapping)[configIdx];

            uint32_t maxClusterSize = ((uint32_t*)templateSizesBuffer.mapping)[configIdx];
            ((uint32_t*)templateSizesStageBuffer.mapping)[lookUpIdx] = maxClusterSize;
            m_maxClusterSize                                         = std::max(m_maxClusterSize, maxClusterSize);
          }

          if(z != y && x > 1)
          {
            uint32_t lookUpIdx = getLookupIndex(x, z, y);

            ((uint64_t*)templateAddressesStageBuffer.mapping)[lookUpIdx] =
                ((uint64_t*)templateAddressesBuffer.mapping)[configIdx + configOffset];

            uint32_t maxClusterSize = ((uint32_t*)templateSizesBuffer.mapping)[configIdx + configOffset];
            ((uint32_t*)templateSizesStageBuffer.mapping)[lookUpIdx] = maxClusterSize;
            m_maxClusterSize                                         = std::max(m_maxClusterSize, maxClusterSize);
          }
        }
      }
    }

    cmd = res.createTempCmdBuffer();
    VkBufferCopy bufferCopy;
    bufferCopy.dstOffset = 0;
    bufferCopy.srcOffset = 0;
    bufferCopy.size      = sizeof(uint32_t) * m_numConfigs;
    vkCmdCopyBuffer(cmd, templateSizesStageBuffer.buffer, m_templateInstantiationSizes.buffer, 1, &bufferCopy);
    bufferCopy.size = sizeof(uint64_t) * m_numConfigs;
    vkCmdCopyBuffer(cmd, templateAddressesStageBuffer.buffer, m_templateAddresses.buffer, 1, &bufferCopy);

    res.tempSyncSubmit(cmd);
  }

  res.m_allocator.destroyBuffer(vertexBuffer);
  res.m_allocator.destroyBuffer(indexBuffer);
  res.m_allocator.destroyBuffer(templateInstInfosBuffer);
  res.m_allocator.destroyBuffer(templateInfosBuffer);
  res.m_allocator.destroyBuffer(templateAddressesBuffer);
  res.m_allocator.destroyBuffer(templateSizesBuffer);
  res.m_allocator.destroyBuffer(templateAddressesStageBuffer);
  res.m_allocator.destroyBuffer(templateSizesStageBuffer);
  res.m_allocator.destroyBuffer(scratchBuffer);
}

void TessellationTable::initCombinedTemplates(Resources& res, uint32_t positionTruncateBitCount)
{
  // Builds the templates that store a batch of independent low-tessellated triangles,
  // see `shaders/tessellation_combined.glsl` for the layout they use.
  //
  // We build one template per batch size, so that a batch which isn't completely filled
  // doesn't have to instantiate more triangles than it actually uses.
  //
  //   index [0                     .. m_combined1xBatchSize-1] : 1 to N independent triangles
  //   index [m_combined1xBatchSize .. numTemplates-1]          : 1 to N independent 2x2x2 patches

  const uint32_t numTemplates = m_combined1xBatchSize + m_combined2xBatchSize;

  // the 2x2x2 patch layout, must stay in sync with `tessellation_combined.glsl`
  // vertex 0,1,2 are the corners, vertex 3,4,5 the mid points of edge 0,1,2
  const glm::vec3 patchBarycentrics[TESS_2X_VERTICES] = {
      glm::vec3(1, 0, 0),       glm::vec3(0, 1, 0),       glm::vec3(0, 0, 1),
      glm::vec3(0.5f, 0.5f, 0), glm::vec3(0, 0.5f, 0.5f), glm::vec3(0.5f, 0, 0.5f),
  };
  const glm::uvec3 patchTriangles[TESS_2X_TRIANGLES] = {
      glm::uvec3(0, 3, 5),
      glm::uvec3(3, 1, 4),
      glm::uvec3(5, 4, 2),
      glm::uvec3(3, 4, 5),
  };

  // total amount of template input data required
  uint32_t totalVertices  = 0;
  uint32_t totalTriangles = 0;
  for(uint32_t n = 1; n <= m_combined1xBatchSize; n++)
  {
    totalVertices += getCombined1xVertices(n);
    totalTriangles += getCombined1xTriangles(n);
    m_combinedMaxVertices  = std::max(m_combinedMaxVertices, getCombined1xVertices(n));
    m_combinedMaxTriangles = std::max(m_combinedMaxTriangles, getCombined1xTriangles(n));
  }
  for(uint32_t n = 1; n <= m_combined2xBatchSize; n++)
  {
    totalVertices += getCombined2xVertices(n);
    totalTriangles += getCombined2xTriangles(n);
    m_combinedMaxVertices  = std::max(m_combinedMaxVertices, getCombined2xVertices(n));
    m_combinedMaxTriangles = std::max(m_combinedMaxTriangles, getCombined2xTriangles(n));
  }

  nvvk::Buffer vertexBuffer;
  res.m_allocator.createBuffer(vertexBuffer, sizeof(glm::vec3) * totalVertices,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  nvvk::Buffer indexBuffer;
  res.m_allocator.createBuffer(indexBuffer, sizeof(uint8_t) * totalTriangles * 3,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  nvvk::Buffer templateInfosBuffer;
  res.m_allocator.createBuffer(templateInfosBuffer,
                               sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV) * numTemplates,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
  nvvk::Buffer templateInstInfosBuffer;
  res.m_allocator.createBuffer(templateInstInfosBuffer, sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV) * numTemplates,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
  nvvk::Buffer templateSizesBuffer;
  res.m_allocator.createBuffer(templateSizesBuffer, sizeof(uint32_t) * numTemplates,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                   | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
  nvvk::Buffer templateAddressesBuffer;
  res.m_allocator.createBuffer(templateAddressesBuffer, sizeof(uint64_t) * numTemplates,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                   | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VMA_MEMORY_USAGE_CPU_ONLY,
                               VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  res.m_allocator.createBuffer(m_combinedTemplateInstantiationSizes, sizeof(uint32_t) * numTemplates,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                   | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  res.m_allocator.createBuffer(m_combinedTemplateAddresses, sizeof(uint64_t) * numTemplates,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                   | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  {
    // Fill the template inputs.
    //
    // The actual vertex positions of a template are only a hint for the builder, the real
    // positions are provided at instantiation time. We lay the independent triangles /
    // patches out on a regular grid within the unit square, which mimics the spatial
    // distribution of neighboring triangles within a cluster reasonably well.

    glm::vec3* vertices = (glm::vec3*)vertexBuffer.mapping;
    uint8_t*   indices  = (uint8_t*)indexBuffer.mapping;
    auto* templateInfos = (VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV*)templateInfosBuffer.mapping;

    uint32_t vertexOffset   = 0;
    uint32_t triangleOffset = 0;

    for(uint32_t t = 0; t < numTemplates; t++)
    {
      const bool     is1x  = t < m_combined1xBatchSize;
      const uint32_t batch = is1x ? (t + 1) : (t - m_combined1xBatchSize + 1);

      const uint32_t numBatchVertices  = is1x ? getCombined1xVertices(batch) : getCombined2xVertices(batch);
      const uint32_t numBatchTriangles = is1x ? getCombined1xTriangles(batch) : getCombined2xTriangles(batch);
      const uint32_t slotVertices      = is1x ? 3 : TESS_2X_VERTICES;
      const uint32_t slotTriangles     = is1x ? 1 : TESS_2X_TRIANGLES;

      const uint32_t gridCols = uint32_t(std::ceil(std::sqrt(float(batch))));
      const float    cellSize = 1.0f / float(gridCols);
      // leave a little space between the cells
      const float cellScale = cellSize * 0.9f;

      for(uint32_t slot = 0; slot < batch; slot++)
      {
        glm::vec3 cellOrigin = glm::vec3(float(slot % gridCols) * cellSize, float(slot / gridCols) * cellSize, 0.0f);

        // corners of the slot's base triangle within its cell
        glm::vec3 corners[3] = {
            cellOrigin,
            cellOrigin + glm::vec3(cellScale, 0.0f, 0.0f),
            cellOrigin + glm::vec3(0.0f, cellScale, 0.0f),
        };

        for(uint32_t v = 0; v < slotVertices; v++)
        {
          glm::vec3 barycentric = is1x ? glm::vec3(v == 0 ? 1 : 0, v == 1 ? 1 : 0, v == 2 ? 1 : 0) : patchBarycentrics[v];

          vertices[vertexOffset + slot * slotVertices + v] =
              corners[0] * barycentric.x + corners[1] * barycentric.y + corners[2] * barycentric.z;
        }

        for(uint32_t tri = 0; tri < slotTriangles; tri++)
        {
          glm::uvec3 triIndices = is1x ? glm::uvec3(0, 1, 2) : patchTriangles[tri];
          triIndices += slot * slotVertices;

          uint8_t* dstIndices = &indices[(triangleOffset + slot * slotTriangles + tri) * 3];
          dstIndices[0]       = uint8_t(triIndices.x);
          dstIndices[1]       = uint8_t(triIndices.y);
          dstIndices[2]       = uint8_t(triIndices.z);
        }
      }

      VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV& templateInfo = templateInfos[t];
      templateInfo                                                                   = {0};

      templateInfo.vertexCount   = numBatchVertices;
      templateInfo.triangleCount = numBatchTriangles;

      templateInfo.baseGeometryIndexAndGeometryFlags.geometryFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV;

      templateInfo.indexBuffer       = indexBuffer.address + triangleOffset * 3;
      templateInfo.indexBufferStride = 1;
      templateInfo.indexType         = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;

      templateInfo.vertexBuffer             = vertexBuffer.address + sizeof(glm::vec3) * vertexOffset;
      templateInfo.vertexBufferStride       = uint32_t(sizeof(glm::vec3));
      templateInfo.positionTruncateBitCount = positionTruncateBitCount;

      vertexOffset += numBatchVertices;
      triangleOffset += numBatchTriangles;
    }

    assert(vertexOffset == totalVertices && triangleOffset == totalTriangles);
  }

  nvvk::Buffer scratchBuffer;

  {
    VkClusterAccelerationStructureTriangleClusterInputNV templateTriangleInput = {
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
    templateTriangleInput.vertexFormat                  = VK_FORMAT_R32G32B32_SFLOAT;
    templateTriangleInput.maxClusterTriangleCount       = m_combinedMaxTriangles;
    templateTriangleInput.maxClusterVertexCount         = m_combinedMaxVertices;
    templateTriangleInput.maxTotalTriangleCount         = totalTriangles;
    templateTriangleInput.maxTotalVertexCount           = totalVertices;
    templateTriangleInput.minPositionTruncateBitCount   = 0;
    templateTriangleInput.maxClusterUniqueGeometryCount = 1;

    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputs.maxAccelerationStructureCount             = numTemplates;
    inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
    inputs.opMode                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opInput.pTriangleClusters = &templateTriangleInput;
    inputs.flags                     = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);

    res.m_allocator.createBuffer(scratchBuffer, sizesInfo.buildScratchSize,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

    VkCommandBuffer cmd;
    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    cmdInfo.srcInfosArray.deviceAddress = templateInfosBuffer.address;
    cmdInfo.srcInfosArray.size          = templateInfosBuffer.bufferSize;
    cmdInfo.srcInfosArray.stride        = sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV);
    cmdInfo.dstSizesArray.deviceAddress = templateSizesBuffer.address;
    cmdInfo.dstSizesArray.size          = templateSizesBuffer.bufferSize;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);
    cmdInfo.scratchData                 = scratchBuffer.address;

    // first pass query sizes of templates
    inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
    inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;

    cmd = res.createTempCmdBuffer();

    cmdInfo.input = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    res.tempSyncSubmit(cmd);

    // compute output sizes and addresses
    VkDeviceSize templateDataSize = 0;
    for(uint32_t t = 0; t < numTemplates; t++)
    {
      templateDataSize += ((uint32_t*)templateSizesBuffer.mapping)[t];
    }

    res.m_allocator.createBuffer(m_combinedTemplateData, templateDataSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

    templateDataSize = 0;
    // prepare explicit address
    for(uint32_t t = 0; t < numTemplates; t++)
    {
      ((uint64_t*)templateAddressesBuffer.mapping)[t] = m_combinedTemplateData.address + templateDataSize;

      templateDataSize += ((uint32_t*)templateSizesBuffer.mapping)[t];
    }

    // second pass fill explicit
    inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;

    cmdInfo.dstAddressesArray.deviceAddress = templateAddressesBuffer.address;
    cmdInfo.dstAddressesArray.size          = templateAddressesBuffer.bufferSize;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

    cmd = res.createTempCmdBuffer();

    cmdInfo.input = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    res.tempSyncSubmit(cmd);

    // third pass get instantiation sizes
    inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
    inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;

    cmdInfo.srcInfosArray.deviceAddress     = templateInstInfosBuffer.address;
    cmdInfo.srcInfosArray.size              = templateInstInfosBuffer.bufferSize;
    cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV);
    cmdInfo.dstAddressesArray.deviceAddress = 0;
    cmdInfo.dstAddressesArray.size          = 0;
    cmdInfo.dstAddressesArray.stride        = 0;

    {
      auto* templateInstInfos = (VkClusterAccelerationStructureInstantiateClusterInfoNV*)templateInstInfosBuffer.mapping;
      for(uint32_t t = 0; t < numTemplates; t++)
      {
        templateInstInfos[t]                        = {0};
        templateInstInfos[t].clusterTemplateAddress = ((uint64_t*)templateAddressesBuffer.mapping)[t];
      }
    }

    cmd = res.createTempCmdBuffer();

    cmdInfo.input = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    res.tempSyncSubmit(cmd);

    for(uint32_t t = 0; t < numTemplates; t++)
    {
      m_combinedMaxClusterSize = std::max(m_combinedMaxClusterSize, ((uint32_t*)templateSizesBuffer.mapping)[t]);
    }

    // the lookup tables are indexed just like the templates, no remapping required
    cmd = res.createTempCmdBuffer();
    VkBufferCopy bufferCopy;
    bufferCopy.dstOffset = 0;
    bufferCopy.srcOffset = 0;
    bufferCopy.size      = sizeof(uint32_t) * numTemplates;
    vkCmdCopyBuffer(cmd, templateSizesBuffer.buffer, m_combinedTemplateInstantiationSizes.buffer, 1, &bufferCopy);
    bufferCopy.size = sizeof(uint64_t) * numTemplates;
    vkCmdCopyBuffer(cmd, templateAddressesBuffer.buffer, m_combinedTemplateAddresses.buffer, 1, &bufferCopy);

    res.tempSyncSubmit(cmd);
  }

  res.m_allocator.destroyBuffer(vertexBuffer);
  res.m_allocator.destroyBuffer(indexBuffer);
  res.m_allocator.destroyBuffer(templateInfosBuffer);
  res.m_allocator.destroyBuffer(templateInstInfosBuffer);
  res.m_allocator.destroyBuffer(templateSizesBuffer);
  res.m_allocator.destroyBuffer(templateAddressesBuffer);
  res.m_allocator.destroyBuffer(scratchBuffer);
}

void TessellationTable::deinit(Resources& res)
{
  res.m_allocator.destroyBuffer(m_ubo);
  res.m_allocator.destroyBuffer(m_configs);
  res.m_allocator.destroyBuffer(m_vertices);
  res.m_allocator.destroyBuffer(m_indices);
  res.m_allocator.destroyBuffer(m_templateData);
  res.m_allocator.destroyBuffer(m_templateAddresses);
  res.m_allocator.destroyBuffer(m_templateInstantiationSizes);
  res.m_allocator.destroyBuffer(m_combinedTemplateData);
  res.m_allocator.destroyBuffer(m_combinedTemplateAddresses);
  res.m_allocator.destroyBuffer(m_combinedTemplateInstantiationSizes);
}

}  // namespace tessellatedclusters