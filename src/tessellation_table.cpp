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

#include <epicgames_tessellation/tessellation_table_epicgames_raw.hpp>
#include <glm/glm.hpp>

#include "tessellation_table.hpp"
#include "shaders/shaderio.h"
#include "vk_nv_cluster_acc.h"

namespace tessellatedclusters {

void TessellationTable::init(Resources& res, bool withTemplates, uint32_t templatePositionTruncateBitCount)
{
  m_maxSize        = tessellation_table::max_edge_segments;
  m_maxSizeConfigs = 1 << (glm::bitCount(m_maxSize) == 1 ? glm::findMSB(m_maxSize) : glm::findMSB(m_maxSize) + 1);
  m_numConfigs     = m_maxSizeConfigs * m_maxSizeConfigs * m_maxSizeConfigs;

  m_vertices = res.createBuffer(sizeof(uint32_t) * tessellation_table::max_vertices, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  m_indices = res.createBuffer(sizeof(uint32_t) * tessellation_table::max_triangles, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  m_configs = res.createBuffer(sizeof(uint16_t) * 4 * m_numConfigs, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  m_ubo     = res.createBuffer(sizeof(shaderio::TessellationTable), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

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
  }

  {
    shaderio::TessellationTable shaderData;
    shaderData.entries                    = m_configs.address;
    shaderData.vertices                   = m_vertices.address;
    shaderData.triangles                  = m_indices.address;
    shaderData.templateAddresses          = m_templateAddresses.address;
    shaderData.templateInstantiationSizes = m_templateInstantiationSizes.address;

    res.simpleUploadBuffer(m_ubo, &shaderData);
  }
}

void TessellationTable::initTemplates(Resources& res, uint32_t positionTruncateBitCount)
{
  // * 2 for flipped version

  RBuffer vertexBuffer = res.createBuffer(sizeof(glm::vec3) * tessellation_table::max_vertices * 2,
                                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  RBuffer indexBuffer = res.createBuffer(sizeof(uint8_t) * tessellation_table::max_triangles * 3 * 2,
                                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

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

  m_templateInstantiationSizes =
      res.createBuffer(sizeof(uint32_t) * m_numConfigs,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  m_templateAddresses = res.createBuffer(sizeof(uint64_t) * m_numConfigs,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                             | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  RBuffer templateSizesStageBuffer =
      res.createBuffer(sizeof(uint32_t) * m_numConfigs, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  RBuffer templateAddressesStageBuffer =
      res.createBuffer(sizeof(uint64_t) * m_numConfigs, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // * 2 because of flipped winding

  RBuffer templateInstInfosBuffer =
      res.createBuffer(sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV) * tessellation_table::max_configs * 2,
                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  RBuffer templateInfosBuffer = res.createBuffer(sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV)
                                                     * tessellation_table::max_configs * 2,
                                                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  RBuffer templateSizesBuffer = res.createBuffer(sizeof(uint32_t) * tessellation_table::max_configs * 2,
                                                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  RBuffer templateAddressesBuffer = res.createBuffer(sizeof(uint32_t) * tessellation_table::max_configs * 2,
                                                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

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

  RBuffer scratchBuffer;

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

    scratchBuffer = res.createBuffer(sizesInfo.buildScratchSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

    // first pass query sizes

    VkCommandBuffer cmd;
    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    cmdInfo.srcInfosArray.deviceAddress = templateInfosBuffer.address;
    cmdInfo.srcInfosArray.size          = templateInfosBuffer.info.range;
    cmdInfo.srcInfosArray.stride        = sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV);
    cmdInfo.dstSizesArray.deviceAddress = templateSizesBuffer.address;
    cmdInfo.dstSizesArray.size          = templateSizesBuffer.info.range;
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

    m_templateData = res.createBuffer(templateDataSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

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
    cmdInfo.dstAddressesArray.size          = templateAddressesBuffer.info.range;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

    cmd = res.createTempCmdBuffer();

    cmdInfo.input = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    res.tempSyncSubmit(cmd);

    // third pass get instantiation sizes
    inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
    inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;

    cmdInfo.srcInfosArray.deviceAddress     = templateInstInfosBuffer.address;
    cmdInfo.srcInfosArray.size              = templateInstInfosBuffer.info.range;
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

  res.destroy(vertexBuffer);
  res.destroy(indexBuffer);
  res.destroy(templateInstInfosBuffer);
  res.destroy(templateInfosBuffer);
  res.destroy(templateAddressesBuffer);
  res.destroy(templateSizesBuffer);
  res.destroy(templateAddressesStageBuffer);
  res.destroy(templateSizesStageBuffer);
  res.destroy(scratchBuffer);
}

void TessellationTable::deinit(Resources& res)
{
  res.destroy(m_ubo);
  res.destroy(m_configs);
  res.destroy(m_vertices);
  res.destroy(m_indices);
  res.destroy(m_templateData);
  res.destroy(m_templateAddresses);
  res.destroy(m_templateInstantiationSizes);
}

}  // namespace tessellatedclusters