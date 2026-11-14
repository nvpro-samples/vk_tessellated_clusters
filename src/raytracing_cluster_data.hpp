/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "renderer.hpp"
#include "tessellation_table.hpp"

namespace tessellatedclusters {

//////////////////////////////////////////////////////////////////////////

class RayTracingClusterData
{
public:
  RayTracingClusterData(std::vector<shaderio::RenderInstance>& renderInstances, Renderer::ResourceUsageInfo& resourceUsageInfo)
      : m_renderInstances(renderInstances)
      , m_resourceUsageInfo(resourceUsageInfo)
  {
  }

  std::vector<shaderio::RenderInstance>& m_renderInstances;
  Renderer::ResourceUsageInfo&           m_resourceUsageInfo;

  struct GeometryTemplate
  {
    // persistent data
    nvvk::Buffer templateData;
    nvvk::Buffer templateAddresses;
    nvvk::Buffer templateInstantiationSizes;
  };

  RendererConfig m_config;
  uint32_t       m_numTotalClusters = 0;

  std::vector<GeometryTemplate> m_geometryTemplates;

  VkClusterAccelerationStructureTriangleClusterInputNV     m_clusterTriangleInput{};
  VkClusterAccelerationStructureClustersBottomLevelInputNV m_clusterBlasInput{};

  VkDeviceSize m_blasDataSize    = 0;
  VkDeviceSize m_clusterDataSize = 0;

  VkDeviceSize m_scratchSize      = 0;
  VkDeviceSize m_scratchAlignment = 0;
  // the transient CLAS builds may overlap with the CLAS instantiations and use their own scratch range
  VkDeviceSize m_scratchTransientOffset = 0;
  nvvk::Buffer m_scratchBuffer;

  // worst-case CLAS size per triangle count, indexed by the number of triangles a
  // transient build will contain
  std::vector<uint32_t> m_maxClusterSizes;
  uint32_t              m_transient2xBatchTriangles = 0;
  uint32_t              m_transient2xBatchVertices  = 0;

  bool init(Resources& res, Scene& scene, const RendererConfig& config, TessellationTable* tessTable = nullptr, uint32_t transient2xBatchSize = 0);
  void deinit(Resources& res);

  void initRayTracingTemplates(Resources& res, Scene& scene, const RendererConfig& config);
  void initRayTracingInstantiations(Resources& res, Scene& scene, const RendererConfig& config);
  void initRayTracingBlas(Resources& res, Scene& scene, const RendererConfig& config, uint32_t maxPerGeometryClusters);
};

}  // namespace tessellatedclusters
