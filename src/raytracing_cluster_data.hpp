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

#pragma once

#include "renderer.hpp"
#include "tessellation_table.hpp"
#include "vk_nv_cluster_acc.h"

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
    RBuffer templateData;
    RBuffer templateAddresses;
    RBuffer templateInstantiationSizes;
  };

  RendererConfig m_config;
  uint32_t       m_numTotalClusters = 0;

  std::vector<GeometryTemplate> m_geometryTemplates;

  VkClusterAccelerationStructureTriangleClusterInputNV     m_clusterTriangleInput{};
  VkClusterAccelerationStructureClustersBottomLevelInputNV m_clusterBlasInput{};

  VkDeviceSize m_blasDataSize    = 0;
  VkDeviceSize m_clusterDataSize = 0;

  VkDeviceSize m_scratchSize = 0;
  RBuffer      m_scratchBuffer;

  std::vector<uint32_t> m_maxClusterSizes;

  bool init(Resources& res, Scene& scene, const RendererConfig& config, TessellationTable* tessTable = nullptr);
  void deinit(Resources& res);

  void initRayTracingTemplates(Resources& res, Scene& scene, const RendererConfig& config);
  void initRayTracingInstantiations(Resources& res, Scene& scene, const RendererConfig& config);
  void initRayTracingBlas(Resources& res, Scene& scene, const RendererConfig& config, uint32_t maxPerGeometryClusters);
};

}  // namespace tessellatedclusters
