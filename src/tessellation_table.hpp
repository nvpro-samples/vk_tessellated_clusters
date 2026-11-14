/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "resources.hpp"

namespace tessellatedclusters {

class TessellationTable
{
public:
  struct ConfigEntry
  {
    uint16_t firstTriangle = 0;
    uint16_t firstVertex   = 0;
    uint16_t numTriangles  = 0;
    uint16_t numVertices   = 0;
  };

  uint32_t m_maxSize;
  uint32_t m_maxSizeConfigs;  // power of 2
  uint32_t m_numConfigs;
  uint32_t m_maxVertices  = 0;
  uint32_t m_maxTriangles = 0;

  nvvk::Buffer m_vertices;
  nvvk::Buffer m_indices;
  nvvk::Buffer m_configs;
  nvvk::Buffer m_ubo;

  nvvk::Buffer m_templateAddresses;
  nvvk::Buffer m_templateInstantiationSizes;
  nvvk::Buffer m_templateData;

  uint32_t m_maxClusterSize = 0;

  // "combined templates" store a batch of independent low-tessellated triangles within
  // a single template, see `shaders/tessellation_combined.glsl`.
  // The templates for batches of 1 to m_combined1xBatchSize independent triangles are
  // stored first, followed by those for 1 to m_combined2xBatchSize independent 2x2x2 patches.
  uint32_t m_combined1xBatchSize    = 0;
  uint32_t m_combined2xBatchSize    = 0;
  uint32_t m_combinedMaxVertices    = 0;
  uint32_t m_combinedMaxTriangles   = 0;
  uint32_t m_combinedMaxClusterSize = 0;

  nvvk::Buffer m_combinedTemplateAddresses;
  nvvk::Buffer m_combinedTemplateInstantiationSizes;
  nvvk::Buffer m_combinedTemplateData;

  void init(Resources& res,
            bool       withTemplates                    = false,
            uint32_t   templatePositionTruncateBitCount = 0,
            uint32_t   combined1xBatchSize              = 0,
            uint32_t   combined2xBatchSize              = 0);
  void initTemplates(Resources& res, uint32_t positionTruncateBitCount);
  void initCombinedTemplates(Resources& res, uint32_t positionTruncateBitCount);
  void deinit(Resources& res);

  // number of triangles and vertices that a "combined template" batch of `n` elements uses
  static uint32_t getCombined1xTriangles(uint32_t n) { return n; }
  static uint32_t getCombined1xVertices(uint32_t n) { return n * 3; }
  static uint32_t getCombined2xTriangles(uint32_t n) { return n * TESS_2X_TRIANGLES; }
  static uint32_t getCombined2xVertices(uint32_t n) { return n * TESS_2X_VERTICES; }

  uint32_t getLookupIndex(uint32_t x, uint32_t y, uint32_t z) const
  {
    return x + y * m_maxSizeConfigs + z * (m_maxSizeConfigs * m_maxSizeConfigs)
           - (1 + m_maxSizeConfigs + m_maxSizeConfigs * m_maxSizeConfigs);
  }
};

}  // namespace tessellatedclusters