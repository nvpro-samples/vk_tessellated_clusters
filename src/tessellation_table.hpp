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

  void init(Resources& res, bool withTemplates = false, uint32_t templatePositionTruncateBitCount = 0);
  void initTemplates(Resources& res, uint32_t positionTruncateBitCount);
  void deinit(Resources& res);

  uint32_t getLookupIndex(uint32_t x, uint32_t y, uint32_t z) const
  {
    return x + y * m_maxSizeConfigs + z * (m_maxSizeConfigs * m_maxSizeConfigs)
           - (1 + m_maxSizeConfigs + m_maxSizeConfigs * m_maxSizeConfigs);
  }
};

}  // namespace tessellatedclusters