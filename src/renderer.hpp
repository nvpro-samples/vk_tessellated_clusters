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

#include <memory>

#include "resources.hpp"
#include "scene.hpp"

namespace tessellatedclusters {
struct RendererConfig
{
  // scene related
  uint32_t  numSceneCopies = 1;
  uint32_t  gridConfig     = 3;
  glm::vec3 refShift       = glm::vec3(1, 1, 1);

  bool flipWinding         = false;
  bool pnDisplacement      = true;
  bool transientClusters1X = true;
  bool transientClusters2X = true;

  bool rasterBatchMeshlets = true;

  uint32_t positionTruncateBits     = 0;
  uint32_t numVisibleClusterBits    = 20;
  uint32_t numSplitTriangleBits     = 16;
  uint32_t numPartTriangleBits      = 20;
  uint32_t numGeneratedVerticesBits = 24;
  size_t   numGeneratedClusterMegs  = 1024;

  uint32_t persistentThreads = 0;

  VkBuildAccelerationStructureFlagsKHR clusterBlasFlags         = 0;
  VkBuildAccelerationStructureFlagsKHR clusterBuildFlags        = 0;
  VkBuildAccelerationStructureFlagsKHR templateInstantiateFlags = 0;
  VkBuildAccelerationStructureFlagsKHR templateBuildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  float                                templateBBoxBloat  = 0.1f;
};

class Renderer
{
public:
  struct ResourceUsageInfo
  {
    size_t rtTlasMemBytes{};
    size_t rtBlasMemBytes{};
    size_t rtClasMemBytes{};
    size_t rtTemplateMemBytes{};
    size_t operationsMemBytes{};
    size_t geometryMemBytes;

    void add(const ResourceUsageInfo& other)
    {
      rtTlasMemBytes += other.rtTlasMemBytes;
      rtBlasMemBytes += other.rtBlasMemBytes;
      rtClasMemBytes += other.rtClasMemBytes;
      rtTemplateMemBytes += other.rtTemplateMemBytes;
      operationsMemBytes += other.operationsMemBytes;
      geometryMemBytes += other.geometryMemBytes;
    }
    size_t getTotalSum() const
    {
      return rtTlasMemBytes + rtTemplateMemBytes + rtBlasMemBytes + rtClasMemBytes + operationsMemBytes + geometryMemBytes;
    }
  };

  virtual bool init(Resources& res, Scene& scene, const RendererConfig& config) = 0;
  virtual void render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerVK& profiler) = 0;
  virtual void deinit(Resources& res) = 0;
  virtual ~Renderer(){};  // Defined only so that inherited classes also have virtual destructors. Use deinit().
  virtual void updatedFrameBuffer(Resources& res) { updatedFrameBufferBasics(res); };

  virtual bool supportsClusters() const { return true; }

  inline ResourceUsageInfo getResourceUsage(bool reserved) const
  {
    return reserved ? m_resourceReservedUsage : m_resourceActualUsage;
  };

protected:
  bool initBasicShaders(Resources& res, Scene& scene, const RendererConfig& config);
  void initBasics(Resources& res, Scene& scene, const RendererConfig& config);
  void deinitBasics(Resources& res);

  void updatedFrameBufferBasics(Resources& res);

  void initRayTracingTlas(Resources& res, Scene& scene, const RendererConfig& config, const VkAccelerationStructureKHR* blas = nullptr);
  void updateRayTracingTlas(VkCommandBuffer cmd, Resources& res, Scene& scene, bool update = false);
  void deinitRayTracingTlas(Resources& res);

  void initWriteRayTracingDepthBuffer(Resources& res, Scene& scene, const RendererConfig& config);
  void writeRayTracingDepthBuffer(VkCommandBuffer cmd);

  struct BasicShaders
  {
    nvvk::ShaderModuleID fullScreenVertexShader;
    nvvk::ShaderModuleID fullscreenWriteDepthFragShader;
  };

  BasicShaders m_basicShaders;

  std::vector<shaderio::RenderInstance> m_renderInstances;
  RBuffer                               m_renderInstanceBuffer;

  RBuffer                                     m_tlasInstancesBuffer;
  VkAccelerationStructureGeometryKHR          m_tlasGeometry;
  VkAccelerationStructureBuildGeometryInfoKHR m_tlasBuildInfo;
  RBuffer                                     m_tlasScratchBuffer;
  nvvk::AccelKHR                              m_tlas;

  ResourceUsageInfo m_resourceReservedUsage{};
  ResourceUsageInfo m_resourceActualUsage{};

  nvvk::DescriptorSetContainer m_writeDepthBufferDsetContainer;
  VkPipeline                   m_writeDepthBufferPipeline = nullptr;
};

//////////////////////////////////////////////////////////////////////////

std::unique_ptr<Renderer> makeRendererRasterClustersTess();
std::unique_ptr<Renderer> makeRendererRayTraceClustersTess();

}  // namespace tessellatedclusters
