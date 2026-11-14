/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif

#include <memory>

#include <nvvk/acceleration_structures.hpp>
#include <nvvk/compute_pipeline.hpp>

#include "resources.hpp"
#include "scene.hpp"

namespace tessellatedclusters {

// How the ray tracing renderer handles clusters with low tessellation factors.
// The two modes are mutually exclusive.
enum LowTessMode : uint32_t
{
  // build a transient CLAS from scratch for them
  LOWTESS_MODE_TRANSIENT_BUILDS = 0,
  // instantiate pre-built templates that combine multiple of them
  LOWTESS_MODE_COMBINED_TEMPLATES = 1,
};

struct RendererConfig
{
  // scene related
  uint32_t  numSceneCopies = 1;
  uint32_t  gridConfig     = 3;
  glm::vec3 refShift       = glm::vec3(1, 1, 1);

  bool debugVisualization = true;
  bool doAnimation        = false;
  bool doCulling          = false;
  bool flipWinding        = false;
  bool pnDisplacement     = true;
  // optimize clusters where all triangles have a tessellation factor of 1x1x1
  bool lowTessClusters1X = true;
  // optimize triangles whose tessellation factors are all <= 2
  bool lowTessClusters2X = true;
  // which mechanism is used for the two options above
  LowTessMode lowTessMode = LOWTESS_MODE_COMBINED_TEMPLATES;
  // number of 2x2x2 patches a transient cluster build batches at most
  uint32_t transient2xBatchSize = 32;
  // number of independent triangles / 2x2x2 patches a combined template stores at most
  uint32_t combined1xBatchSize = 64;
  uint32_t combined2xBatchSize = 16;
  bool     generateHistogram   = false;
  // extra atomic counters for statistics, costs performance
  bool useRenderStats = false;

  bool rasterBatchMeshlets = true;
  bool useEXTmeshShader    = false;
  bool persistentKernel    = false;

  // super resolution in rasterization and ray reconstruction in ray tracing
  bool useDlss = true;
#if USE_DLSS
  NVSDK_NGX_PerfQuality_Value dlssQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
#else
  int dlssQuality = 0;
#endif

  uint32_t positionTruncateBits     = 0;
  uint32_t numVisibleClusterBits    = 20;
  uint32_t numSplitTriangleBits     = 16;
  uint32_t numPartTriangleBits      = 20;
  uint32_t numGeneratedVerticesBits = 24;
  uint32_t numGeneratedClusterMegs  = 1024;

  uint32_t persistentThreads = 0;
  uint32_t splitFactor       = 8;

  VkBuildAccelerationStructureFlagsKHR clusterBlasFlags         = 0;
  VkBuildAccelerationStructureFlagsKHR clusterBuildFlags        = 0;
  VkBuildAccelerationStructureFlagsKHR templateInstantiateFlags = 0;
  VkBuildAccelerationStructureFlagsKHR templateBuildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  float                                templateBBoxBloat  = 0.1f;

  bool useLowTessOptimization() const { return lowTessClusters1X || lowTessClusters2X; }
  bool useTransientBuilds() const { return useLowTessOptimization() && lowTessMode == LOWTESS_MODE_TRANSIENT_BUILDS; }
  bool useCombinedTemplates() const
  {
    return useLowTessOptimization() && lowTessMode == LOWTESS_MODE_COMBINED_TEMPLATES;
  }
  // both store their per-CLAS information at the back of the `partTriangles` array
  bool useLowtessParts() const { return useTransientBuilds() || useCombinedTemplates(); }
};

class Renderer
{
public:
  virtual bool init(Resources& res, Scene& scene, const RendererConfig& config) = 0;
  virtual void render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) = 0;
  virtual void deinit(Resources& res) = 0;
  virtual ~Renderer() {};  // Defined only so that inherited classes also have virtual destructors. Use deinit().
  virtual void updatedFrameBuffer(Resources& res) { updateBasicDescriptors(res); };

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

  inline ResourceUsageInfo getResourceUsage(bool reserved) const
  {
    return reserved ? m_resourceReservedUsage : m_resourceActualUsage;
  };

  // maximum number of triangles a single cluster can be tessellated into by this
  // renderer's tessellation table (0 if the renderer does not tessellate).
  virtual uint32_t getTessellationMaxTriangles() const { return 0; }

protected:
  void initBasics(Resources& res, Scene& scene, const RendererConfig& config);
  void deinitBasics(Resources& res);

  bool initBasicShaders(Resources& res, Scene& scene, const RendererConfig& config);
  void initBasicPipelines(Resources& res, Scene& scene, const RendererConfig& config);
  void updateBasicDescriptors(Resources& res);

  void initRayTracingTlas(Resources& res, Scene& scene, const RendererConfig& config, const VkAccelerationStructureKHR* blas = nullptr);
  void updateRayTracingTlas(VkCommandBuffer cmd, Resources& res, Scene& scene, bool update = false);
  void deinitRayTracingTlas(Resources& res);

  void writeRayTracingDepthBuffer(VkCommandBuffer cmd);
  void writeBackgroundSky(VkCommandBuffer cmd);

  struct BasicShaders
  {
    Shader fullScreenVertexShader;
    Shader fullscreenWriteDepthFragShader;
    Shader fullscreenBackgroundFragShader;
  };

  struct BasicPipelines
  {
    VkPipeline writeDepth{};
    VkPipeline background{};
  };

  BasicShaders   m_basicShaders;
  BasicPipelines m_basicPipelines;

  std::vector<shaderio::RenderInstance> m_renderInstances;
  nvvk::Buffer                          m_renderInstanceBuffer;

  nvvk::Buffer                                m_tlasInstancesBuffer;
  VkAccelerationStructureGeometryKHR          m_tlasGeometry{};
  VkAccelerationStructureBuildGeometryInfoKHR m_tlasBuildInfo{};
  nvvk::Buffer                                m_tlasScratchBuffer;
  nvvk::AccelerationStructure                 m_tlas;

  ResourceUsageInfo m_resourceReservedUsage{};
  ResourceUsageInfo m_resourceActualUsage{};

  nvvk::DescriptorPack m_basicDset;
  VkPipelineLayout     m_basicPipelineLayout{};
};

//////////////////////////////////////////////////////////////////////////

std::unique_ptr<Renderer> makeRendererRasterClustersTess();
std::unique_ptr<Renderer> makeRendererRayTraceClustersTess();

}  // namespace tessellatedclusters
