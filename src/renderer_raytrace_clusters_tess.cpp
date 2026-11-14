/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <volk.h>

#include <nvvk/acceleration_structures.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/commands.hpp>
#include <fmt/format.h>

#include "renderer.hpp"
#include "raytracing_cluster_data.hpp"

//////////////////////////////////////////////////////////////////////////

namespace tessellatedclusters {

class RendererRayTraceClustersTess : public Renderer
{
public:
  RendererRayTraceClustersTess()
      : m_cluster(m_renderInstances, m_resourceReservedUsage)
  {
  }

  virtual bool init(Resources& res, Scene& scene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) override;
  virtual void deinit(Resources& res) override;
  virtual void updatedFrameBuffer(Resources& res);

  uint32_t getTessellationMaxTriangles() const override { return m_tessTable.m_maxTriangles; }

private:
  bool initShaders(Resources& res, Scene& scene, const RendererConfig& config);

  // the render target, followed by the DLSS guide buffers when DLSS-RR is active
  static constexpr uint32_t RENDER_TARGET_COUNT = 6;

  void     getRenderTargetInfos(Resources& res, VkDescriptorImageInfo* infos) const;
  uint32_t getRenderTargetCount() const;

  bool initRayTracingScene(Resources& res, Scene& scene, const RendererConfig& config);

  void initRayTracingPipeline(Resources& res);

  struct Shaders
  {
    Shader rayGen;
    Shader rayClosestHit;
    Shader rayMiss;
    Shader rayMissAO;
    Shader computeInstancesClassify;
    Shader computeClustersCull;
    Shader computeClusterClassify;
    Shader computeTriangleInstantiate;
    Shader computeTriangleSplit;
    Shader computeBlasClustersInsert;
    Shader computeBlasSetup;
    Shader computeBuildSetup;
  };

  struct Pipelines
  {
    VkPipeline rayTracing                 = nullptr;
    VkPipeline computeInstancesClassify   = nullptr;
    VkPipeline computeClustersCull        = nullptr;
    VkPipeline computeClusterClassify     = nullptr;
    VkPipeline computeTriangleSplit       = nullptr;
    VkPipeline computeTriangleInstantiate = nullptr;
    VkPipeline computeBlasSetup           = nullptr;
    VkPipeline computeBlasClustersInsert  = nullptr;
    VkPipeline computeBuildSetup          = nullptr;
  };

  RendererConfig m_config;

  Shaders   m_shaders;
  Pipelines m_pipelines;

  VkShaderStageFlags   m_stageFlags;
  VkPipelineLayout     m_pipelineLayout{};
  nvvk::DescriptorPack m_dsetPack;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};

  nvvk::SBTGenerator::Regions m_sbtRegions;
  nvvk::Buffer                m_sbtBuffer;

  uint32_t m_maxGeneratedClusters = 0;
  uint32_t m_maxVisibleClusters   = 0;
  uint32_t m_transient2xBatchSize = 0;

  bool m_buildTlas = true;

  nvvk::Buffer            m_sceneBuildBuffer;
  nvvk::Buffer            m_sceneDataBuffer;
  nvvk::LargeBuffer       m_sceneBlasDataBuffer;
  nvvk::LargeBuffer       m_sceneClasDataBuffer;
  nvvk::Buffer            m_sceneSplitBuffer;
  shaderio::SceneBuilding m_sceneBuildShaderio{};

  TessellationTable     m_tessTable;
  RayTracingClusterData m_cluster;
};

bool RendererRayTraceClustersTess::initShaders(Resources& res, Scene& scene, const RendererConfig& config)
{
  shaderc::CompileOptions options = res.makeCompilerOptions();

  options.AddMacroDefinition("CLUSTER_VERTEX_COUNT", fmt::format("{}", scene.m_maxClusterVertices));
  options.AddMacroDefinition("CLUSTER_TRIANGLE_COUNT", fmt::format("{}", scene.m_maxClusterTriangles));
  options.AddMacroDefinition("TESSTABLE_SIZE", fmt::format("{}", m_tessTable.m_maxSize));
  options.AddMacroDefinition("TESSTABLE_LOOKUP_SIZE", fmt::format("{}", m_tessTable.m_maxSizeConfigs));
  options.AddMacroDefinition("TARGETS_RASTERIZATION", fmt::format("{}", 0));
  options.AddMacroDefinition("TESS_USE_PN", fmt::format("{}", config.pnDisplacement ? 1 : 0));
  options.AddMacroDefinition("TESS_USE_1X_TRANSIENTBUILDS",
                             fmt::format("{}", config.useTransientBuilds() && config.lowTessClusters1X ? 1 : 0));
  options.AddMacroDefinition("TESS_USE_2X_TRANSIENTBUILDS",
                             fmt::format("{}", config.useTransientBuilds() && config.lowTessClusters2X ? 1 : 0));
  options.AddMacroDefinition("TESS_USE_1X_COMBINEDTEMPLATES",
                             fmt::format("{}", config.useCombinedTemplates() && config.lowTessClusters1X ? 1 : 0));
  options.AddMacroDefinition("TESS_USE_2X_COMBINEDTEMPLATES",
                             fmt::format("{}", config.useCombinedTemplates() && config.lowTessClusters2X ? 1 : 0));
  options.AddMacroDefinition("TESS_COMBINED_1X_BATCHSIZE", fmt::format("{}", std::max(1u, m_tessTable.m_combined1xBatchSize)));
  options.AddMacroDefinition("TESS_COMBINED_2X_BATCHSIZE", fmt::format("{}", std::max(1u, m_tessTable.m_combined2xBatchSize)));
  options.AddMacroDefinition("TESS_TRANSIENT_2X_BATCHSIZE", fmt::format("{}", std::max(1u, m_transient2xBatchSize)));
  options.AddMacroDefinition("TESS_USE_PERSISTENT_KERNEL", fmt::format("{}", config.persistentKernel ? 1 : 0));
  options.AddMacroDefinition("TESS_MAX_SPLIT_FACTOR",
                             fmt::format("{}", std::max(2u, std::min(config.splitFactor, uint32_t(TESSTABLE_SIZE)))));
  options.AddMacroDefinition("TESS_ACTIVE", fmt::format("{}", 1));
  options.AddMacroDefinition("MAX_PART_TRIANGLES", fmt::format("{}", 1 << config.numPartTriangleBits));
  options.AddMacroDefinition("MAX_VISIBLE_CLUSTERS", fmt::format("{}", 1 << config.numVisibleClusterBits));
  options.AddMacroDefinition("MAX_SPLIT_TRIANGLES", fmt::format("{}", 1 << config.numSplitTriangleBits));
  options.AddMacroDefinition("MAX_GENERATED_CLUSTER_MEGS", fmt::format("{}", uint32_t(config.numGeneratedClusterMegs)));
  options.AddMacroDefinition("MAX_GENERATED_CLUSTERS", fmt::format("{}", m_maxGeneratedClusters));
  options.AddMacroDefinition("MAX_GENERATED_VERTICES", fmt::format("{}", 1 << config.numGeneratedVerticesBits));
  options.AddMacroDefinition("HAS_DISPLACEMENT_TEXTURES", fmt::format("{}", scene.m_textureImages.size() ? 1 : 0));
  options.AddMacroDefinition("DO_CULLING", fmt::format("{}", config.doCulling ? 1 : 0));
  options.AddMacroDefinition("DO_ANIMATION", fmt::format("{}", config.doAnimation ? 1 : 0));
  options.AddMacroDefinition("DEBUG_VISUALIZATION", fmt::format("{}", config.debugVisualization ? 1 : 0));
  options.AddMacroDefinition("TESS_GENERATE_HISTOGRAM", fmt::format("{}", config.generateHistogram ? 1 : 0));
  options.AddMacroDefinition("USE_RENDER_STATS", fmt::format("{}", config.useRenderStats ? 1 : 0));
  options.AddMacroDefinition("USE_DLSS", fmt::format("{}", config.useDlss ? 1 : 0));

  shaderc::CompileOptions optionsAO = options;
  options.AddMacroDefinition("RAYTRACING_PAYLOAD_INDEX", "0");
  optionsAO.AddMacroDefinition("RAYTRACING_PAYLOAD_INDEX", "1");

  res.compileShader(m_shaders.rayGen, VK_SHADER_STAGE_RAYGEN_BIT_KHR, "render_raytrace.rgen.glsl", &options);
  res.compileShader(m_shaders.rayClosestHit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "render_raytrace_clusters.rchit.glsl", &options);
  res.compileShader(m_shaders.rayMiss, VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl", &options, "primary");
  res.compileShader(m_shaders.rayMissAO, VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl", &optionsAO, "ao");
  res.compileShader(m_shaders.computeInstancesClassify, VK_SHADER_STAGE_COMPUTE_BIT, "instances_classify.comp.glsl", &options);
  res.compileShader(m_shaders.computeClustersCull, VK_SHADER_STAGE_COMPUTE_BIT, "clusters_cull.comp.glsl", &options);
  res.compileShader(m_shaders.computeTriangleInstantiate, VK_SHADER_STAGE_COMPUTE_BIT,
                    "triangle_tess_template_instantiate.comp.glsl", &options);
  res.compileShader(m_shaders.computeBlasClustersInsert, VK_SHADER_STAGE_COMPUTE_BIT, "blas_clusters_insert.comp.glsl", &options);
  res.compileShader(m_shaders.computeBlasSetup, VK_SHADER_STAGE_COMPUTE_BIT, "blas_setup_insertion.comp.glsl", &options);
  res.compileShader(m_shaders.computeBuildSetup, VK_SHADER_STAGE_COMPUTE_BIT, "build_setup.comp.glsl", &options);
  // each low tessellation optimization has its own classification kernel, this keeps the
  // amount of permutations per file manageable and makes the files diffable against each other
  const char* clusterClassifyShader = "cluster_classify.comp.glsl";
  if(config.useTransientBuilds())
  {
    clusterClassifyShader = "cluster_classify_transient_lowtess.comp.glsl";
  }
  else if(config.useCombinedTemplates())
  {
    clusterClassifyShader = "cluster_classify_combined_lowtess.comp.glsl";
  }
  res.compileShader(m_shaders.computeClusterClassify, VK_SHADER_STAGE_COMPUTE_BIT, clusterClassifyShader, &options);
  res.compileShader(m_shaders.computeTriangleSplit, VK_SHADER_STAGE_COMPUTE_BIT, "triangle_split.comp.glsl", &options);

  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  return initBasicShaders(res, scene, config);
}

bool RendererRayTraceClustersTess::init(Resources& res, Scene& scene, const RendererConfig& config)
{
  m_config = config;

#if USE_DLSS
  res.setFramebufferDlss(m_config.useDlss ? Resources::DlssMode::eRayReconstruction : Resources::DlssMode::eNone,
                         m_config.dlssQuality);
  // falls back to native rendering if DLSS-RR failed to initialize
  m_config.useDlss = m_config.useDlss && res.m_frameBuffer.dlssMode == Resources::DlssMode::eRayReconstruction;
#else
  m_config.useDlss = false;
#endif

  m_maxGeneratedClusters = (1u << config.numVisibleClusterBits) + (1 << config.numPartTriangleBits);
  m_maxVisibleClusters   = 1u << config.numVisibleClusterBits;

  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m_rtProperties};
  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV propCluster{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};
  propCluster.pNext = &m_rtProperties;
  prop2.pNext       = &propCluster;
  vkGetPhysicalDeviceProperties2(res.m_physicalDevice, &prop2);

  // The batches of the low tessellation optimizations may be bigger than the scene's own clusters,
  // up to what the device supports. Both mechanisms use 8-bit indices, which limits them to
  // 256 vertices no matter how many the device would allow per cluster.
  const uint32_t maxBatchVertices  = std::min(propCluster.maxVerticesPerCluster, 256u);
  const uint32_t maxBatchTriangles = propCluster.maxTrianglesPerCluster;

  // A "2x" batch element stores 4 triangles and 6 vertices. The batching operates within a
  // subgroup, which is also why the batch size must be a power of two.
  auto clamp2xBatchSize = [&](uint32_t batchSize) {
    batchSize = std::clamp(batchSize, 1u, std::min(maxBatchVertices / TESS_2X_VERTICES, maxBatchTriangles / TESS_2X_TRIANGLES));
    batchSize = std::min(batchSize, 32u);
    return 1u << uint32_t(std::log2(float(batchSize)));
  };

  m_transient2xBatchSize = 0;
  if(m_config.useTransientBuilds() && m_config.lowTessClusters2X)
  {
    m_transient2xBatchSize = clamp2xBatchSize(m_config.transient2xBatchSize);
    if(m_transient2xBatchSize != m_config.transient2xBatchSize)
    {
      LOGI("transient batch size clamped to 2x: %d\n", m_transient2xBatchSize);
    }
  }

  uint32_t combined1xBatchSize = 0;
  uint32_t combined2xBatchSize = 0;
  if(m_config.useCombinedTemplates())
  {
    // a "1x" batch element stores 1 triangle and 3 vertices
    combined1xBatchSize = std::clamp(m_config.combined1xBatchSize, 1u, std::min(maxBatchVertices / 3, maxBatchTriangles));
    combined2xBatchSize = clamp2xBatchSize(m_config.combined2xBatchSize);

    // only the actually enabled families need templates
    combined1xBatchSize = m_config.lowTessClusters1X ? combined1xBatchSize : 1;
    combined2xBatchSize = m_config.lowTessClusters2X ? combined2xBatchSize : 1;

    if(combined1xBatchSize != m_config.combined1xBatchSize || combined2xBatchSize != m_config.combined2xBatchSize)
    {
      LOGI("combined template batch sizes clamped to 1x: %d, 2x: %d\n", combined1xBatchSize, combined2xBatchSize);
    }
  }

  m_tessTable.init(res, true, m_config.positionTruncateBits, combined1xBatchSize, combined2xBatchSize);
  m_resourceReservedUsage.rtTemplateMemBytes += m_tessTable.m_templateData.bufferSize;
  m_resourceReservedUsage.operationsMemBytes += m_tessTable.m_templateAddresses.bufferSize;
  m_resourceReservedUsage.operationsMemBytes += m_tessTable.m_templateInstantiationSizes.bufferSize;
  m_resourceReservedUsage.rtTemplateMemBytes += m_tessTable.m_combinedTemplateData.bufferSize;
  m_resourceReservedUsage.operationsMemBytes += m_tessTable.m_combinedTemplateAddresses.bufferSize;
  m_resourceReservedUsage.operationsMemBytes += m_tessTable.m_combinedTemplateInstantiationSizes.bufferSize;

  if(!initShaders(res, scene, m_config))
  {
    m_tessTable.deinit(res);
    return false;
  }

  initBasics(res, scene, m_config);

  if(!initRayTracingScene(res, scene, config))
  {
    LOGI("Resources exceeding max buf GB\n");
    deinit(res);
    return false;
  }

  {
    res.m_allocator.createBuffer(m_sceneBuildBuffer, sizeof(shaderio::SceneBuilding),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

    memset(&m_sceneBuildShaderio, 0, sizeof(m_sceneBuildShaderio));
    m_sceneBuildShaderio.numRenderInstances   = uint32_t(m_renderInstances.size());
    m_sceneBuildShaderio.numBlasReservedSizes = uint32_t(m_cluster.m_blasDataSize);


    // m_sceneBlasDataBuffer has two overlapping memory regions
    // - the blas space itself
    // - and the various temporaries that are used to create CLAS and aren't required after their build

    BufferRanges rangesBlas;
    rangesBlas.beginOverlap();
    m_sceneBuildShaderio.blasBuildData = rangesBlas.append(m_cluster.m_blasDataSize, propCluster.clusterBottomLevelByteAlignment);
    rangesBlas.splitOverlap();
    m_sceneBuildShaderio.genVertices = rangesBlas.append(sizeof(glm::vec3) * uint32_t(1u << config.numGeneratedVerticesBits), 4);
    m_sceneBuildShaderio.tempInstantiations =
        rangesBlas.append(sizeof(shaderio::TemplateInstantiateInfo) * m_maxGeneratedClusters, 16);
    m_sceneBuildShaderio.tempClusterAddresses = rangesBlas.append(sizeof(uint64_t) * m_maxGeneratedClusters, 8);
    m_sceneBuildShaderio.tempClusterSizes     = rangesBlas.append(sizeof(uint32_t) * m_maxGeneratedClusters, 4);
    m_sceneBuildShaderio.tempInstanceIDs      = rangesBlas.append(sizeof(uint32_t) * m_maxGeneratedClusters, 4);
    if(m_config.useTransientBuilds())
    {
      m_sceneBuildShaderio.transBuilds = rangesBlas.append(sizeof(shaderio::ClasBuildInfo) * m_maxGeneratedClusters, 16);
      m_sceneBuildShaderio.transClusterAddresses = rangesBlas.append(sizeof(uint64_t) * m_maxGeneratedClusters, 8);
      m_sceneBuildShaderio.transClusterSizes     = rangesBlas.append(sizeof(uint32_t) * m_maxGeneratedClusters, 4);
      m_sceneBuildShaderio.transInstanceIDs      = rangesBlas.append(sizeof(uint32_t) * m_maxGeneratedClusters, 4);
    }
    rangesBlas.endOverlap();

    res.m_allocator.createLargeBuffer(m_sceneBlasDataBuffer, rangesBlas.tempOffset,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                          | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                      res.m_queue.queue);
    m_resourceReservedUsage.rtBlasMemBytes += m_sceneBlasDataBuffer.bufferSize;

    m_sceneBuildShaderio.blasBuildData += m_sceneBlasDataBuffer.address;
    m_sceneBuildShaderio.genVertices += m_sceneBlasDataBuffer.address;
    m_sceneBuildShaderio.tempInstantiations += m_sceneBlasDataBuffer.address;
    m_sceneBuildShaderio.tempClusterAddresses += m_sceneBlasDataBuffer.address;
    m_sceneBuildShaderio.tempClusterSizes += m_sceneBlasDataBuffer.address;
    m_sceneBuildShaderio.tempInstanceIDs += m_sceneBlasDataBuffer.address;
    if(m_config.useTransientBuilds())
    {
      m_sceneBuildShaderio.transBuilds += m_sceneBlasDataBuffer.address;
      m_sceneBuildShaderio.transClusterAddresses += m_sceneBlasDataBuffer.address;
      m_sceneBuildShaderio.transClusterSizes += m_sceneBlasDataBuffer.address;
      m_sceneBuildShaderio.transInstanceIDs += m_sceneBlasDataBuffer.address;
      m_sceneBuildShaderio.transTriIndices = m_sceneBuildShaderio.genVertices;
    }


    // the main building buffer
    BufferRanges rangesScene;
    m_sceneBuildShaderio.instanceStates = rangesScene.append(sizeof(uint32_t) * m_renderInstances.size(), 4);

    rangesScene.beginOverlap();
    m_sceneBuildShaderio.visibleClusters =
        rangesScene.append(sizeof(shaderio::ClusterInfo) * uint32_t(1u << config.numVisibleClusterBits), 8);
    rangesScene.splitOverlap();
    m_sceneBuildShaderio.blasClusterAddresses = rangesScene.append(sizeof(uint64_t) * m_maxGeneratedClusters, 8);
    rangesScene.endOverlap();

    m_sceneBuildShaderio.partTriangles =
        rangesScene.append(sizeof(shaderio::TessTriangleInfo) * uint32_t(1 << config.numPartTriangleBits), 16);
    m_sceneBuildShaderio.blasBuildInfos = rangesScene.append(sizeof(shaderio::BlasBuildInfo) * m_renderInstances.size(), 16);
    m_sceneBuildShaderio.blasBuildSizes = rangesScene.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
    if(m_config.useTransientBuilds())
    {
      m_sceneBuildShaderio.basicClusterSizes = rangesScene.append(sizeof(uint32_t) * m_cluster.m_maxClusterSizes.size(), 4);
    }

    res.m_allocator.createBuffer(m_sceneDataBuffer, rangesScene.tempOffset,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    m_resourceReservedUsage.operationsMemBytes += m_sceneDataBuffer.bufferSize;

    m_sceneBuildShaderio.instanceStates += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.visibleClusters += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasClusterAddresses += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.partTriangles += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildSizes += m_sceneDataBuffer.address;
    if(m_config.useTransientBuilds())
    {
      res.simpleUploadBuffer(m_sceneDataBuffer, m_sceneBuildShaderio.basicClusterSizes,
                             sizeof(uint32_t) * m_cluster.m_maxClusterSizes.size(), m_cluster.m_maxClusterSizes.data());
      m_sceneBuildShaderio.basicClusterSizes += m_sceneDataBuffer.address;
    }
    if(m_config.useLowtessParts())
    {
      // the low tessellation optimizations store their information
      // at the back of the part triangles, alias this memory with existing
      m_sceneBuildShaderio.lowtessTriMappings = m_sceneBuildShaderio.partTriangles;
    }

    // clas data buffer
    res.m_allocator.createLargeBuffer(m_sceneClasDataBuffer, size_t(config.numGeneratedClusterMegs) * 1024 * 1024,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                      res.m_queue.queue);
    m_resourceReservedUsage.rtClasMemBytes += m_sceneClasDataBuffer.bufferSize;

    m_sceneBuildShaderio.genClusterData = m_sceneClasDataBuffer.address;

    // the buffer that contains recursive splitting information
    res.m_allocator.createBuffer(m_sceneSplitBuffer, sizeof(shaderio::TessTriangleInfo) * uint32_t(1 << config.numSplitTriangleBits),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += m_sceneSplitBuffer.bufferSize;

    m_sceneBuildShaderio.splitTriangles = m_sceneSplitBuffer.address;
  }

  {
    m_stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR
                   | VK_SHADER_STAGE_COMPUTE_BIT;

    nvvk::DescriptorBindings bindings;
    bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_TESSTABLE_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_SCENEBUILDING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_HIZ_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_TLAS, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_RENDER_TARGET, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);
#if USE_DLSS
    if(m_config.useDlss)
    {
      // the DLSS guide buffers follow the render target
      for(uint32_t i = 1; i < DlssDenoiser::eDlssCount; i++)
      {
        bindings.addBinding(BINDINGS_RENDER_TARGET + i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);
      }
    }
#endif
    bindings.addBinding(BINDINGS_RAYTRACING_DEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);

    const uint32_t numDisplacedTextures = uint32_t(scene.m_textureImages.size());
    if(numDisplacedTextures > 0)
    {
      bindings.addBinding(BINDINGS_DISPLACED_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, numDisplacedTextures, m_stageFlags);
    }

    m_dsetPack.init(bindings, res.m_device);
    nvvk::createPipelineLayout(res.m_device, &m_pipelineLayout, {m_dsetPack.getLayout()}, {{m_stageFlags, 0, sizeof(uint32_t)}});

    nvvk::WriteSetContainer writeSets;
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_FRAME_UBO), res.m_commonBuffers.frameConstants);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_TESSTABLE_UBO), m_tessTable.m_ubo);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_READBACK_SSBO), res.m_commonBuffers.readBack);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDERINSTANCES_SSBO), m_renderInstanceBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_SSBO), m_sceneBuildBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_UBO), m_sceneBuildBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX), res.m_hizUpdate.farImageInfo);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_TLAS), m_tlas);

    VkDescriptorImageInfo renderTargetInfos[RENDER_TARGET_COUNT];
    getRenderTargetInfos(res, renderTargetInfos);
    for(uint32_t i = 0; i < getRenderTargetCount(); i++)
    {
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDER_TARGET + i), &renderTargetInfos[i]);
    }

    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RAYTRACING_DEPTH), res.m_frameBuffer.imgRaytracingDepth.descriptor);

    if(numDisplacedTextures > 0)
    {
      std::vector<VkDescriptorImageInfo> imageInfo;
      imageInfo.reserve(numDisplacedTextures + writeSets.size());
      for(const nvvk::Image& texture : scene.m_textureImages)
      {
        VkDescriptorImageInfo descriptor = texture.descriptor;
        descriptor.sampler               = res.m_samplerLinear;
        imageInfo.emplace_back(descriptor);
      }
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_DISPLACED_TEXTURES), imageInfo.data());
    }

    vkUpdateDescriptorSets(res.m_device, writeSets.size(), writeSets.data(), 0, nullptr);
  }

  initRayTracingPipeline(res);

  {
    VkComputePipelineCreateInfo compInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compInfo.stage                       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                 = "main";
    compInfo.layout                      = m_pipelineLayout;

    res.createComputePipeline(compInfo, m_shaders.computeClustersCull, m_pipelines.computeClustersCull);

    res.createComputePipeline(compInfo, m_shaders.computeClusterClassify, m_pipelines.computeClusterClassify);

    res.createComputePipeline(compInfo, m_shaders.computeTriangleSplit, m_pipelines.computeTriangleSplit);

    res.createComputePipeline(compInfo, m_shaders.computeTriangleInstantiate, m_pipelines.computeTriangleInstantiate);

    res.createComputePipeline(compInfo, m_shaders.computeBlasSetup, m_pipelines.computeBlasSetup);

    res.createComputePipeline(compInfo, m_shaders.computeBlasClustersInsert, m_pipelines.computeBlasClustersInsert);

    res.createComputePipeline(compInfo, m_shaders.computeBuildSetup, m_pipelines.computeBuildSetup);

    res.createComputePipeline(compInfo, m_shaders.computeInstancesClassify, m_pipelines.computeInstancesClassify);
  }

  return true;
}


void RendererRayTraceClustersTess::render(VkCommandBuffer cmd, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  m_sceneBuildShaderio.viewPos = frame.freezeCulling ? frame.frameConstantsLast.viewPos : frame.frameConstants.viewPos;
  m_sceneBuildShaderio.positionTruncateBitCount = m_config.positionTruncateBits;

  vkCmdUpdateBuffer(cmd, res.m_commonBuffers.frameConstants.buffer, 0, sizeof(shaderio::FrameConstants) * 2,
                    (const uint32_t*)&frame.frameConstants);
  vkCmdUpdateBuffer(cmd, m_sceneBuildBuffer.buffer, 0, sizeof(shaderio::SceneBuilding), (const uint32_t*)&m_sceneBuildShaderio);
  vkCmdFillBuffer(cmd, res.m_commonBuffers.readBack.buffer, 0, sizeof(shaderio::Readback), 0);
  vkCmdFillBuffer(cmd, m_sceneSplitBuffer.buffer, 0, m_sceneSplitBuffer.bufferSize, ~0);


  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0,
                       nullptr, 0, nullptr);

  res.cmdImageTransition(cmd, res.m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);

  {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Instances Classify");

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeInstancesClassify);

      vkCmdDispatch(cmd, (m_sceneBuildShaderio.numRenderInstances + INSTANCES_CLASSIFY_WORKGROUP - 1) / INSTANCES_CLASSIFY_WORKGROUP,
                    1, 1);


      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Cull");

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeClustersCull);

      for(size_t i = 0; i < m_renderInstances.size(); i++)
      {
        const shaderio::RenderInstance& renderInstance = m_renderInstances[i];
        uint32_t                        instanceId     = uint32_t(i);
        vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &instanceId);
        vkCmdDispatch(cmd, (renderInstance.numClusters + CLUSTERS_CULL_WORKGROUP - 1) / CLUSTERS_CULL_WORKGROUP, 1, 1);
      }

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);

      uint32_t buildSetupID = BUILD_SETUP_CLASSIFY;
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(cmd, 1, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Cluster Classify");

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeClusterClassify);

      vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, dispatchClassify));

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);

      uint32_t buildSetupID = BUILD_SETUP_SPLIT;
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(cmd, 1, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Split");

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTriangleSplit);

      if(m_config.persistentKernel)
      {
        vkCmdDispatch(cmd, (m_config.persistentThreads + TRIANGLE_SPLIT_WORKGROUP - 1) / TRIANGLE_SPLIT_WORKGROUP, 1, 1);
      }
      else
      {
        uint32_t coord = TESSTABLE_COORD_MAX;
        while(coord > m_config.splitFactor)
        {
          coord /= m_config.splitFactor;

          vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, dispatchTriangleSplit));

          if(coord > m_config.splitFactor)
          {
            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                 &memBarrier, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
            uint32_t buildSetupID = BUILD_SETUP_SPLIT_PASS;
            vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
            vkCmdDispatch(cmd, 1, 1, 1);

            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                                 &memBarrier, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTriangleSplit);
          }
        }
      }

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);

      uint32_t buildSetupID = BUILD_SETUP_INSTANTIATE_TESS;
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(cmd, 1, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }


    {
      {
        auto timerSection = profiler.cmdFrameSection(cmd, "PrepInstantiate");

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTriangleInstantiate);

        vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, dispatchTriangleInstantiate));

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);

        uint32_t buildSetupID = BUILD_SETUP_BUILD_BLAS;
        vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
        vkCmdDispatch(cmd, 1, 1, 1);

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }


      {
        auto timerSection = profiler.cmdFrameSection(cmd, "Clas Instantiate / Build");

        VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
        VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

        // setup instantiation inputs
        inputs.maxAccelerationStructureCount = m_maxGeneratedClusters;
        inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
        inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
        inputs.opInput.pTriangleClusters = &m_cluster.m_clusterTriangleInput;
        inputs.flags                     = m_config.templateInstantiateFlags;

        cmdInfo.dstAddressesArray.deviceAddress = m_sceneBuildShaderio.tempClusterAddresses;
        cmdInfo.dstAddressesArray.size          = sizeof(uint64_t) * m_maxGeneratedClusters;
        cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

        cmdInfo.dstSizesArray.deviceAddress = m_sceneBuildShaderio.tempClusterSizes;
        cmdInfo.dstSizesArray.size          = sizeof(uint32_t) * m_maxGeneratedClusters;
        cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

        cmdInfo.dstImplicitData = 0;

        cmdInfo.srcInfosArray.deviceAddress = m_sceneBuildShaderio.tempInstantiations;
        cmdInfo.srcInfosArray.size = sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV) * m_maxGeneratedClusters;
        cmdInfo.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV);

        cmdInfo.srcInfosCount = m_sceneBuildBuffer.address + offsetof(shaderio::SceneBuilding, tempInstantiateCounter);

        cmdInfo.scratchData = m_cluster.m_scratchBuffer.address;
        cmdInfo.input       = inputs;
        vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

        if(m_config.useTransientBuilds())
        {
          //auto timerSection = profiler.cmdFrameSection(primary,"Clas Build");

          VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
          VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

          // setup instantiation inputs
          inputs.maxAccelerationStructureCount = m_maxGeneratedClusters;
          inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
          inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
          inputs.opInput.pTriangleClusters     = &m_cluster.m_clusterTriangleInput;
          inputs.flags                         = m_config.clusterBuildFlags;

          cmdInfo.dstAddressesArray.deviceAddress = m_sceneBuildShaderio.transClusterAddresses;
          cmdInfo.dstAddressesArray.size          = sizeof(uint64_t) * m_maxGeneratedClusters;
          cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

          cmdInfo.dstSizesArray.deviceAddress = m_sceneBuildShaderio.transClusterSizes;
          cmdInfo.dstSizesArray.size          = sizeof(uint32_t) * m_maxGeneratedClusters;
          cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

          cmdInfo.dstImplicitData = 0;

          cmdInfo.srcInfosArray.deviceAddress = m_sceneBuildShaderio.transBuilds;
          cmdInfo.srcInfosArray.size = sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV) * m_maxGeneratedClusters;
          cmdInfo.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV);

          cmdInfo.srcInfosCount = m_sceneBuildBuffer.address + offsetof(shaderio::SceneBuilding, transBuildCounter);

          // separate scratch range used to allow overlap with the instantiation
          cmdInfo.scratchData = m_cluster.m_scratchBuffer.address + m_cluster.m_scratchTransientOffset;
          cmdInfo.input       = inputs;
          vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasSetup);

        vkCmdDispatch(cmd, uint32_t(m_renderInstances.size() + BLAS_BUILD_SETUP_WORKGROUP - 1) / BLAS_BUILD_SETUP_WORKGROUP, 1, 1);

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);
      }

      {
        auto timerSection = profiler.cmdFrameSection(cmd, "Insert");

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasClustersInsert);

        uint32_t specialInstanceID = 0;
        vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &specialInstanceID);
        vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, dispatchBlasTempInsert));

        if(m_config.useTransientBuilds())
        {
          uint32_t specialInstanceID = 1;
          vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &specialInstanceID);
          vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, dispatchBlasTransInsert));
        }

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }

      {
        auto timerSection = profiler.cmdFrameSection(cmd, "Blas Build");

        VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
        VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

        // setup blas inputs
        inputs.maxAccelerationStructureCount = uint32_t(m_renderInstances.size());
        inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
        inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
        inputs.opInput.pClustersBottomLevel  = &m_cluster.m_clusterBlasInput;
        inputs.flags                         = m_config.clusterBlasFlags;

        // we feed the generated blas addresses directly into the ray instances
        cmdInfo.dstAddressesArray.deviceAddress =
            m_tlasInstancesBuffer.address + offsetof(VkAccelerationStructureInstanceKHR, accelerationStructureReference);
        cmdInfo.dstAddressesArray.size   = m_tlasInstancesBuffer.bufferSize;
        cmdInfo.dstAddressesArray.stride = sizeof(VkAccelerationStructureInstanceKHR);

        cmdInfo.dstSizesArray.deviceAddress = m_sceneBuildShaderio.blasBuildSizes;
        cmdInfo.dstSizesArray.size          = sizeof(uint32_t) * m_renderInstances.size();
        cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

        cmdInfo.srcInfosArray.deviceAddress = m_sceneBuildShaderio.blasBuildInfos;
        cmdInfo.srcInfosArray.size =
            sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV) * m_renderInstances.size();
        cmdInfo.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV);

        // in implicit mode we provide one big chunk from which outputs are sub-allocated
        cmdInfo.dstImplicitData = m_sceneBuildShaderio.blasBuildData;

        cmdInfo.scratchData = m_cluster.m_scratchBuffer.address;
        cmdInfo.input       = inputs;
        vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

        memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }

      {
        auto timerSection = profiler.cmdFrameSection(cmd, "Tlas Build");

        updateRayTracingTlas(cmd, res, scene, !m_buildTlas);
        m_buildTlas = false;

        memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }
    }
  }

  // Ray trace
  if(true)
  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Render");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelines.rayTracing);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

    vkCmdTraceRaysKHR(cmd, &m_sbtRegions.raygen, &m_sbtRegions.miss, &m_sbtRegions.hit, &m_sbtRegions.callable,
                      frame.frameConstants.viewport.x, frame.frameConstants.viewport.y, 1);

    res.cmdBeginRendering(cmd, false, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    writeRayTracingDepthBuffer(cmd);
    vkCmdEndRendering(cmd);
  }

  {
    if(!frame.freezeCulling)
    {
      res.cmdBuildHiz(cmd, frame, profiler);
    }
  }

#if USE_DLSS
  if(m_config.useDlss)
  {
    auto timerSection = profiler.cmdFrameSection(cmd, "DLSS-RR");

    // ray tracing wrote the guide buffers, DLSS-RR writes the final color
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
    res.cmdImageTransition(cmd, res.m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);

    // on failure the app disables DLSS before the next frame
    if(!res.m_frameBuffer.dlssDenoiser.denoise(cmd, frame.frameConstants.jitter, frame.frameConstants.viewMatrix,
                                               frame.frameConstants.projMatrix))
    {
      res.m_dlssEvaluateFailed = true;
    }
  }
#endif

  {
    m_resourceActualUsage = m_resourceReservedUsage;

    shaderio::Readback readback;
    res.getReadbackData(readback);
    m_resourceActualUsage.rtBlasMemBytes = readback.numBlasActualSizes;
    m_resourceActualUsage.rtClasMemBytes = readback.numGenActualDatas;
  }
}

void RendererRayTraceClustersTess::deinit(Resources& res)
{
  m_tessTable.deinit(res);

  deinitBasics(res);
  m_cluster.deinit(res);
  deinitRayTracingTlas(res);

  res.m_allocator.destroyBuffer(m_sceneBuildBuffer);
  res.m_allocator.destroyBuffer(m_sceneDataBuffer);
  res.m_allocator.destroyBuffer(m_sceneSplitBuffer);
  res.m_allocator.destroyBuffer(m_sbtBuffer);
  res.m_allocator.destroyLargeBuffer(m_sceneBlasDataBuffer);
  res.m_allocator.destroyLargeBuffer(m_sceneClasDataBuffer);

  res.destroyPipelines(m_pipelines);
  vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);
  m_dsetPack.deinit();

  m_resourceReservedUsage = {};
}

bool RendererRayTraceClustersTess::initRayTracingScene(Resources& res, Scene& scene, const RendererConfig& config)
{
  if(!m_cluster.init(res, scene, config, &m_tessTable, m_transient2xBatchSize))
  {
    return false;
  }

  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    shaderio::RenderInstance& renderInstance = m_renderInstances[i];
    renderInstance.clusterTemplateAdresses = m_cluster.m_geometryTemplates[renderInstance.geometryID].templateAddresses.address;
    renderInstance.clusterTemplateInstantiatonSizes =
        m_cluster.m_geometryTemplates[renderInstance.geometryID].templateInstantiationSizes.address;
  }

  res.simpleUploadBuffer(m_renderInstanceBuffer, m_renderInstances.data());

  // TLAS creation
  initRayTracingTlas(res, scene, config);


  return true;
}

void RendererRayTraceClustersTess::initRayTracingPipeline(Resources& res)
{
  VkDevice device = res.m_device;

  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMissAO,
    eClosestHit,
    eShaderGroupCount
  };
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  std::array<VkShaderModuleCreateInfo, eShaderGroupCount>        stageShaders{};
  for(uint32_t s = 0; s < eShaderGroupCount; s++)
  {
    stageShaders[s].sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  }
  for(uint32_t s = 0; s < eShaderGroupCount; s++)
  {
    stages[s].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[s].pNext = &stageShaders[s];
    stages[s].pName = "main";
  }

  stages[eRaygen].stage              = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stageShaders[eRaygen].codeSize     = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayGen);
  stageShaders[eRaygen].pCode        = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayGen);
  stages[eMiss].stage                = VK_SHADER_STAGE_MISS_BIT_KHR;
  stageShaders[eMiss].codeSize       = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayMiss);
  stageShaders[eMiss].pCode          = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayMiss);
  stages[eMissAO].stage              = VK_SHADER_STAGE_MISS_BIT_KHR;
  stageShaders[eMissAO].codeSize     = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayMissAO);
  stageShaders[eMissAO].pCode        = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayMissAO);
  stages[eClosestHit].stage          = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stageShaders[eClosestHit].codeSize = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayClosestHit);
  stageShaders[eClosestHit].pCode    = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayClosestHit);

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                             .generalShader      = VK_SHADER_UNUSED_KHR,
                                             .closestHitShader   = VK_SHADER_UNUSED_KHR,
                                             .anyHitShader       = VK_SHADER_UNUSED_KHR,
                                             .intersectionShader = VK_SHADER_UNUSED_KHR};

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  shaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  shaderGroups.push_back(group);

  // Miss Ao
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMissAO;
  shaderGroups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  shaderGroups.push_back(group);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{
      .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .flags                        = res.getPipelineCreateFlags(),
      .stageCount                   = uint32_t(eShaderGroupCount),
      .pStages                      = stages.data(),
      .groupCount                   = static_cast<uint32_t>(shaderGroups.size()),
      .pGroups                      = shaderGroups.data(),
      .maxPipelineRayRecursionDepth = 2,
      .layout                       = m_pipelineLayout,
  };

  // NEW for clusters! we need to enable their usage explicitly for a ray tracing pipeline
  VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV pipeClusters = {
      VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV};
  pipeClusters.allowClusterAccelerationStructure = true;

  rayPipelineInfo.pNext = &pipeClusters;

  NVVK_CHECK(vkCreateRayTracingPipelinesKHR(res.m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_pipelines.rayTracing));
  NVVK_DBG_NAME(m_pipelines.rayTracing);
  res.dumpPipelineInternals(m_pipelines.rayTracing, m_shaders.rayGen);

  // Creating the SBT
  {
    // Shader Binding Table (SBT) setup
    nvvk::SBTGenerator sbtGenerator;
    sbtGenerator.init(res.m_device, m_rtProperties);

    // Prepare SBT data from ray pipeline
    size_t bufferSize = sbtGenerator.calculateSBTBufferSize(m_pipelines.rayTracing, rayPipelineInfo);

    // Create SBT buffer using the size from above
    NVVK_CHECK(res.m_allocator.createBuffer(m_sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR,
                                            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, sbtGenerator.getBufferAlignment()));
    NVVK_DBG_NAME(m_sbtBuffer.buffer);

    nvvk::StagingUploader uploader;
    uploader.init(&res.m_allocator);

    void* mapping = nullptr;
    NVVK_CHECK(uploader.appendBufferMapping(m_sbtBuffer, 0, bufferSize, mapping));
    NVVK_CHECK(sbtGenerator.populateSBTBuffer(m_sbtBuffer.address, bufferSize, mapping));

    VkCommandBuffer cmd = res.createTempCmdBuffer();
    uploader.cmdUploadAppended(cmd);
    res.tempSyncSubmit(cmd);
    uploader.deinit();

    // Retrieve the regions, which are using addresses based on the m_sbtBuffer.address
    m_sbtRegions = sbtGenerator.getSBTRegions();

    sbtGenerator.deinit();
  }
}

std::unique_ptr<Renderer> makeRendererRayTraceClustersTess()
{
  return std::make_unique<RendererRayTraceClustersTess>();
}

void RendererRayTraceClustersTess::getRenderTargetInfos(Resources& res, VkDescriptorImageInfo* infos) const
{
#if USE_DLSS
  if(m_config.useDlss)
  {
    // render into the DLSS-RR input
    for(uint32_t i = 0; i < DlssDenoiser::eDlssCount; i++)
    {
      infos[i] = res.m_frameBuffer.dlssDenoiser.getDescriptorImageInfo(DlssDenoiser::DlssBufferType(i));
    }
    return;
  }
#endif
  infos[0]             = {};
  infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  infos[0].imageView   = res.m_frameBuffer.imgColor.descriptor.imageView;
}

uint32_t RendererRayTraceClustersTess::getRenderTargetCount() const
{
#if USE_DLSS
  if(m_config.useDlss)
  {
    return DlssDenoiser::eDlssCount;
  }
#endif
  return 1;
}

void RendererRayTraceClustersTess::updatedFrameBuffer(Resources& res)
{
  nvvk::WriteSetContainer writeSets;

  VkDescriptorImageInfo renderTargetInfos[RENDER_TARGET_COUNT];
  getRenderTargetInfos(res, renderTargetInfos);
  for(uint32_t i = 0; i < getRenderTargetCount(); i++)
  {
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDER_TARGET + i), &renderTargetInfos[i]);
  }
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_RAYTRACING_DEPTH), res.m_frameBuffer.imgRaytracingDepth.descriptor);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX), res.m_hizUpdate.farImageInfo);

  vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);

  Renderer::updatedFrameBuffer(res);
}

}  // namespace tessellatedclusters
