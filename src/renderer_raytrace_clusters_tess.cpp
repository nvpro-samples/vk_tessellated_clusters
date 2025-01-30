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

#include <nvvk/raytraceKHR_vk.hpp>
#include <nvvk/sbtwrapper_vk.hpp>
#include <nvvkhl/pipeline_container.hpp>
#include <nvvk/images_vk.hpp>
#include <nvh/parallel_work.hpp>
#include <nvh/misc.hpp>

#include "renderer.hpp"
#include "vk_nv_cluster_acc.h"
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
  virtual void render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerVK& profiler) override;
  virtual void deinit(Resources& res) override;
  virtual void updatedFrameBuffer(Resources& res);

private:
  bool initShaders(Resources& res, Scene& scene, const RendererConfig& config);

  bool initRayTracingScene(Resources& res, Scene& scene, const RendererConfig& config);

  void initRayTracingPipeline(Resources& res);

  struct Shaders
  {
    nvvk::ShaderModuleID rayGen;
    nvvk::ShaderModuleID closestHit;
    nvvk::ShaderModuleID miss;
    nvvk::ShaderModuleID missAO;

    nvvk::ShaderModuleID computeInstancesClassify;
    nvvk::ShaderModuleID computeClustersCull;
    nvvk::ShaderModuleID computeClusterClassify;
    nvvk::ShaderModuleID computeTriangleInstantiate;
    nvvk::ShaderModuleID computeTriangleSplit;
    nvvk::ShaderModuleID computeBlasClustersInsert;
    nvvk::ShaderModuleID computeBlasSetup;
    nvvk::ShaderModuleID computeBuildSetup;
  };

  struct Pipelines
  {
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

  Shaders                      m_shaders;
  VkShaderStageFlags           m_stageFlags{};
  Pipelines                    m_pipelines;
  nvvk::DescriptorSetContainer m_dsetContainer;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::SBTWrapper          m_rtSbt;   // Shading binding table wrapper
  nvvkhl::PipelineContainer m_rtPipe;  // Hold pipelines and layout

  uint32_t m_maxGeneratedClusters = 0;
  uint32_t m_maxVisibleClusters   = 0;

  bool m_buildTlas = true;

  RBuffer                 m_sceneBuildBuffer;
  RBuffer                 m_sceneDataBuffer;
  RLargeBuffer            m_sceneBlasDataBuffer;
  RLargeBuffer            m_sceneClasDataBuffer;
  RBuffer                 m_sceneSplitBuffer;
  shaderio::SceneBuilding m_sceneBuildShaderio{};

  TessellationTable     m_tessTable;
  RayTracingClusterData m_cluster;
};

bool RendererRayTraceClustersTess::initShaders(Resources& res, Scene& scene, const RendererConfig& config)
{
  std::string prepend;
  prepend += nvh::stringFormat("#define CLUSTER_VERTEX_COUNT %d\n", shaderio::adjustClusterProperty(scene.m_config.clusterVertices));
  prepend += nvh::stringFormat("#define CLUSTER_TRIANGLE_COUNT %d\n",
                               shaderio::adjustClusterProperty(scene.m_config.clusterTriangles));
  prepend += nvh::stringFormat("#define TESSTABLE_SIZE %d\n", m_tessTable.m_maxSize);
  prepend += nvh::stringFormat("#define TESSTABLE_LOOKUP_SIZE %d\n", m_tessTable.m_maxSizeConfigs);
  prepend += nvh::stringFormat("#define TARGETS_RASTERIZATION %d\n", 0);
  prepend += nvh::stringFormat("#define TESS_USE_PN %d\n", config.pnDisplacement ? 1 : 0);
  prepend += nvh::stringFormat("#define TESS_USE_1X_TRANSIENTBUILDS %d\n", config.transientClusters1X ? 1 : 0);
  prepend += nvh::stringFormat("#define TESS_USE_2X_TRANSIENTBUILDS %d\n", config.transientClusters2X ? 1 : 0);
  prepend += nvh::stringFormat("#define TESS_ACTIVE %d\n", 1);
  prepend += nvh::stringFormat("#define MAX_PART_TRIANGLES %d\n", 1 << config.numPartTriangleBits);
  prepend += nvh::stringFormat("#define MAX_VISIBLE_CLUSTERS %d\n", 1 << config.numVisibleClusterBits);
  prepend += nvh::stringFormat("#define MAX_SPLIT_TRIANGLES %d\n", 1 << config.numSplitTriangleBits);
  prepend += nvh::stringFormat("#define MAX_GENERATED_CLUSTER_MEGS %d\n", uint32_t(config.numGeneratedClusterMegs));
  prepend += nvh::stringFormat("#define MAX_GENERATED_CLUSTERS %d\n", m_maxGeneratedClusters);
  prepend += nvh::stringFormat("#define MAX_GENERATED_VERTICES %d\n", 1 << config.numGeneratedVerticesBits);
  prepend += nvh::stringFormat("#define HAS_DISPLACEMENT_TEXTURES %d\n", scene.m_textureImages.size() ? 1 : 0);

  m_shaders.rayGen = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_RAYGEN_BIT_KHR, "render_raytrace.rgen.glsl");
  m_shaders.closestHit = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                                                                "render_raytrace_clusters.rchit.glsl", prepend);
  m_shaders.miss   = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl",
                                                            "#define RAYTRACING_PAYLOAD_INDEX 0\n");
  m_shaders.missAO = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl",
                                                            "#define RAYTRACING_PAYLOAD_INDEX 1\n");

  m_shaders.computeInstancesClassify =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "instances_classify.comp.glsl", prepend);
  m_shaders.computeClustersCull =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "clusters_cull.comp.glsl", prepend);
  m_shaders.computeTriangleInstantiate =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "triangle_tess_template_instantiate.comp.glsl", prepend);
  m_shaders.computeBlasClustersInsert =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "blas_clusters_insert.comp.glsl", prepend);
  m_shaders.computeBlasSetup =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "blas_setup_insertion.comp.glsl", prepend);
  m_shaders.computeBuildSetup =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "build_setup.comp.glsl", prepend);
  m_shaders.computeClusterClassify =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "cluster_classify.comp.glsl", prepend);
  m_shaders.computeTriangleSplit =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "triangle_split.comp.glsl", prepend);

  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  return initBasicShaders(res, scene, config);
}

bool RendererRayTraceClustersTess::init(Resources& res, Scene& scene, const RendererConfig& config)
{
  m_config = config;

  m_maxGeneratedClusters = (1u << config.numVisibleClusterBits) + (1 << config.numPartTriangleBits);
  m_maxVisibleClusters   = 1u << config.numVisibleClusterBits;

  m_tessTable.init(res, true, m_config.positionTruncateBits);
  m_resourceReservedUsage.rtTemplateMemBytes += m_tessTable.m_templateData.info.range;
  m_resourceReservedUsage.operationsMemBytes += m_tessTable.m_templateAddresses.info.range;
  m_resourceReservedUsage.operationsMemBytes += m_tessTable.m_templateInstantiationSizes.info.range;

  if(!initShaders(res, scene, config))
  {
    m_tessTable.deinit(res);
    return false;
  }

  initBasics(res, scene, config);

  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m_rtProperties};
  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV propCluster{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};
  propCluster.pNext = &m_rtProperties;
  prop2.pNext       = &propCluster;
  vkGetPhysicalDeviceProperties2(res.m_physical, &prop2);

  if(!initRayTracingScene(res, scene, config))
  {
    LOGI("Resources exceeding max buf GB\n");
    deinit(res);
    return false;
  }

  {
    m_sceneBuildBuffer = res.createBuffer(sizeof(shaderio::SceneBuilding), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                                                               | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

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
    if(m_config.transientClusters1X || m_config.transientClusters2X)
    {
      m_sceneBuildShaderio.transBuilds = rangesBlas.append(sizeof(shaderio::ClasBuildInfo) * m_maxGeneratedClusters, 16);
      m_sceneBuildShaderio.transClusterAddresses = rangesBlas.append(sizeof(uint64_t) * m_maxGeneratedClusters, 8);
      m_sceneBuildShaderio.transClusterSizes     = rangesBlas.append(sizeof(uint32_t) * m_maxGeneratedClusters, 4);
      m_sceneBuildShaderio.transInstanceIDs      = rangesBlas.append(sizeof(uint32_t) * m_maxGeneratedClusters, 4);
    }
    rangesBlas.endOverlap();

    m_sceneBlasDataBuffer =
        res.createLargeBuffer(rangesBlas.tempOffset, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                         | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    m_resourceReservedUsage.rtBlasMemBytes += m_sceneBlasDataBuffer.info.range;

    m_sceneBuildShaderio.blasBuildData += m_sceneBlasDataBuffer.address;
    m_sceneBuildShaderio.genVertices += m_sceneBlasDataBuffer.address;
    m_sceneBuildShaderio.tempInstantiations += m_sceneBlasDataBuffer.address;
    m_sceneBuildShaderio.tempClusterAddresses += m_sceneBlasDataBuffer.address;
    m_sceneBuildShaderio.tempClusterSizes += m_sceneBlasDataBuffer.address;
    m_sceneBuildShaderio.tempInstanceIDs += m_sceneBlasDataBuffer.address;
    if(m_config.transientClusters1X || m_config.transientClusters2X)
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
    if(m_config.transientClusters1X || m_config.transientClusters2X)
    {
      m_sceneBuildShaderio.basicClusterSizes = rangesScene.append(sizeof(uint32_t) * m_cluster.m_maxClusterSizes.size(), 4);
    }

    m_sceneDataBuffer = res.createBuffer(rangesScene.tempOffset, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    m_resourceReservedUsage.operationsMemBytes += m_sceneDataBuffer.info.range;

    m_sceneBuildShaderio.instanceStates += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.visibleClusters += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasClusterAddresses += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.partTriangles += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildSizes += m_sceneDataBuffer.address;
    if(m_config.transientClusters1X || m_config.transientClusters2X)
    {
      res.simpleUploadBuffer(m_sceneDataBuffer, m_sceneBuildShaderio.basicClusterSizes,
                             sizeof(uint32_t) * m_cluster.m_maxClusterSizes.size(), m_cluster.m_maxClusterSizes.data());
      m_sceneBuildShaderio.basicClusterSizes += m_sceneDataBuffer.address;

      // alias this memory with existing
      m_sceneBuildShaderio.transTriMappings = m_sceneBuildShaderio.partTriangles;
    }

    // clas data buffer
    m_sceneClasDataBuffer =
        res.createLargeBuffer(size_t(config.numGeneratedClusterMegs) * 1024 * 1024,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    m_resourceReservedUsage.rtClasMemBytes += m_sceneClasDataBuffer.info.range;

    m_sceneBuildShaderio.genClusterData = m_sceneClasDataBuffer.address;

    // the buffer that contains recursive splitting information
    m_sceneSplitBuffer = res.createBuffer(sizeof(shaderio::TessTriangleInfo) * uint32_t(1 << config.numSplitTriangleBits),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += m_sceneSplitBuffer.info.range;

    m_sceneBuildShaderio.splitTriangles = m_sceneSplitBuffer.address;
  }

  {
    m_dsetContainer.init(res.m_device);

    m_stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR
                   | VK_SHADER_STAGE_COMPUTE_BIT;

    m_dsetContainer.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_TESSTABLE_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_SCENEBUILDING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_HIZ_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_TLAS, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_RENDER_TARGET, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_RAYTRACING_DEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);

    const uint32_t numDisplacedTextures = uint32_t(scene.m_textureImages.size());
    if(numDisplacedTextures > 0)
    {
      m_dsetContainer.addBinding(BINDINGS_DISPLACED_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                 numDisplacedTextures, m_stageFlags);
    }

    m_dsetContainer.initLayout();

    VkPushConstantRange pushRange;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(uint32_t);
    pushRange.stageFlags = m_stageFlags;
    m_dsetContainer.initPipeLayout(1, &pushRange);

    m_dsetContainer.initPool(1);
    std::vector<VkWriteDescriptorSet> writeSets;
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_FRAME_UBO, &res.m_common.view.info));
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_TESSTABLE_UBO, &m_tessTable.m_ubo.info));
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_READBACK_SSBO, &res.m_common.readbackDevice.info));
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_RENDERINSTANCES_SSBO, &m_renderInstanceBuffer.info));
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_SCENEBUILDING_SSBO, &m_sceneBuildBuffer.info));
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_SCENEBUILDING_UBO, &m_sceneBuildBuffer.info));
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_HIZ_TEX, &res.m_hizUpdate.farImageInfo));

    VkWriteDescriptorSetAccelerationStructureKHR accelInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    accelInfo.accelerationStructureCount = 1;
    VkAccelerationStructureKHR accel     = m_tlas.accel;
    accelInfo.pAccelerationStructures    = &accel;
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_TLAS, &accelInfo));

    VkDescriptorImageInfo renderTargetInfo;
    renderTargetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    renderTargetInfo.imageView   = res.m_framebuffer.viewColor;
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_RENDER_TARGET, &renderTargetInfo));

    VkDescriptorImageInfo raytracingDepthInfo;
    raytracingDepthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    raytracingDepthInfo.imageView   = res.m_framebuffer.viewRaytracingDepth;
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_RAYTRACING_DEPTH, &raytracingDepthInfo));

    std::vector<VkDescriptorImageInfo> imageInfo;
    imageInfo.reserve(numDisplacedTextures + writeSets.size());
    if(numDisplacedTextures > 0)
    {
      for(const nvvk::Texture& texture : scene.m_textureImages)  // All texture samplers
      {
        imageInfo.emplace_back(texture.descriptor);
      }
      writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_DISPLACED_TEXTURES, imageInfo.data()));
    }

    vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);
  }

  initRayTracingPipeline(res);

  {
    VkComputePipelineCreateInfo compInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compInfo.stage                       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                 = "main";
    compInfo.layout                      = m_dsetContainer.getPipeLayout();
    compInfo.flags                       = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeClustersCull);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeClustersCull);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeClusterClassify);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeClusterClassify);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeTriangleSplit);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTriangleSplit);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeTriangleInstantiate);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTriangleInstantiate);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeBlasSetup);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBlasSetup);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeBlasClustersInsert);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBlasClustersInsert);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeBuildSetup);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBuildSetup);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeInstancesClassify);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeInstancesClassify);
  }

  return true;
}


void RendererRayTraceClustersTess::render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerVK& profiler)
{
  m_sceneBuildShaderio.viewPos = frame.freezeCulling ? frame.frameConstantsLast.viewPos : frame.frameConstants.viewPos;
  m_sceneBuildShaderio.positionTruncateBitCount = m_config.positionTruncateBits;

  vkCmdUpdateBuffer(primary, res.m_common.view.buffer, 0, sizeof(shaderio::FrameConstants) * 2, (const uint32_t*)&frame.frameConstants);
  vkCmdUpdateBuffer(primary, m_sceneBuildBuffer.buffer, 0, sizeof(shaderio::SceneBuilding), (const uint32_t*)&m_sceneBuildShaderio);
  vkCmdFillBuffer(primary, res.m_common.readbackDevice.buffer, 0, sizeof(shaderio::Readback), 0);
  vkCmdFillBuffer(primary, m_sceneSplitBuffer.buffer, 0, m_sceneSplitBuffer.info.range, ~0);


  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier,
                       0, nullptr, 0, nullptr);

  res.cmdImageTransition(primary, res.m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);

  {
    vkCmdBindDescriptorSets(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_dsetContainer.getPipeLayout(), 0, 1,
                            m_dsetContainer.getSets(), 0, nullptr);

    {
      auto timerSection = profiler.timeRecurring("Instances Classify", primary);

      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeInstancesClassify);

      vkCmdDispatch(primary, (m_sceneBuildShaderio.numRenderInstances + INSTANCES_CLASSIFY_WORKGROUP - 1) / INSTANCES_CLASSIFY_WORKGROUP,
                    1, 1);


      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    {
      auto timerSection = profiler.timeRecurring("Cull", primary);

      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeClustersCull);

      for(size_t i = 0; i < m_renderInstances.size(); i++)
      {
        const shaderio::RenderInstance& renderInstance = m_renderInstances[i];
        uint32_t                        instanceId     = uint32_t(i);
        vkCmdPushConstants(primary, m_dsetContainer.getPipeLayout(), m_stageFlags, 0, sizeof(uint32_t), &instanceId);
        vkCmdDispatch(primary, (renderInstance.numClusters + CLUSTERS_CULL_WORKGROUP - 1) / CLUSTERS_CULL_WORKGROUP, 1, 1);
      }

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);

      uint32_t buildSetupID = BUILD_SETUP_CLASSIFY;
      vkCmdPushConstants(primary, m_dsetContainer.getPipeLayout(), m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(primary, 1, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    {
      auto timerSection = profiler.timeRecurring("Cluster Classify", primary);

      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeClusterClassify);

      vkCmdDispatchIndirect(primary, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, dispatchClassify));

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);

      uint32_t buildSetupID = BUILD_SETUP_SPLIT;
      vkCmdPushConstants(primary, m_dsetContainer.getPipeLayout(), m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(primary, 1, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    {
      auto timerSection = profiler.timeRecurring("Split", primary);

      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTriangleSplit);

      vkCmdDispatch(primary, (m_config.persistentThreads + TRIANGLE_SPLIT_WORKGROUP - 1) / TRIANGLE_SPLIT_WORKGROUP, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);

      uint32_t buildSetupID = BUILD_SETUP_INSTANTIATE_TESS;
      vkCmdPushConstants(primary, m_dsetContainer.getPipeLayout(), m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(primary, 1, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }


    {
      {
        auto timerSection = profiler.timeRecurring("PrepInstantiate", primary);

        vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTriangleInstantiate);

        vkCmdDispatchIndirect(primary, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, dispatchTriangleInstantiate));

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);

        uint32_t buildSetupID = BUILD_SETUP_BUILD_BLAS;
        vkCmdPushConstants(primary, m_dsetContainer.getPipeLayout(), m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
        vkCmdDispatch(primary, 1, 1, 1);

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }


      {
        auto timerSection = profiler.timeRecurring("Clas Instantiate / Build", primary);

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
        vkCmdBuildClusterAccelerationStructureIndirectNV(primary, &cmdInfo);

        if(m_config.transientClusters1X || m_config.transientClusters2X)
        {
          //auto timerSection = profiler.timeRecurring("Clas Build", primary);

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

          // separate scratch space used to allow overlap with previous build
          cmdInfo.scratchData = m_cluster.m_scratchBuffer.address + m_cluster.m_scratchSize;
          cmdInfo.input       = inputs;
          vkCmdBuildClusterAccelerationStructureIndirectNV(primary, &cmdInfo);
        }

        vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasSetup);

        vkCmdDispatch(primary, uint32_t(m_renderInstances.size() + BLAS_BUILD_SETUP_WORKGROUP - 1) / BLAS_BUILD_SETUP_WORKGROUP,
                      1, 1);

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);
      }

      {
        auto timerSection = profiler.timeRecurring("Insert", primary);

        vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasClustersInsert);

        uint32_t specialInstanceID = 0;
        vkCmdPushConstants(primary, m_dsetContainer.getPipeLayout(), m_stageFlags, 0, sizeof(uint32_t), &specialInstanceID);
        vkCmdDispatchIndirect(primary, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, dispatchBlasTempInsert));

        if(m_config.transientClusters1X || m_config.transientClusters2X)
        {
          uint32_t specialInstanceID = 1;
          vkCmdPushConstants(primary, m_dsetContainer.getPipeLayout(), m_stageFlags, 0, sizeof(uint32_t), &specialInstanceID);
          vkCmdDispatchIndirect(primary, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, dispatchBlasTransInsert));
        }

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }

      {
        auto timerSection = profiler.timeRecurring("Blas Build", primary);

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
        cmdInfo.dstAddressesArray.size   = m_tlasInstancesBuffer.info.range;
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
        vkCmdBuildClusterAccelerationStructureIndirectNV(primary, &cmdInfo);

        memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }

      {
        auto timerSection = profiler.timeRecurring("Tlas Build", primary);

        updateRayTracingTlas(primary, res, scene, !m_buildTlas);
        m_buildTlas = false;

        memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }
    }
  }

  // Ray trace
  if(true)
  {
    auto timerSection = profiler.timeRecurring("Render", primary);

    vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.plines[0]);
    vkCmdBindDescriptorSets(primary, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.layout, 0, 1,
                            m_dsetContainer.getSets(), 0, nullptr);

    const std::array<VkStridedDeviceAddressRegionKHR, 4>& bindingTables = m_rtSbt.getRegions();
    vkCmdTraceRaysKHR(primary, &bindingTables[0], &bindingTables[1], &bindingTables[2], &bindingTables[3],
                      frame.frameConstants.viewport.x, frame.frameConstants.viewport.y, 1);


    res.cmdBeginRendering(primary, false, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    res.cmdDynamicState(primary);
    writeRayTracingDepthBuffer(primary);
    vkCmdEndRendering(primary);
  }

  {
    if(!frame.freezeCulling)
    {
      res.cmdBuildHiz(primary, frame, profiler);
    }
  }

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
  deinitBasics(res);
  m_cluster.deinit(res);
  deinitRayTracingTlas(res);

  res.destroy(m_sceneBuildBuffer);
  res.destroy(m_sceneDataBuffer);
  res.destroy(m_sceneSplitBuffer);
  res.destroy(m_sceneBlasDataBuffer);
  res.destroy(m_sceneClasDataBuffer);

  m_tessTable.deinit(res);

  m_rtSbt.destroy();               // Shading binding table wrapper
  m_rtPipe.destroy(res.m_device);  // Hold pipelines and layout

  vkDestroyPipeline(res.m_device, m_pipelines.computeClusterClassify, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeClustersCull, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeBlasClustersInsert, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeTriangleInstantiate, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeTriangleSplit, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeBlasSetup, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeBuildSetup, nullptr);

  res.destroyShaders(m_shaders);

  m_dsetContainer.deinit();
  m_resourceReservedUsage = {};
}

bool RendererRayTraceClustersTess::initRayTracingScene(Resources& res, Scene& scene, const RendererConfig& config)
{
  if(!m_cluster.init(res, scene, config, &m_tessTable))
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
  nvvkhl::PipelineContainer& p = m_rtPipe;
  p.plines.resize(1);

  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMissAO,
    eClosestHit,
    eShaderGroupCount
  };
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  for(auto& s : stages)
    s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

  stages[eRaygen].module     = res.m_shaderManager.getShaderModule(m_shaders.rayGen).module;
  stages[eRaygen].pName      = "main";
  stages[eRaygen].stage      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eMiss].module       = res.m_shaderManager.getShaderModule(m_shaders.miss).module;
  stages[eMiss].pName        = "main";
  stages[eMiss].stage        = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMissAO].module     = res.m_shaderManager.getShaderModule(m_shaders.missAO).module;
  stages[eMissAO].pName      = "main";
  stages[eMissAO].stage      = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eClosestHit].module = res.m_shaderManager.getShaderModule(m_shaders.closestHit).module;
  stages[eClosestHit].pName  = "main";
  stages[eClosestHit].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  shaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  shaderGroups.push_back(group);

  // Miss AO
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMissAO;
  shaderGroups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  shaderGroups.push_back(group);

  // Push constant: we want to be able to update constants used by the shaders
  //const VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant)};

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> dsetLayouts = {m_dsetContainer.getLayout()};  // , m_pContainer[eGraphic].dstLayout};
  VkPipelineLayoutCreateInfo layoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutCreateInfo.setLayoutCount         = static_cast<uint32_t>(dsetLayouts.size());
  layoutCreateInfo.pSetLayouts            = dsetLayouts.data();
  layoutCreateInfo.pushConstantRangeCount = 0;  //1;
  //pipeline_layout_create_info.pPushConstantRanges    = &push_constant,

  vkCreatePipelineLayout(res.m_device, &layoutCreateInfo, nullptr, &p.layout);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV pipeClusters = {
      VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV};

  pipelineInfo.stageCount                   = static_cast<uint32_t>(stages.size());
  pipelineInfo.pStages                      = stages.data();
  pipelineInfo.groupCount                   = static_cast<uint32_t>(shaderGroups.size());
  pipelineInfo.pGroups                      = shaderGroups.data();
  pipelineInfo.maxPipelineRayRecursionDepth = 2;
  pipelineInfo.layout                       = p.layout;
  pipelineInfo.flags                        = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;

  // new for clusters
  {
    pipelineInfo.pNext                              = &pipeClusters;
    pipeClusters.allowClusterAccelerationStructures = true;
  }

  VkResult result = vkCreateRayTracingPipelinesKHR(res.m_device, {}, {}, 1, &pipelineInfo, nullptr, &p.plines[0]);

  // Creating the SBT
  m_rtSbt.setup(res.m_device, res.m_queueFamily, &res.m_allocator, m_rtProperties);
  m_rtSbt.create(p.plines[0], pipelineInfo);
}


std::unique_ptr<Renderer> makeRendererRayTraceClustersTess()
{
  return std::make_unique<RendererRayTraceClustersTess>();
}

void RendererRayTraceClustersTess::updatedFrameBuffer(Resources& res)
{
  vkDeviceWaitIdle(res.m_device);
  std::array<VkWriteDescriptorSet, 3> writeSets;
  VkDescriptorImageInfo               renderTargetInfo;
  renderTargetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  renderTargetInfo.imageView   = res.m_framebuffer.viewColor;
  writeSets[0]                 = m_dsetContainer.makeWrite(0, BINDINGS_RENDER_TARGET, &renderTargetInfo);

  VkDescriptorImageInfo raytracingDepthInfo;
  raytracingDepthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  raytracingDepthInfo.imageView   = res.m_framebuffer.viewRaytracingDepth;

  writeSets[1] = m_dsetContainer.makeWrite(0, BINDINGS_RAYTRACING_DEPTH, &raytracingDepthInfo);
  writeSets[2] = m_dsetContainer.makeWrite(0, BINDINGS_HIZ_TEX, &res.m_hizUpdate.farImageInfo);

  vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);

  Renderer::updatedFrameBuffer(res);
}

}  // namespace tessellatedclusters
