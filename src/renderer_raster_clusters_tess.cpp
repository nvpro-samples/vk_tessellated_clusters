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

#include <nvh/misc.hpp>
#include <nvh/alignment.hpp>

#include "renderer.hpp"
#include "tessellation_table.hpp"
#include "shaders/shaderio.h"

namespace tessellatedclusters {

class RendererRasterClustersTess : public Renderer
{
public:
  virtual bool init(Resources& res, Scene& scene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerVK& profiler) override;
  virtual void updatedFrameBuffer(Resources& res) override;
  virtual void deinit(Resources& res) override;

private:
  bool initShaders(Resources& res, Scene& scene, const RendererConfig& config);

  struct Shaders
  {
    nvvk::ShaderModuleID meshShaderFull;
    nvvk::ShaderModuleID meshShaderTess;
    nvvk::ShaderModuleID meshShaderTessBatched;
    nvvk::ShaderModuleID taskShaderTessBatched;
    nvvk::ShaderModuleID fragmentShader;

    nvvk::ShaderModuleID computeInstancesClassify;

    nvvk::ShaderModuleID computeClustersCull;
    nvvk::ShaderModuleID computeClusterClassify;
    nvvk::ShaderModuleID computeTriangleSplit;

    nvvk::ShaderModuleID computeBuildSetup;
  };

  struct Pipelines
  {
    VkPipeline graphicsMeshFull         = nullptr;
    VkPipeline graphicsMeshTess         = nullptr;
    VkPipeline computeTriangleSplit     = nullptr;
    VkPipeline computeClustersCull      = nullptr;
    VkPipeline computeClusterClassify   = nullptr;
    VkPipeline computeBuildSetup        = nullptr;
    VkPipeline computeInstancesClassify = nullptr;
  };

  RendererConfig m_config;

  Shaders                      m_shaders;
  VkShaderStageFlags           m_stageFlags;
  Pipelines                    m_pipelines;
  nvvk::DescriptorSetContainer m_dsetContainer;

  RBuffer                 m_sceneBuildBuffer;
  RBuffer                 m_sceneDataBuffer;
  RBuffer                 m_sceneSplitBuffer;
  shaderio::SceneBuilding m_sceneBuildShaderio;

  TessellationTable m_tessTable;
};

bool RendererRasterClustersTess::initShaders(Resources& res, Scene& scene, const RendererConfig& config)
{
  std::string prepend;
  prepend += nvh::stringFormat("#define CLUSTER_VERTEX_COUNT %d\n", shaderio::adjustClusterProperty(scene.m_config.clusterVertices));
  prepend += nvh::stringFormat("#define CLUSTER_TRIANGLE_COUNT %d\n",
                               shaderio::adjustClusterProperty(scene.m_config.clusterTriangles));
  prepend += nvh::stringFormat("#define TESSTABLE_SIZE %d\n", m_tessTable.m_maxSize);
  prepend += nvh::stringFormat("#define TESSTABLE_LOOKUP_SIZE %d\n", m_tessTable.m_maxSizeConfigs);
  prepend += nvh::stringFormat("#define TARGETS_RASTERIZATION %d\n", 1);
  prepend += nvh::stringFormat("#define TESS_RASTER_USE_BATCH %d\n", config.rasterBatchMeshlets ? 1 : 0);
  prepend += nvh::stringFormat("#define TESS_USE_PN %d\n", config.pnDisplacement ? 1 : 0);
  prepend += nvh::stringFormat("#define TESS_USE_1X_TRANSIENTBUILDS %d\n", 0);
  prepend += nvh::stringFormat("#define TESS_USE_2X_TRANSIENTBUILDS %d\n", 0);
  prepend += nvh::stringFormat("#define TESS_ACTIVE %d\n", 1);
  prepend += nvh::stringFormat("#define MAX_PART_TRIANGLES %d\n", 1 << config.numPartTriangleBits);
  prepend += nvh::stringFormat("#define MAX_VISIBLE_CLUSTERS %d\n", 1 << config.numVisibleClusterBits);
  prepend += nvh::stringFormat("#define MAX_SPLIT_TRIANGLES %d\n", 1 << config.numSplitTriangleBits);
  prepend += nvh::stringFormat("#define MESHSHADER_WORKGROUP_SIZE %d\n", 32);
  prepend += nvh::stringFormat("#define HAS_DISPLACEMENT_TEXTURES %d\n", scene.m_textureImages.size() ? 1 : 0);

  m_shaders.meshShaderFull =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_NV, "render_raster_clusters.mesh.glsl", prepend);
  m_shaders.meshShaderTess =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_NV, "render_raster_clusters_tess.mesh.glsl", prepend);
  m_shaders.meshShaderTessBatched =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_NV, "render_raster_clusters_batched.mesh.glsl", prepend);
  m_shaders.taskShaderTessBatched =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_TASK_BIT_NV, "render_raster_clusters_batched.task.glsl", prepend);
  m_shaders.fragmentShader =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "render_raster.frag.glsl", prepend);
  m_shaders.computeInstancesClassify =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "instances_classify.comp.glsl", prepend);
  m_shaders.computeClusterClassify =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "cluster_classify.comp.glsl", prepend);
  m_shaders.computeClustersCull =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "clusters_cull.comp.glsl", prepend);
  m_shaders.computeTriangleSplit =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "triangle_split.comp.glsl", prepend);
  m_shaders.computeBuildSetup =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "build_setup.comp.glsl", prepend);

  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  return initBasicShaders(res, scene, config);
}

bool RendererRasterClustersTess::init(Resources& res, Scene& scene, const RendererConfig& config)
{
  m_config = config;

  m_tessTable.init(res);

  if(!initShaders(res, scene, config))
  {
    m_tessTable.deinit(res);
    return false;
  }

  initBasics(res, scene, config);


  {
    m_sceneBuildBuffer = res.createBuffer(sizeof(shaderio::SceneBuilding), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                                                               | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    size_t offsetVisibles = 0;
    size_t offsetFull =
        nvh::align_up(offsetVisibles + sizeof(shaderio::ClusterInfo) * uint32_t(1u << config.numVisibleClusterBits), 128);
    size_t offsetTess =
        nvh::align_up(offsetFull + sizeof(shaderio::ClusterInfo) * uint32_t(1u << config.numVisibleClusterBits), 128);
    size_t offsetInstances =
        nvh::align_up(offsetTess + sizeof(shaderio::TessTriangleInfo) * uint32_t(1u << config.numPartTriangleBits), 128);
    size_t size = offsetInstances + sizeof(uint32_t) * m_renderInstances.size();

    m_sceneDataBuffer = res.createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += m_sceneDataBuffer.info.range;


    memset(&m_sceneBuildShaderio, 0, sizeof(m_sceneBuildShaderio));
    m_sceneBuildShaderio.numRenderInstances = uint32_t(m_renderInstances.size());
    m_sceneBuildShaderio.visibleClusters    = m_sceneDataBuffer.address + offsetVisibles;
    m_sceneBuildShaderio.fullClusters       = m_sceneDataBuffer.address + offsetFull;
    m_sceneBuildShaderio.partTriangles      = m_sceneDataBuffer.address + offsetTess;
    m_sceneBuildShaderio.instanceStates     = m_sceneDataBuffer.address + offsetInstances;

    m_sceneSplitBuffer = res.createBuffer(sizeof(shaderio::TessTriangleInfo) * uint32_t(1 << config.numSplitTriangleBits),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += m_sceneSplitBuffer.info.range;
    m_sceneBuildShaderio.splitTriangles = m_sceneSplitBuffer.address;
  }

  {
    m_dsetContainer.init(res.m_device);

    m_stageFlags = VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    if(config.rasterBatchMeshlets)
    {
      m_stageFlags |= VK_SHADER_STAGE_TASK_BIT_NV;
    }

    m_dsetContainer.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_TESSTABLE_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_SCENEBUILDING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_HIZ_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, m_stageFlags);

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

  {
    nvvk::GraphicsPipelineState     state = res.m_basicGraphicsState;
    nvvk::GraphicsPipelineGenerator gfxGen(res.m_device, m_dsetContainer.getPipeLayout(),
                                           res.m_framebuffer.pipelineRenderingInfo, state);
    state.rasterizationState.frontFace = config.flipWinding ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    gfxGen.addShader(res.m_shaderManager.get(m_shaders.meshShaderFull), VK_SHADER_STAGE_MESH_BIT_NV);
    gfxGen.addShader(res.m_shaderManager.get(m_shaders.fragmentShader), VK_SHADER_STAGE_FRAGMENT_BIT);
    m_pipelines.graphicsMeshFull = gfxGen.createPipeline();

    gfxGen.clearShaders();
    if(config.rasterBatchMeshlets)
    {
      gfxGen.addShader(res.m_shaderManager.get(m_shaders.taskShaderTessBatched), VK_SHADER_STAGE_TASK_BIT_NV);
      gfxGen.addShader(res.m_shaderManager.get(m_shaders.meshShaderTessBatched), VK_SHADER_STAGE_MESH_BIT_NV);
      gfxGen.addShader(res.m_shaderManager.get(m_shaders.fragmentShader), VK_SHADER_STAGE_FRAGMENT_BIT);
    }
    else
    {
      gfxGen.addShader(res.m_shaderManager.get(m_shaders.meshShaderTess), VK_SHADER_STAGE_MESH_BIT_NV);
      gfxGen.addShader(res.m_shaderManager.get(m_shaders.fragmentShader), VK_SHADER_STAGE_FRAGMENT_BIT);
    }
    m_pipelines.graphicsMeshTess = gfxGen.createPipeline();
  }

  {
    VkComputePipelineCreateInfo compInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compInfo.stage                       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                 = "main";
    compInfo.layout                      = m_dsetContainer.getPipeLayout();
    compInfo.flags                       = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeClustersCull);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeClustersCull);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeTriangleSplit);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTriangleSplit);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeClusterClassify);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeClusterClassify);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeBuildSetup);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBuildSetup);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeInstancesClassify);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeInstancesClassify);
  }

  LOGI("persistent warps %d\n", m_config.persistentThreads / SUBGROUP_SIZE);

  m_resourceActualUsage = m_resourceReservedUsage;

  return true;
}

void RendererRasterClustersTess::render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerVK& profiler)
{
  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  m_sceneBuildShaderio.viewPos = frame.freezeCulling ? frame.frameConstantsLast.viewPos : frame.frameConstants.viewPos;

  vkCmdUpdateBuffer(primary, res.m_common.view.buffer, 0, sizeof(shaderio::FrameConstants) * 2, (const uint32_t*)&frame.frameConstants);
  vkCmdUpdateBuffer(primary, m_sceneBuildBuffer.buffer, 0, sizeof(shaderio::SceneBuilding), (const uint32_t*)&m_sceneBuildShaderio);
  vkCmdFillBuffer(primary, res.m_common.readbackDevice.buffer, 0, sizeof(shaderio::Readback), 0);
  vkCmdFillBuffer(primary, m_sceneSplitBuffer.buffer, 0, m_sceneSplitBuffer.info.range, ~0);

  const bool useSky = true;  // When using Sky, the sky is rendered first and the rest of the scene is rendered on top of it.

  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);

  {
    if(useSky)
    {
      res.m_sky.skyParams() = frame.frameConstants.skyParams;
      res.m_sky.updateParameterBuffer(primary);
      res.cmdImageTransition(primary, res.m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
      res.m_sky.draw(primary, frame.frameConstants.viewMatrix, frame.frameConstants.projMatrix,
                     res.m_framebuffer.scissor.extent);
    }

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
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    {
      auto timerSection = profiler.timeRecurring("Cluster Classify", primary);

      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeClusterClassify);

      vkCmdDispatchIndirect(primary, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, dispatchClassify));

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
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

      uint32_t buildSetupID = BUILD_SETUP_DRAW_TESS;
      vkCmdPushConstants(primary, m_dsetContainer.getPipeLayout(), m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(primary, 1, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV | VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV
                               | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                           0, 1, &memBarrier, 0, nullptr, 0, nullptr);
    }

    {
      auto timerSection = profiler.timeRecurring("Draw", primary);

      res.cmdBeginRendering(primary, false, useSky ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR);

      res.cmdDynamicState(primary);
      vkCmdBindDescriptorSets(primary, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dsetContainer.getPipeLayout(), 0, 1,
                              m_dsetContainer.getSets(), 0, nullptr);

      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines.graphicsMeshTess);

      vkCmdDrawMeshTasksIndirectNV(primary, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, drawPartTriangles), 1, 0);

      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines.graphicsMeshFull);

      vkCmdDrawMeshTasksIndirectNV(primary, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, drawFullClusters), 1, 0);

      vkCmdEndRendering(primary);
    }
  }

  {
    // hiz
    if(!frame.freezeCulling)
    {
      res.cmdBuildHiz(primary, frame, profiler);
    }
  }
}

void RendererRasterClustersTess::updatedFrameBuffer(Resources& res)
{
  vkDeviceWaitIdle(res.m_device);
  std::array<VkWriteDescriptorSet, 1> writeSets;
  writeSets[0] = m_dsetContainer.makeWrite(0, BINDINGS_HIZ_TEX, &res.m_hizUpdate.farImageInfo);
  vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);

  Renderer::updatedFrameBuffer(res);
}

void RendererRasterClustersTess::deinit(Resources& res)
{
  vkDestroyPipeline(res.m_device, m_pipelines.graphicsMeshFull, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.graphicsMeshTess, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeClusterClassify, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeTriangleSplit, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeClustersCull, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeBuildSetup, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeInstancesClassify, nullptr);

  res.destroy(m_sceneDataBuffer);
  res.destroy(m_sceneBuildBuffer);
  res.destroy(m_sceneSplitBuffer);

  m_dsetContainer.deinit();

  res.destroyShaders(m_shaders);

  m_tessTable.deinit(res);

  deinitBasics(res);
}

std::unique_ptr<Renderer> makeRendererRasterClustersTess()
{
  return std::make_unique<RendererRasterClustersTess>();
}
}  // namespace tessellatedclusters
