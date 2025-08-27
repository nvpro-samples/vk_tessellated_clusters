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

#include <nvutils/alignment.hpp>
#include <fmt/format.h>

#include "renderer.hpp"
#include "tessellation_table.hpp"
#include "../shaders/shaderio.h"

namespace tessellatedclusters {

class RendererRasterClustersTess : public Renderer
{
public:
  virtual bool init(Resources& res, Scene& scene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) override;
  virtual void updatedFrameBuffer(Resources& res) override;
  virtual void deinit(Resources& res) override;

private:
  bool initShaders(Resources& res, Scene& scene, const RendererConfig& config);

  struct Shaders
  {
    shaderc::SpvCompilationResult meshShaderFull;
    shaderc::SpvCompilationResult meshShaderTess;
    shaderc::SpvCompilationResult meshShaderTessBatched;
    shaderc::SpvCompilationResult taskShaderTessBatched;
    shaderc::SpvCompilationResult fragmentShader;

    shaderc::SpvCompilationResult computeInstancesClassify;

    shaderc::SpvCompilationResult computeClustersCull;
    shaderc::SpvCompilationResult computeClusterClassify;
    shaderc::SpvCompilationResult computeTriangleSplit;

    shaderc::SpvCompilationResult computeBuildSetup;
  };

  struct Pipelines
  {
    VkPipeline graphicsMeshFull         = {};
    VkPipeline graphicsMeshTess         = {};
    VkPipeline computeTriangleSplit     = {};
    VkPipeline computeClustersCull      = {};
    VkPipeline computeClusterClassify   = {};
    VkPipeline computeBuildSetup        = {};
    VkPipeline computeInstancesClassify = {};
  };

  RendererConfig m_config;

  Shaders   m_shaders;
  Pipelines m_pipelines;

  VkShaderStageFlags   m_stageFlags{};
  VkPipelineLayout     m_pipelineLayout{};
  nvvk::DescriptorPack m_dsetPack;

  nvvk::Buffer            m_sceneBuildBuffer;
  nvvk::Buffer            m_sceneDataBuffer;
  nvvk::Buffer            m_sceneSplitBuffer;
  shaderio::SceneBuilding m_sceneBuildShaderio;

  TessellationTable m_tessTable;
};

bool RendererRasterClustersTess::initShaders(Resources& res, Scene& scene, const RendererConfig& config)
{
  shaderc::CompileOptions options = res.makeCompilerOptions();

  uint32_t meshletTriangles = shaderio::adjustClusterProperty(scene.m_maxClusterTriangles);
  uint32_t meshletVertices  = shaderio::adjustClusterProperty(scene.m_maxClusterVertices);
  LOGI("mesh shader config: %d triangles %d vertices\n", meshletTriangles, meshletVertices);

  options.AddMacroDefinition("CLUSTER_VERTEX_COUNT", fmt::format("{}", meshletVertices));
  options.AddMacroDefinition("CLUSTER_TRIANGLE_COUNT", fmt::format("{}", meshletTriangles));
  options.AddMacroDefinition("TESSTABLE_SIZE", fmt::format("{}", m_tessTable.m_maxSize));
  options.AddMacroDefinition("TESSTABLE_LOOKUP_SIZE", fmt::format("{}", m_tessTable.m_maxSizeConfigs));
  options.AddMacroDefinition("TARGETS_RASTERIZATION", "1");
  options.AddMacroDefinition("TESS_RASTER_USE_BATCH", fmt::format("{}", config.rasterBatchMeshlets ? 1 : 0));
  options.AddMacroDefinition("TESS_USE_PN", fmt::format("{}", config.pnDisplacement ? 1 : 0));
  options.AddMacroDefinition("TESS_USE_1X_TRANSIENTBUILDS", "0");
  options.AddMacroDefinition("TESS_USE_2X_TRANSIENTBUILDS", "0");
  options.AddMacroDefinition("TESS_ACTIVE", "1");
  options.AddMacroDefinition("MAX_PART_TRIANGLES", fmt::format("{}", 1 << config.numPartTriangleBits));
  options.AddMacroDefinition("MAX_VISIBLE_CLUSTERS", fmt::format("{}", 1 << config.numVisibleClusterBits));
  options.AddMacroDefinition("MAX_SPLIT_TRIANGLES", fmt::format("{}", 1 << config.numSplitTriangleBits));
  options.AddMacroDefinition("MESHSHADER_WORKGROUP_SIZE", "32");
  options.AddMacroDefinition("HAS_DISPLACEMENT_TEXTURES", fmt::format("{}", scene.m_textureImages.size() ? 1 : 0));
  options.AddMacroDefinition("DO_CULLING", fmt::format("{}", config.doCulling ? 1 : 0));
  options.AddMacroDefinition("DO_ANIMATION", fmt::format("{}", config.doAnimation ? 1 : 0));
  options.AddMacroDefinition("DEBUG_VISUALIZATION", fmt::format("{}", config.debugVisualization ? 1 : 0));

  res.compileShader(m_shaders.meshShaderFull, VK_SHADER_STAGE_MESH_BIT_NV, "render_raster_clusters.mesh.glsl", &options);

  res.compileShader(m_shaders.meshShaderTess, VK_SHADER_STAGE_MESH_BIT_NV, "render_raster_clusters_tess.mesh.glsl", &options);

  res.compileShader(m_shaders.meshShaderTessBatched, VK_SHADER_STAGE_MESH_BIT_NV,
                    "render_raster_clusters_batched.mesh.glsl", &options);

  res.compileShader(m_shaders.taskShaderTessBatched, VK_SHADER_STAGE_TASK_BIT_NV,
                    "render_raster_clusters_batched.task.glsl", &options);

  res.compileShader(m_shaders.fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT, "render_raster.frag.glsl", &options);

  res.compileShader(m_shaders.computeInstancesClassify, VK_SHADER_STAGE_COMPUTE_BIT, "instances_classify.comp.glsl", &options);

  res.compileShader(m_shaders.computeClusterClassify, VK_SHADER_STAGE_COMPUTE_BIT, "cluster_classify.comp.glsl", &options);

  res.compileShader(m_shaders.computeClustersCull, VK_SHADER_STAGE_COMPUTE_BIT, "clusters_cull.comp.glsl", &options);

  res.compileShader(m_shaders.computeTriangleSplit, VK_SHADER_STAGE_COMPUTE_BIT, "triangle_split.comp.glsl", &options);

  res.compileShader(m_shaders.computeBuildSetup, VK_SHADER_STAGE_COMPUTE_BIT, "build_setup.comp.glsl", &options);

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
    res.m_allocator.createBuffer(m_sceneBuildBuffer, sizeof(shaderio::SceneBuilding),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                     | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    size_t offsetVisibles = 0;
    size_t offsetFull =
        nvutils::align_up(offsetVisibles + sizeof(shaderio::ClusterInfo) * uint32_t(1u << config.numVisibleClusterBits), 128);
    size_t offsetTess =
        nvutils::align_up(offsetFull + sizeof(shaderio::ClusterInfo) * uint32_t(1u << config.numVisibleClusterBits), 128);
    size_t offsetInstances =
        nvutils::align_up(offsetTess + sizeof(shaderio::TessTriangleInfo) * uint32_t(1u << config.numPartTriangleBits), 128);
    size_t size = offsetInstances + sizeof(uint32_t) * m_renderInstances.size();

    res.m_allocator.createBuffer(m_sceneDataBuffer, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += m_sceneDataBuffer.bufferSize;


    memset(&m_sceneBuildShaderio, 0, sizeof(m_sceneBuildShaderio));
    m_sceneBuildShaderio.numRenderInstances = uint32_t(m_renderInstances.size());
    m_sceneBuildShaderio.visibleClusters    = m_sceneDataBuffer.address + offsetVisibles;
    m_sceneBuildShaderio.fullClusters       = m_sceneDataBuffer.address + offsetFull;
    m_sceneBuildShaderio.partTriangles      = m_sceneDataBuffer.address + offsetTess;
    m_sceneBuildShaderio.instanceStates     = m_sceneDataBuffer.address + offsetInstances;

    res.m_allocator.createBuffer(m_sceneSplitBuffer, sizeof(shaderio::TessTriangleInfo) * uint32_t(1 << config.numSplitTriangleBits),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += m_sceneSplitBuffer.bufferSize;
    m_sceneBuildShaderio.splitTriangles = m_sceneSplitBuffer.address;
  }

  {
    m_stageFlags = VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    if(config.rasterBatchMeshlets)
    {
      m_stageFlags |= VK_SHADER_STAGE_TASK_BIT_NV;
    }

    nvvk::DescriptorBindings bindings;
    bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_TESSTABLE_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_SCENEBUILDING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_HIZ_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, m_stageFlags);

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

  {
    nvvk::GraphicsPipelineCreator graphicsGen;
    nvvk::GraphicsPipelineState   graphicsState = res.m_basicGraphicsState;

    graphicsGen.pipelineInfo.layout                  = m_pipelineLayout;
    graphicsGen.renderingState.depthAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.depthAttachmentFormat;
    graphicsGen.renderingState.stencilAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.stencilAttachmentFormat;
    graphicsGen.colorFormats = {res.m_frameBuffer.colorFormat};

    graphicsState.rasterizationState.frontFace = config.flipWinding ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;

    graphicsGen.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.meshShaderFull));
    graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.fragmentShader));

    graphicsGen.createGraphicsPipeline(res.m_device, nullptr, graphicsState, &m_pipelines.graphicsMeshFull);

    graphicsGen.clearShaders();
    if(config.rasterBatchMeshlets)
    {
      graphicsGen.addShader(VK_SHADER_STAGE_TASK_BIT_NV, "main",
                            nvvkglsl::GlslCompiler::getSpirvData(m_shaders.taskShaderTessBatched));
      graphicsGen.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "main",
                            nvvkglsl::GlslCompiler::getSpirvData(m_shaders.meshShaderTessBatched));
      graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.fragmentShader));
    }
    else
    {
      graphicsGen.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.meshShaderTess));
      graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.fragmentShader));
    }
    graphicsGen.createGraphicsPipeline(res.m_device, nullptr, graphicsState, &m_pipelines.graphicsMeshTess);
  }

  {
    VkComputePipelineCreateInfo compInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkShaderModuleCreateInfo    shaderInfo{};
    compInfo.stage       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pNext = &shaderInfo;
    compInfo.stage.pName = "main";
    compInfo.layout      = m_pipelineLayout;

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeClustersCull);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeClustersCull);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTriangleSplit);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTriangleSplit);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeClusterClassify);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeClusterClassify);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBuildSetup);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBuildSetup);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeInstancesClassify);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeInstancesClassify);
  }

  LOGI("persistent warps %d\n", m_config.persistentThreads / SUBGROUP_SIZE);

  m_resourceActualUsage = m_resourceReservedUsage;

  return true;
}

void RendererRasterClustersTess::render(VkCommandBuffer cmd, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  m_sceneBuildShaderio.viewPos = frame.freezeCulling ? frame.frameConstantsLast.viewPos : frame.frameConstants.viewPos;

  vkCmdUpdateBuffer(cmd, res.m_commonBuffers.frameConstants.buffer, 0, sizeof(shaderio::FrameConstants) * 2,
                    (const uint32_t*)&frame.frameConstants);
  vkCmdUpdateBuffer(cmd, m_sceneBuildBuffer.buffer, 0, sizeof(shaderio::SceneBuilding), (const uint32_t*)&m_sceneBuildShaderio);
  vkCmdFillBuffer(cmd, res.m_commonBuffers.readBack.buffer, 0, sizeof(shaderio::Readback), 0);
  vkCmdFillBuffer(cmd, m_sceneSplitBuffer.buffer, 0, m_sceneSplitBuffer.bufferSize, ~0);

  const bool useSky = true;  // When using Sky, the sky is rendered first and the rest of the scene is rendered on top of it.

  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);

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
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Cluster Classify");

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeClusterClassify);

      vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, dispatchClassify));

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);

      uint32_t buildSetupID = BUILD_SETUP_SPLIT;
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(cmd, 1, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Split");

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTriangleSplit);

      vkCmdDispatch(cmd, (m_config.persistentThreads + TRIANGLE_SPLIT_WORKGROUP - 1) / TRIANGLE_SPLIT_WORKGROUP, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);

      uint32_t buildSetupID = BUILD_SETUP_DRAW_TESS;
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(cmd, 1, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV | VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV
                               | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                           0, 1, &memBarrier, 0, nullptr, 0, nullptr);
    }

    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Draw");

      VkAttachmentLoadOp op = useSky ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_CLEAR;

      res.cmdBeginRendering(cmd, false, op, op);

      if(useSky)
      {
        writeBackgroundSky(cmd);
      }

      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines.graphicsMeshTess);

      vkCmdDrawMeshTasksIndirectNV(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, drawPartTriangles), 1, 0);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines.graphicsMeshFull);

      vkCmdDrawMeshTasksIndirectNV(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, drawFullClusters), 1, 0);

      vkCmdEndRendering(cmd);
    }
  }

  {
    // hiz
    if(!frame.freezeCulling)
    {
      res.cmdBuildHiz(cmd, frame, profiler);
    }
  }
}

void RendererRasterClustersTess::updatedFrameBuffer(Resources& res)
{
  nvvk::WriteSetContainer writeSets;
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX), res.m_hizUpdate.farImageInfo);
  vkUpdateDescriptorSets(res.m_device, writeSets.size(), writeSets.data(), 0, nullptr);

  Renderer::updatedFrameBuffer(res);
}

void RendererRasterClustersTess::deinit(Resources& res)
{
  m_tessTable.deinit(res);

  res.m_allocator.destroyBuffer(m_sceneDataBuffer);
  res.m_allocator.destroyBuffer(m_sceneBuildBuffer);
  res.m_allocator.destroyBuffer(m_sceneSplitBuffer);

  res.destroyPipelines(m_pipelines);
  vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);
  m_dsetPack.deinit();

  deinitBasics(res);
}

std::unique_ptr<Renderer> makeRendererRasterClustersTess()
{
  return std::make_unique<RendererRasterClustersTess>();
}
}  // namespace tessellatedclusters
