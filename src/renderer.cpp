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

#include <random>
#include <vector>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>
#include "glm/gtc/matrix_transform.hpp"

#include "renderer.hpp"
#include "../shaders/shaderio.h"

namespace tessellatedclusters {

bool Renderer::initBasicShaders(Resources& res, Scene& scene, const RendererConfig& config)
{
  res.compileShader(m_basicShaders.fullScreenVertexShader, VK_SHADER_STAGE_VERTEX_BIT, "fullscreen.vert.glsl");
  res.compileShader(m_basicShaders.fullscreenWriteDepthFragShader, VK_SHADER_STAGE_FRAGMENT_BIT, "fullscreen_write_depth.frag.glsl");
  res.compileShader(m_basicShaders.fullscreenBackgroundFragShader, VK_SHADER_STAGE_FRAGMENT_BIT, "fullscreen_background.frag.glsl");

  if(!res.verifyShaders(m_basicShaders))
  {
    return false;
  }

  return true;
}

void Renderer::initBasics(Resources& res, Scene& scene, const RendererConfig& config)
{
  initBasicPipelines(res, scene, config);
  m_renderInstances.resize(scene.m_instances.size() * config.numSceneCopies);

  std::default_random_engine            rng(2342);
  std::uniform_real_distribution<float> randomUnorm(0.0f, 1.0f);

  uint32_t axis    = config.gridConfig;
  size_t   sq      = 1;
  int      numAxis = 0;
  if(!axis)
    axis = 3;

  for(int i = 0; i < 3; i++)
  {
    numAxis += (axis & (1 << i)) ? 1 : 0;
  }

  switch(numAxis)
  {
    case 1:
      sq = config.numSceneCopies;
      break;
    case 2:
      while(sq * sq < config.numSceneCopies)
      {
        sq++;
      }
      break;
    case 3:
      while(sq * sq * sq < config.numSceneCopies)
      {
        sq++;
      }
      break;
  }

  size_t lastCopyIndex = 0;

  glm::vec3 gridShift;
  glm::mat4 gridRotMatrix;


  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    size_t originalIndex = i % scene.m_instances.size();
    size_t copyIndex     = i / scene.m_instances.size();

    // todo modify matrix etc. for grid layout

    shaderio::RenderInstance& renderInstance = m_renderInstances[i];
    renderInstance                           = {};

    const uint32_t         geometryID = scene.m_instances[originalIndex].geometryID;
    const Scene::Geometry& geometry   = scene.m_geometries[geometryID];

    glm::mat4 worldMatrix = scene.m_instances[originalIndex].matrix;

    if(copyIndex)
    {
      if(copyIndex != lastCopyIndex)
      {
        lastCopyIndex = copyIndex;

        gridShift = config.refShift * (scene.m_bbox.hi - scene.m_bbox.lo);
        size_t c  = copyIndex;

        float u = 0;
        float v = 0;
        float w = 0;

        switch(numAxis)
        {
          case 1:
            u = float(c);
            break;
          case 2:
            u = float(c % sq);
            v = float(c / sq);
            break;
          case 3:
            u = float(c % sq);
            v = float((c / sq) % sq);
            w = float(c / (sq * sq));
            break;
        }

        float use = u;

        if(axis & (1 << 0))
        {
          gridShift.x *= -use;
          if(numAxis > 1)
            use = v;
        }
        else
        {
          gridShift.x = 0;
        }

        if(axis & (1 << 1))
        {
          gridShift.y *= use;
          if(numAxis > 2)
            use = w;
          else if(numAxis > 1)
            use = v;
        }
        else
        {
          gridShift.y = 0;
        }

        if(axis & (1 << 2))
        {
          gridShift.z *= -use;
        }
        else
        {
          gridShift.z = 0;
        }

        if(axis & (8 | 16 | 32))
        {
          glm::vec3 mask    = {axis & 8 ? 1.0f : 0.0f, axis & 16 ? 1.0f : 0.0f, axis & 32 ? 1.0f : 0.0f};
          glm::vec3 gridDir = glm::vec3(randomUnorm(rng), randomUnorm(rng), randomUnorm(rng));
          gridDir           = glm::max(gridDir * mask, mask * 0.00001f);
          float gridAngle   = randomUnorm(rng) * glm::pi<float>() * 2.0f;
          gridDir           = glm::normalize(gridDir);

          gridRotMatrix = glm::rotate(glm::mat4(1), gridAngle, gridDir);
        }
      }

      glm::vec3 translation;
      translation = worldMatrix[3];
      if(axis & (8 | 16 | 32))
      {
        worldMatrix[3] = glm::vec4(0, 0, 0, 1);
        worldMatrix    = gridRotMatrix * worldMatrix;
      }
      worldMatrix[3] = glm::vec4(translation + gridShift, 1.f);
    }

    renderInstance.worldMatrix  = worldMatrix;
    renderInstance.numVertices  = geometry.numVertices;
    renderInstance.numClusters  = geometry.numClusters;
    renderInstance.numTriangles = geometry.numTriangles;
    renderInstance.geometryID   = geometryID;

    renderInstance.positions             = geometry.positionsBuffer.address;
    renderInstance.normals               = geometry.normalsBuffer.address;
    renderInstance.texcoords             = geometry.texCoordsBuffer.address;
    renderInstance.clusters              = geometry.clustersBuffer.address;
    renderInstance.clusterLocalTriangles = geometry.clusterLocalTrianglesBuffer.address;
    renderInstance.clusterBboxes         = geometry.clusterBboxesBuffer.address;

    renderInstance.displacementIndex  = geometry.displacement.textureIndex;
    renderInstance.displacementOffset = geometry.displacement.offset;
    renderInstance.displacementScale  = geometry.displacement.factor;

    renderInstance.geoLo = glm::vec4(geometry.bbox.lo, 1.0f);
    renderInstance.geoHi = glm::vec4(geometry.bbox.hi, glm::length(geometry.bbox.hi - geometry.bbox.lo));
  }


  res.m_allocator.createBuffer(m_renderInstanceBuffer, sizeof(shaderio::RenderInstance) * m_renderInstances.size(),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  res.simpleUploadBuffer(m_renderInstanceBuffer, m_renderInstances.data());

  m_resourceReservedUsage.operationsMemBytes += m_renderInstanceBuffer.bufferSize;
  m_resourceReservedUsage.geometryMemBytes = scene.m_sceneMemBytes;
}

void Renderer::deinitBasics(Resources& res)
{
  res.destroyPipelines(m_basicPipelines);

  m_basicDset.deinit();
  vkDestroyPipelineLayout(res.m_device, m_basicPipelineLayout, nullptr);

  res.m_allocator.destroyBuffer(m_renderInstanceBuffer);
}

void Renderer::updateBasicDescriptors(Resources& res)
{
  nvvk::WriteSetContainer writeSets;
  writeSets.append(m_basicDset.makeWrite(BINDINGS_FRAME_UBO), res.m_commonBuffers.frameConstants);
  writeSets.append(m_basicDset.makeWrite(BINDINGS_RAYTRACING_DEPTH), res.m_frameBuffer.imgRaytracingDepth.descriptor);
  vkUpdateDescriptorSets(res.m_device, writeSets.size(), writeSets.data(), 0, nullptr);
}

void Renderer::initRayTracingTlas(Resources& res, Scene& scene, const RendererConfig& config, const VkAccelerationStructureKHR* blas)
{
  std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(m_renderInstances.size());

  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    VkDeviceAddress blasAddress{};
    if(blas != nullptr)
    {
      VkAccelerationStructureDeviceAddressInfoKHR addressInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
      addressInfo.accelerationStructure = blas[i];
      blasAddress                       = vkGetAccelerationStructureDeviceAddressKHR(res.m_device, &addressInfo);
    }

    VkAccelerationStructureInstanceKHR instance{};
    instance.transform                              = nvvk::toTransformMatrixKHR(m_renderInstances[i].worldMatrix);
    instance.instanceCustomIndex                    = static_cast<uint32_t>(i);  // gl_InstanceCustomIndexEX
    instance.mask                                   = 0xFF;                      // All objects
    instance.instanceShaderBindingTableRecordOffset = 0,  // We will use the same hit group for all object
        instance.flags                              = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    if(config.flipWinding)
    {
      instance.flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR;
    }
    instance.accelerationStructureReference = blasAddress;
    tlasInstances[i]                        = instance;
  }


  // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
  res.m_allocator.createBuffer(m_tlasInstancesBuffer, tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR),
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  m_resourceReservedUsage.operationsMemBytes += tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR);
  res.simpleUploadBuffer(m_tlasInstancesBuffer, tlasInstances.data());

  // Wraps a device pointer to the above uploaded instances.
  VkAccelerationStructureGeometryInstancesDataKHR instancesVk{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
  instancesVk.data.deviceAddress = m_tlasInstancesBuffer.address;

  // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
  m_tlasGeometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  m_tlasGeometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  m_tlasGeometry.geometry.instances = instancesVk;

  // Find sizes
  m_tlasBuildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  m_tlasBuildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  m_tlasBuildInfo.geometryCount = 1;
  m_tlasBuildInfo.pGeometries   = &m_tlasGeometry;
  // FIXME
  m_tlasBuildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  m_tlasBuildInfo.type                     = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  m_tlasBuildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

  uint32_t                                 instanceCount = uint32_t(m_renderInstances.size());
  VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(res.m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                          &m_tlasBuildInfo, &instanceCount, &sizeInfo);


  // Create TLAS
  VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  createInfo.size = sizeInfo.accelerationStructureSize;

  res.m_allocator.createAcceleration(m_tlas, createInfo);
  m_resourceReservedUsage.rtTlasMemBytes += createInfo.size;
  // Allocate the scratch memory
  res.m_allocator.createBuffer(m_tlasScratchBuffer, sizeInfo.buildScratchSize,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
  m_resourceReservedUsage.operationsMemBytes += sizeInfo.buildScratchSize;

  // Update build information
  m_tlasBuildInfo.srcAccelerationStructure  = VK_NULL_HANDLE;
  m_tlasBuildInfo.dstAccelerationStructure  = m_tlas.accel;
  m_tlasBuildInfo.scratchData.deviceAddress = m_tlasScratchBuffer.address;
}

void Renderer::updateRayTracingTlas(VkCommandBuffer cmd, Resources& res, Scene& scene, bool update)
{
  if(update)
  {
    m_tlasBuildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    m_tlasBuildInfo.srcAccelerationStructure = m_tlas.accel;
  }
  else
  {
    m_tlasBuildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    m_tlasBuildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
  }

  // Build Offsets info: n instances
  VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo{uint32_t(m_renderInstances.size()), 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

  // Build the TLAS
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &m_tlasBuildInfo, &pBuildOffsetInfo);
}


void Renderer::deinitRayTracingTlas(Resources& res)
{
  res.m_allocator.destroyBuffer(m_tlasInstancesBuffer);
  res.m_allocator.destroyBuffer(m_tlasScratchBuffer);
  res.m_allocator.destroyAcceleration(m_tlas);
}

void Renderer::initBasicPipelines(Resources& res, Scene& scene, const RendererConfig& config)
{
  VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  nvvk::DescriptorBindings bindings;
  bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  bindings.addBinding(BINDINGS_RAYTRACING_DEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stageFlags);
  m_basicDset.init(bindings, res.m_device);
  nvvk::createPipelineLayout(res.m_device, &m_basicPipelineLayout, {m_basicDset.getLayout()});

  updateBasicDescriptors(res);

  nvvk::GraphicsPipelineState state = res.m_basicGraphicsState;

  nvvk::GraphicsPipelineCreator graphicsGen;
  graphicsGen.pipelineInfo.layout                    = m_basicPipelineLayout;
  graphicsGen.renderingState.depthAttachmentFormat   = res.m_frameBuffer.pipelineRenderingInfo.depthAttachmentFormat;
  graphicsGen.renderingState.stencilAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.stencilAttachmentFormat;
  graphicsGen.colorFormats                           = {res.m_frameBuffer.colorFormat};

  state.depthStencilState.depthWriteEnable = VK_TRUE;
  state.depthStencilState.depthCompareOp   = VK_COMPARE_OP_ALWAYS;
  state.rasterizationState.cullMode        = VK_CULL_MODE_NONE;

  graphicsGen.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main",
                        nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullScreenVertexShader));
  graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main",
                        nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullscreenBackgroundFragShader));

  graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_basicPipelines.background);

  graphicsGen.clearShaders();
  state.colorWriteMasks = {0};

  graphicsGen.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main",
                        nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullScreenVertexShader));
  graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main",
                        nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullscreenWriteDepthFragShader));

  graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_basicPipelines.writeDepth);
}

void Renderer::writeRayTracingDepthBuffer(VkCommandBuffer cmd)
{
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelineLayout, 0, 1, m_basicDset.getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelines.writeDepth);

  vkCmdDraw(cmd, 3, 1, 0, 0);
}

void Renderer::writeBackgroundSky(VkCommandBuffer cmd)
{
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelineLayout, 0, 1, m_basicDset.getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelines.background);

  vkCmdDraw(cmd, 3, 1, 0, 0);
}

}  // namespace tessellatedclusters
