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

#include <nvvk/raytraceKHR_vk.hpp>
#include <glm/ext/scalar_constants.hpp>
#include "glm/gtc/constants.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "renderer.hpp"
#include "vk_nv_cluster_acc.h"
#include "shaders/shaderio.h"

namespace tessellatedclusters {

bool Renderer::initBasicShaders(Resources& res, Scene& scene, const RendererConfig& config)
{
  m_basicShaders.fullScreenVertexShader = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "fullscreen.vert.glsl");
  m_basicShaders.fullscreenWriteDepthFragShader =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "fullscreen_write_depth.frag.glsl");

  return res.verifyShaders(m_basicShaders);
}

void Renderer::initBasics(Resources& res, Scene& scene, const RendererConfig& config)
{
  m_renderInstances.resize(scene.m_instances.size() * config.numSceneCopies);


  initWriteRayTracingDepthBuffer(res, scene, config);

  srand(2342);
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

  std::default_random_engine            rng(std::random_device{}());
  std::uniform_real_distribution<float> randomUnorm(0.0f, 1.0f);

  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    size_t originalIndex = i % scene.m_instances.size();
    size_t copyIndex     = i / scene.m_instances.size();

    // todo modify matrix etc. for grid layout

    shaderio::RenderInstance& renderInstance = m_renderInstances[i];
    const uint32_t            geometryID     = scene.m_instances[originalIndex].geometryID;
    const Scene::Geometry&    geometry       = scene.m_geometries[geometryID];

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

  m_renderInstanceBuffer =
      res.createBuffer(sizeof(shaderio::RenderInstance) * m_renderInstances.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  res.simpleUploadBuffer(m_renderInstanceBuffer, m_renderInstances.data());

  m_resourceReservedUsage.operationsMemBytes += m_renderInstanceBuffer.info.range;
  m_resourceReservedUsage.geometryMemBytes = scene.m_sceneMemBytes;
}


void Renderer::deinitBasics(Resources& res)
{
  res.destroyShaders(m_basicShaders);

  vkDestroyPipeline(res.m_device, m_writeDepthBufferPipeline, nullptr);
  m_writeDepthBufferPipeline = VK_NULL_HANDLE;

  m_writeDepthBufferDsetContainer.deinit();

  res.destroy(m_renderInstanceBuffer);
}

void Renderer::updatedFrameBufferBasics(Resources& res)
{
  {
    std::array<VkWriteDescriptorSet, 1> writeSets;

    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageView   = res.m_framebuffer.viewRaytracingDepth;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    writeSets[0] = m_writeDepthBufferDsetContainer.makeWrite(0, BINDINGS_RAYTRACING_DEPTH, &imgInfo);
    vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);
  }
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
  m_tlasInstancesBuffer = res.createBuffer(tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR),
                                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceReservedUsage.operationsMemBytes += tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR);
  res.simpleUploadBuffer(m_tlasInstancesBuffer, tlasInstances.data());
  res.tempResetResources();

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

  m_tlas = res.createAccelKHR(createInfo);
  m_resourceReservedUsage.rtTlasMemBytes += createInfo.size;
  // Allocate the scratch memory
  m_tlasScratchBuffer = res.createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
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
  res.destroy(m_tlasInstancesBuffer);
  res.destroy(m_tlasScratchBuffer);
  res.destroy(m_tlas);
}

void Renderer::initWriteRayTracingDepthBuffer(Resources& res, Scene& scene, const RendererConfig& config)
{
  VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  m_writeDepthBufferDsetContainer.init(res.m_device);

  m_writeDepthBufferDsetContainer.addBinding(BINDINGS_RAYTRACING_DEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stageFlags);
  m_writeDepthBufferDsetContainer.initLayout();
  m_writeDepthBufferDsetContainer.initPipeLayout();

  m_writeDepthBufferDsetContainer.initPool(1);
  std::array<VkWriteDescriptorSet, 1> writeSets;

  VkDescriptorImageInfo imgInfo{};
  imgInfo.imageView   = res.m_framebuffer.viewRaytracingDepth;
  imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  writeSets[0] = m_writeDepthBufferDsetContainer.makeWrite(0, BINDINGS_RAYTRACING_DEPTH, &imgInfo);
  vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);

  nvvk::GraphicsPipelineState state = res.m_basicGraphicsState;

  nvvk::GraphicsPipelineGenerator gfxGen(res.m_device, m_writeDepthBufferDsetContainer.getPipeLayout(),
                                         res.m_framebuffer.pipelineRenderingInfo, state);

  state.setBlendAttachmentColorMask(0, 0);
  state.depthStencilState.depthWriteEnable = VK_TRUE;
  state.depthStencilState.depthCompareOp   = VK_COMPARE_OP_ALWAYS;
  state.rasterizationState.cullMode        = VK_CULL_MODE_NONE;
  gfxGen.addShader(res.m_shaderManager.get(m_basicShaders.fullScreenVertexShader), VK_SHADER_STAGE_VERTEX_BIT);
  gfxGen.addShader(res.m_shaderManager.get(m_basicShaders.fullscreenWriteDepthFragShader), VK_SHADER_STAGE_FRAGMENT_BIT);
  m_writeDepthBufferPipeline = gfxGen.createPipeline();
}

void Renderer::writeRayTracingDepthBuffer(VkCommandBuffer cmd)
{
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_writeDepthBufferDsetContainer.getPipeLayout(), 0, 1,
                          m_writeDepthBufferDsetContainer.getSets(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_writeDepthBufferPipeline);

  vkCmdDraw(cmd, 3, 1, 0, 0);
}

}  // namespace tessellatedclusters
