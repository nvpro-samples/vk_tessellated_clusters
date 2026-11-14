/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/spirv.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/pipeline.hpp>

#include "resources.hpp"

namespace tessellatedclusters {

void Resources::beginFrame(uint32_t cycleIndex)
{
  m_cycleIndex = cycleIndex;
}

void Resources::postProcessFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  auto sec = profiler.cmdFrameSection(cmd, "Post-process");

  bool doHbao = frame.hbaoActive;
#if USE_DLSS
  // DLSS-SR requires hbao to be applied to its input, hence the renderer does it prior upscaling
  doHbao = doHbao && m_frameBuffer.dlssMode != DlssMode::eSuperResolution;
#endif

  // do hbao on the full-res input image
  if(doHbao)
  {
    cmdHBAO(cmd, frame, profiler);
  }

  if(m_frameBuffer.useResolved)
  {
    // blit to resolved
    VkImageBlit region               = {0};
    region.dstOffsets[1].x           = frame.windowSize.width;
    region.dstOffsets[1].y           = frame.windowSize.height;
    region.dstOffsets[1].z           = 1;
    region.srcOffsets[1].x           = m_frameBuffer.targetSize.width;
    region.srcOffsets[1].y           = m_frameBuffer.targetSize.height;
    region.srcOffsets[1].z           = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;

    cmdImageTransition(cmd, m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    cmdImageTransition(cmd, m_frameBuffer.imgColorResolved, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    vkCmdBlitImage(cmd, m_frameBuffer.imgColor.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   m_frameBuffer.imgColorResolved.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);

    cmdImageTransition(cmd, m_frameBuffer.imgColorResolved, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  else
  {
    cmdImageTransition(cmd, m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  {
    nvvk::cmdMemoryBarrier(cmd, s_supportedShaderStages, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferCopy region;
    region.size      = sizeof(shaderio::Readback);
    region.srcOffset = 0;
    region.dstOffset = m_cycleIndex * sizeof(shaderio::Readback);
    vkCmdCopyBuffer(cmd, m_commonBuffers.readBack.buffer, m_commonBuffers.readBackHost.buffer, 1, &region);
  }
}

void Resources::endFrame() {}

void Resources::emptyFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  auto sec = profiler.cmdFrameSection(cmd, "Render");
  cmdBeginRendering(cmd);
  vkCmdEndRendering(cmd);
}


void Resources::init(VkDevice device, VkPhysicalDevice physicalDevice, VkInstance instance, nvvk::QueueInfo queue)
{
  m_device         = device;
  m_physicalDevice = physicalDevice;
  m_queue          = queue;

  m_physicalDeviceInfo.init(physicalDevice);

  m_dumpInternal = m_dumpInternal && m_supportsPipelineExecutableInfo;

  // reversed-Z
  m_basicGraphicsState.depthStencilState.depthCompareOp = VK_COMPARE_OP_GREATER;

  {
    VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext                       = &m_meshShaderPropsEXT;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
  }

  {
    VmaAllocatorCreateInfo allocatorInfo = {
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = physicalDevice,
        .device         = device,
        .instance       = instance,
    };

    NVVK_CHECK(m_allocator.init(allocatorInfo));

    //m_allocator.setLeakID(30);
  }

  m_uploader.init(&m_allocator);

  m_samplerPool.init(device);
  m_samplerPool.acquireSampler(m_samplerLinear);

  // temp command pool
  {
    VkCommandPoolCreateInfo createInfo = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = m_queue.familyIndex,
    };

    NVVK_CHECK(vkCreateCommandPool(m_device, &createInfo, nullptr, &m_tempCommandPool));
  }

  {
    std::filesystem::path                    exeDirectoryPath = nvutils::getExecutablePath().parent_path();
    const std::vector<std::filesystem::path> searchPaths      = {
        // regular build
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_NVSHADERS_DIRECTORY),
        // install build
        std::filesystem::absolute(exeDirectoryPath / TARGET_NAME "_files" / "shaders"),
        std::filesystem::absolute(exeDirectoryPath),
    };
    m_glslCompiler.addSearchPaths(searchPaths);
    m_glslCompiler.defaultOptions();
    m_glslCompiler.defaultTarget();
    m_glslCompiler.options().SetGenerateDebugInfo();
  }

  // common resources
  {
    m_allocator.createBuffer(m_commonBuffers.frameConstants, sizeof(shaderio::FrameConstants) * 2,
                             VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    m_allocator.createBuffer(m_commonBuffers.readBack, sizeof(shaderio::Readback),
                             VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                             VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m_allocator.createBuffer(m_commonBuffers.readBackHost, sizeof(shaderio::Readback) * 4,
                             VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                             VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
  }

  {
    HbaoPass::Config config;
    config.maxFrames    = 1;
    config.targetFormat = m_frameBuffer.colorFormat;

    m_hbaoPass.init(&m_allocator, &m_samplerPool, &m_glslCompiler, config);
  }
#if USE_DLSS
  {
    DlssDenoiser::InitInfo info;
    info.resourceAllocator = &m_allocator;
    info.queueFamilyIndex  = m_queue.familyIndex;
    info.queue             = m_queue.queue;
    info.samplerPool       = &m_samplerPool;
    info.instance          = instance;

    // NGX itself is initialized on first use, it takes several seconds
    m_frameBuffer.dlssDenoiser.init(info);
  }
  {
    DlssUpscaler::InitInfo info;
    info.resourceAllocator = &m_allocator;
    info.queueFamilyIndex  = m_queue.familyIndex;
    info.queue             = m_queue.queue;
    info.samplerPool       = &m_samplerPool;
    info.instance          = instance;

    m_frameBuffer.dlssUpscaler.init(info);
  }
#endif

  {
    NVHizVK::Config config;
    config.msaaSamples             = 0;
    config.reversedZ               = true;
    config.supportsMinmaxFilter    = true;
    config.supportsSubGroupShuffle = true;
    m_hiz.init(m_device, config, 1);

    shaderc::SpvCompilationResult shaderResults[NVHizVK::SHADER_COUNT];
    for(uint32_t i = 0; i < NVHizVK::SHADER_COUNT; i++)
    {
      shaderc::CompileOptions options = makeCompilerOptions();
      m_hiz.appendShaderDefines(i, options);
      compileShader(shaderResults[i], VK_SHADER_STAGE_COMPUTE_BIT, "nvhiz-update.comp.glsl", &options);
    }
    m_hiz.initPipelines(shaderResults);
  }
}

void Resources::deinit()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_device));

  m_allocator.destroyBuffer(m_commonBuffers.frameConstants);
  m_allocator.destroyBuffer(m_commonBuffers.readBack);
  m_allocator.destroyBuffer(m_commonBuffers.readBackHost);

  vkDestroyCommandPool(m_device, m_tempCommandPool, nullptr);

  deinitFramebuffer();
  m_hbaoPass.deinit();
  m_hiz.deinit();
#if USE_DLSS
  m_frameBuffer.dlssDenoiser.deinit();
  m_frameBuffer.dlssUpscaler.deinit();
#endif

  m_samplerPool.releaseSampler(m_samplerLinear);
  m_samplerPool.deinit();
  m_uploader.deinit();
  m_allocator.deinit();
}

bool Resources::initFramebuffer(const VkExtent2D& windowSize, int supersample, bool hbaoQuarterRes)
{
  // only the HBAO resolution changed, its images are the only thing that follows it
  if(m_frameBuffer.imgColor.image != 0 && windowSize.width == m_frameBuffer.windowSize.width
     && windowSize.height == m_frameBuffer.windowSize.height && supersample == m_frameBuffer.supersample
     && hbaoQuarterRes != m_frameBuffer.hbaoQuarterRes)
  {
    m_frameBuffer.hbaoQuarterRes = hbaoQuarterRes;
    if(m_hbaoFrame.slot != ~0u)
    {
      NVVK_CHECK(vkDeviceWaitIdle(m_device));
      HbaoPass::FrameConfig config = m_hbaoFrame.config;
      config.quarterRes            = hbaoQuarterRes;

      VkCommandBuffer cmd = createTempCmdBuffer();
      m_hbaoPass.initFrame(m_hbaoFrame, config, cmd);
      tempSyncSubmit(cmd);
    }
    return true;
  }

  m_fboChangeID++;

  if(m_frameBuffer.imgColor.image != 0)
  {
    deinitFramebuffer();
  }

  // the line width is relative to this size
  VkExtent2D referenceSize = windowSize;
  bool       fixedSize     = true;  // all but the default case

  switch(supersample)
  {
    case 540:
      m_frameBuffer.targetSize = {960, 540};
      referenceSize            = m_frameBuffer.targetSize;
      break;
    case 720:
      m_frameBuffer.targetSize = {1280, 720};
      referenceSize            = m_frameBuffer.targetSize;
      break;
    case 1080:
      m_frameBuffer.targetSize = {1920, 1080};
      referenceSize            = m_frameBuffer.targetSize;
      break;
    case 1440:
      m_frameBuffer.targetSize = {2560, 1440};
      referenceSize            = m_frameBuffer.targetSize;
      break;
    case 2160:
      m_frameBuffer.targetSize = {3840, 2160};
      referenceSize            = m_frameBuffer.targetSize;
      break;
    case 1024:
      m_frameBuffer.targetSize = {1024, 1024};
      referenceSize            = m_frameBuffer.targetSize;
      break;
    case 2048:
      m_frameBuffer.targetSize = {2048, 2048};
      referenceSize            = m_frameBuffer.targetSize;
      break;
    case 4096:
      m_frameBuffer.targetSize = {4096, 4096};
      referenceSize            = m_frameBuffer.targetSize;
      break;
    default:
      fixedSize                = false;
      supersample              = std::clamp(supersample, 1, FrameBuffer::MAX_SUPERSAMPLE);
      m_frameBuffer.targetSize = {windowSize.width * supersample, windowSize.height * supersample};
      break;
  }

  m_frameBuffer.pixelScale = std::max(float(m_frameBuffer.targetSize.width) / float(referenceSize.width),
                                      float(m_frameBuffer.targetSize.height) / float(referenceSize.height));
  m_basicGraphicsState.rasterizationState.lineWidth = m_frameBuffer.pixelScale;

  m_frameBuffer.windowSize     = windowSize;
  m_frameBuffer.supersample    = supersample;
  m_frameBuffer.hbaoQuarterRes = hbaoQuarterRes;
  m_frameBuffer.fixedSize      = fixedSize;

  // may be lowered due to DLSS
  m_frameBuffer.renderSize = m_frameBuffer.targetSize;

  // fixed resolutions are displayed and saved as is, the window downscale is left to the viewport
  m_frameBuffer.useResolved =
      !fixedSize && (m_frameBuffer.targetSize.width != windowSize.width || m_frameBuffer.targetSize.height != windowSize.height);

  VkSampleCountFlagBits samplesUsed = VK_SAMPLE_COUNT_1_BIT;
  {
    // color
    VkImageCreateInfo cbImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    cbImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    cbImageInfo.format            = m_frameBuffer.colorFormat;
    cbImageInfo.extent.width      = m_frameBuffer.targetSize.width;
    cbImageInfo.extent.height     = m_frameBuffer.targetSize.height;
    cbImageInfo.extent.depth      = 1;
    cbImageInfo.mipLevels         = 1;
    cbImageInfo.arrayLayers       = 1;
    cbImageInfo.samples           = samplesUsed;
    cbImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    cbImageInfo.flags             = 0;
    cbImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    cbImageInfo.usage             = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                        | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImageViewCreateInfo cbImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    cbImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    cbImageViewInfo.format                          = m_frameBuffer.colorFormat;
    cbImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
    cbImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
    cbImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
    cbImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
    cbImageViewInfo.flags                           = 0;
    cbImageViewInfo.subresourceRange.levelCount     = 1;
    cbImageViewInfo.subresourceRange.baseMipLevel   = 0;
    cbImageViewInfo.subresourceRange.layerCount     = 1;
    cbImageViewInfo.subresourceRange.baseArrayLayer = 0;
    cbImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;

    NVVK_CHECK(m_allocator.createImage(m_frameBuffer.imgColor, cbImageInfo, cbImageViewInfo));
    NVVK_DBG_NAME(m_frameBuffer.imgColor.image);
    NVVK_DBG_NAME(m_frameBuffer.imgColor.descriptor.imageView);
  }

  VkCommandBuffer cmd = createTempCmdBuffer();

#if USE_DLSS
  if(m_frameBuffer.dlssMode != DlssMode::eNone)
  {
    updateFramebufferDlss(cmd);
  }
#endif

  LOGI("framebuffer: %4d x %4d (window)\n", m_frameBuffer.windowSize.width, m_frameBuffer.windowSize.height);
  LOGI("             %4d x %4d (target)\n", m_frameBuffer.targetSize.width, m_frameBuffer.targetSize.height);
  LOGI("             %4d x %4d (render)\n", m_frameBuffer.renderSize.width, m_frameBuffer.renderSize.height);

  // depth stencil
  m_frameBuffer.depthStencilFormat = nvvk::findDepthStencilFormat(m_physicalDevice);

  {
    VkImageCreateInfo dsImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    dsImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    dsImageInfo.format            = m_frameBuffer.depthStencilFormat;
    dsImageInfo.extent.width      = m_frameBuffer.renderSize.width;
    dsImageInfo.extent.height     = m_frameBuffer.renderSize.height;
    dsImageInfo.extent.depth      = 1;
    dsImageInfo.mipLevels         = 1;
    dsImageInfo.arrayLayers       = 1;
    dsImageInfo.samples           = samplesUsed;
    dsImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    dsImageInfo.usage             = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    dsImageInfo.flags             = 0;
    dsImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageViewCreateInfo dsImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    dsImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    dsImageViewInfo.format                          = m_frameBuffer.depthStencilFormat;
    dsImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
    dsImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
    dsImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
    dsImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
    dsImageViewInfo.flags                           = 0;
    dsImageViewInfo.subresourceRange.levelCount     = 1;
    dsImageViewInfo.subresourceRange.baseMipLevel   = 0;
    dsImageViewInfo.subresourceRange.layerCount     = 1;
    dsImageViewInfo.subresourceRange.baseArrayLayer = 0;
    dsImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_DEPTH_BIT;

    NVVK_CHECK(m_allocator.createImage(m_frameBuffer.imgDepthStencil, dsImageInfo, dsImageViewInfo));
    NVVK_DBG_NAME(m_frameBuffer.imgDepthStencil.image);
    NVVK_DBG_NAME(m_frameBuffer.imgDepthStencil.descriptor.imageView);

    dsImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    dsImageViewInfo.image                       = m_frameBuffer.imgDepthStencil.image;

    NVVK_CHECK(vkCreateImageView(m_device, &dsImageViewInfo, nullptr, &m_frameBuffer.viewDepth));

#if USE_DLSS
    if(m_frameBuffer.dlssMode == DlssMode::eSuperResolution)
    {
      m_frameBuffer.dlssUpscaler.setDepthResource(m_frameBuffer.imgDepthStencil.image, m_frameBuffer.viewDepth,
                                                  m_frameBuffer.depthStencilFormat);
    }
#endif
  }

  if(m_frameBuffer.useResolved)
  {
    // resolve image
    VkImageCreateInfo resImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    resImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    resImageInfo.format            = m_frameBuffer.colorFormat;
    resImageInfo.extent.width      = windowSize.width;
    resImageInfo.extent.height     = windowSize.height;
    resImageInfo.extent.depth      = 1;
    resImageInfo.mipLevels         = 1;
    resImageInfo.arrayLayers       = 1;
    resImageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    resImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    resImageInfo.usage             = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                         | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    resImageInfo.flags         = 0;
    resImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageViewCreateInfo resImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    resImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    resImageViewInfo.format                          = m_frameBuffer.colorFormat;
    resImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
    resImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
    resImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
    resImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
    resImageViewInfo.flags                           = 0;
    resImageViewInfo.subresourceRange.levelCount     = 1;
    resImageViewInfo.subresourceRange.baseMipLevel   = 0;
    resImageViewInfo.subresourceRange.layerCount     = 1;
    resImageViewInfo.subresourceRange.baseArrayLayer = 0;
    resImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;

    NVVK_CHECK(m_allocator.createImage(m_frameBuffer.imgColorResolved, resImageInfo, resImageViewInfo));
    NVVK_DBG_NAME(m_frameBuffer.imgColorResolved.image);
    NVVK_DBG_NAME(m_frameBuffer.imgColorResolved.descriptor.imageView);
  }

  {
    // ray tracing depth
    VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType         = VK_IMAGE_TYPE_2D;
    imageInfo.format            = m_frameBuffer.raytracingDepthFormat;
    imageInfo.extent.width      = m_frameBuffer.renderSize.width;
    imageInfo.extent.height     = m_frameBuffer.renderSize.height;
    imageInfo.extent.depth      = 1;
    imageInfo.mipLevels         = 1;
    imageInfo.arrayLayers       = 1;
    imageInfo.samples           = samplesUsed;
    imageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.flags             = 0;
    imageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                      | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImageViewCreateInfo imageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    imageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    imageViewInfo.format                          = m_frameBuffer.raytracingDepthFormat;
    imageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
    imageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
    imageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
    imageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
    imageViewInfo.flags                           = 0;
    imageViewInfo.subresourceRange.levelCount     = 1;
    imageViewInfo.subresourceRange.baseMipLevel   = 0;
    imageViewInfo.subresourceRange.layerCount     = 1;
    imageViewInfo.subresourceRange.baseArrayLayer = 0;
    imageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;

    NVVK_CHECK(m_allocator.createImage(m_frameBuffer.imgRaytracingDepth, imageInfo, imageViewInfo));
    NVVK_DBG_NAME(m_frameBuffer.imgRaytracingDepth.image);
    NVVK_DBG_NAME(m_frameBuffer.imgRaytracingDepth.descriptor.imageView);

#if USE_DLSS
    if(m_frameBuffer.dlssMode == DlssMode::eRayReconstruction)
    {
      m_frameBuffer.dlssDenoiser.setResource(DlssRayReconstruction::ResourceType::eDepth,
                                             m_frameBuffer.imgRaytracingDepth.image,
                                             m_frameBuffer.imgRaytracingDepth.descriptor.imageView,
                                             m_frameBuffer.imgRaytracingDepth.format);
    }
#endif
  }

  {
    m_hiz.setupUpdateInfos(m_hizUpdate, m_frameBuffer.renderSize.width, m_frameBuffer.renderSize.height,
                           m_frameBuffer.depthStencilFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

    // hiz
    VkImageCreateInfo hizImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    hizImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    hizImageInfo.format            = m_hizUpdate.farInfo.format;
    hizImageInfo.extent.width      = m_hizUpdate.farInfo.width;
    hizImageInfo.extent.height     = m_hizUpdate.farInfo.height;
    hizImageInfo.mipLevels         = m_hizUpdate.farInfo.mipLevels;
    hizImageInfo.extent.depth      = 1;
    hizImageInfo.arrayLayers       = 1;
    hizImageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    hizImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    hizImageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    hizImageInfo.flags = 0;
    hizImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    NVVK_CHECK(m_allocator.createImage(m_frameBuffer.imgHizFar, hizImageInfo));
    NVVK_DBG_NAME(m_frameBuffer.imgHizFar.image);

    m_hizUpdate.sourceImage = m_frameBuffer.imgDepthStencil.image;
    m_hizUpdate.farImage    = m_frameBuffer.imgHizFar.image;
    m_hizUpdate.nearImage   = VK_NULL_HANDLE;
  }

  m_hiz.initUpdateViews(m_hizUpdate);
  m_hiz.updateDescriptorSet(m_hizUpdate, 0);

  // initial resource transitions
  {
    {
      HbaoPass::FrameConfig config;
      config.targetWidth             = m_frameBuffer.renderSize.width;
      config.targetHeight            = m_frameBuffer.renderSize.height;
      config.quarterRes              = m_frameBuffer.hbaoQuarterRes;
      config.sourceDepth.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      config.sourceDepth.imageView   = m_frameBuffer.viewDepth;
      config.sourceDepth.sampler     = VK_NULL_HANDLE;
      config.targetColor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      config.targetColor.imageView   = m_frameBuffer.imgColor.descriptor.imageView;
      config.targetColor.sampler     = VK_NULL_HANDLE;
#if USE_DLSS
      if(m_frameBuffer.dlssMode == DlssMode::eSuperResolution)
      {
        // applied to the DLSS input prior upscaling
        config.targetColor.imageView = m_frameBuffer.dlssUpscaler.getGBuffer().getColorImageView(DlssUpscaler::eDlssInputColor);
      }
#endif

      m_hbaoPass.initFrame(m_hbaoFrame, config, cmd);
    }

    cmdImageTransition(cmd, m_frameBuffer.imgHizFar, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);

    {
      // far plane, nothing is occluded before the first frame wrote the hiz
      VkClearColorValue clear = {};
      VkImageSubresourceRange subResourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
      vkCmdClearColorImage(cmd, m_frameBuffer.imgHizFar.image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &subResourceRange);
    }
    cmdImageTransition(cmd, m_frameBuffer.imgRaytracingDepth, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);

    tempSyncSubmit(cmd);
  }

  {
    VkViewport vp;
    VkRect2D   sc;
    vp.x        = 0;
    vp.y        = 0;
    vp.width    = float(m_frameBuffer.renderSize.width);
    vp.height   = float(m_frameBuffer.renderSize.height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;

    sc.offset.x = 0;
    sc.offset.y = 0;
    sc.extent   = m_frameBuffer.renderSize;

    m_frameBuffer.viewport = vp;
    m_frameBuffer.scissor  = sc;
  }

  {
    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};

    pipelineRenderingInfo.colorAttachmentCount    = 1;
    pipelineRenderingInfo.pColorAttachmentFormats = &m_frameBuffer.colorFormat;
    pipelineRenderingInfo.depthAttachmentFormat   = m_frameBuffer.depthStencilFormat;

    m_frameBuffer.pipelineRenderingInfo = pipelineRenderingInfo;
  }

  return true;
}

void Resources::deinitFramebuffer()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_device));

  m_allocator.destroyImage(m_frameBuffer.imgColor);
  m_allocator.destroyImage(m_frameBuffer.imgColorResolved);
  m_allocator.destroyImage(m_frameBuffer.imgDepthStencil);
  m_allocator.destroyImage(m_frameBuffer.imgHizFar);
  m_allocator.destroyImage(m_frameBuffer.imgRaytracingDepth);

  vkDestroyImageView(m_device, m_frameBuffer.viewDepth, nullptr);
  m_frameBuffer.viewDepth = VK_NULL_HANDLE;

  m_hiz.deinitUpdateViews(m_hizUpdate);
  m_hbaoPass.deinitFrame(m_hbaoFrame);
}

#if USE_DLSS
void Resources::updateFramebufferDlss(VkCommandBuffer cmd)
{
  switch(m_frameBuffer.dlssMode)
  {
    case DlssMode::eRayReconstruction: {
      VkExtent2D renderSize = m_frameBuffer.dlssDenoiser.updateSize(cmd, m_frameBuffer.targetSize, m_frameBuffer.dlssQuality);

      if(!m_frameBuffer.dlssDenoiser.isActive())
      {
        LOGW("DLSS-RR initialization failed; falling back to native rendering.\n");
        m_frameBuffer.dlssDenoiser.deinitResources();
        m_frameBuffer.dlssMode = DlssMode::eNone;
        break;
      }

      m_frameBuffer.dlssDenoiser.setResource(DlssRayReconstruction::ResourceType::eColorOut, m_frameBuffer.imgColor.image,
                                             m_frameBuffer.imgColor.descriptor.imageView, m_frameBuffer.imgColor.format);

      m_frameBuffer.renderSize = renderSize;
      break;
    }
    case DlssMode::eSuperResolution: {
      VkExtent2D renderSize = m_frameBuffer.dlssUpscaler.updateSize(cmd, m_frameBuffer.targetSize, m_frameBuffer.dlssQuality);

      if(!m_frameBuffer.dlssUpscaler.isActive())
      {
        LOGW("DLSS-SR initialization failed; falling back to native rendering.\n");
        m_frameBuffer.dlssUpscaler.deinitResources();
        m_frameBuffer.dlssMode = DlssMode::eNone;
        break;
      }

      m_frameBuffer.dlssUpscaler.setOutputResource(m_frameBuffer.imgColor.image, m_frameBuffer.imgColor.descriptor.imageView,
                                                   m_frameBuffer.imgColor.format);
      m_frameBuffer.dlssSrColorLayout  = VK_IMAGE_LAYOUT_GENERAL;
      m_frameBuffer.dlssSrMotionLayout = VK_IMAGE_LAYOUT_GENERAL;

      m_frameBuffer.renderSize = renderSize;
      break;
    }
    case DlssMode::eNone:
    default:
      break;
  }
}

void Resources::initDlssAsync(DlssMode mode, NVSDK_NGX_PerfQuality_Value dlssQuality)
{
  // one at a time, the other is started once this one is polled
  if(isDlssLoading())
    return;

  if(mode == DlssMode::eRayReconstruction)
  {
    m_frameBuffer.dlssDenoiser.initAsync();
  }
  else if(mode == DlssMode::eSuperResolution)
  {
    m_frameBuffer.dlssUpscaler.initAsync(dlssQuality);
  }
}

bool Resources::isDlssLoading() const
{
  return m_frameBuffer.dlssDenoiser.isInitPending() || m_frameBuffer.dlssUpscaler.isInitPending();
}

bool Resources::pollDlssInit()
{
  bool finished = m_frameBuffer.dlssDenoiser.pollInit();
  finished      = m_frameBuffer.dlssUpscaler.pollInit() || finished;
  return finished;
}

void Resources::setFramebufferDlss(DlssMode mode, NVSDK_NGX_PerfQuality_Value dlssQuality)
{
  if(!m_dlssWaitForInit
     && ((mode == DlssMode::eRayReconstruction && !m_frameBuffer.dlssDenoiser.isInitialized())
         || (mode == DlssMode::eSuperResolution && !m_frameBuffer.dlssUpscaler.isInitialized())))
  {
    // render natively until the init is done
    initDlssAsync(mode, dlssQuality);
    mode = DlssMode::eNone;
  }

  if(m_frameBuffer.dlssMode != mode || m_frameBuffer.dlssQuality != dlssQuality)
  {
    m_frameBuffer.dlssMode    = mode;
    m_frameBuffer.dlssQuality = dlssQuality;

    if(mode == DlssMode::eRayReconstruction)
    {
      m_frameBuffer.dlssDenoiser.ensureInitialized();
    }
    else if(mode == DlssMode::eSuperResolution)
    {
      m_frameBuffer.dlssUpscaler.ensureInitialized(dlssQuality);
    }

    if(mode != DlssMode::eRayReconstruction)
    {
      m_frameBuffer.dlssDenoiser.deinitResources();
    }
    if(mode != DlssMode::eSuperResolution)
    {
      m_frameBuffer.dlssUpscaler.deinitResources();
    }

    initFramebuffer(m_frameBuffer.windowSize, m_frameBuffer.supersample, m_frameBuffer.hbaoQuarterRes);
  }
}
#endif

void Resources::getReadbackData(shaderio::Readback& readback)
{
  const shaderio::Readback* pReadback = m_commonBuffers.readBackHost.data();
  readback                            = pReadback[m_cycleIndex];
}

void Resources::cmdBuildHiz(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  auto timerSection = profiler.cmdFrameSection(cmd, "HiZ");

  // transition depth read optimal
  cmdImageTransition(cmd, m_frameBuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  m_hiz.cmdUpdateHiz(cmd, m_hizUpdate, (uint32_t)0);
}

#if USE_DLSS
static void cmdTransitionImageLayout(VkCommandBuffer cmd, VkImage image, VkImageLayout& currentLayout, VkImageLayout newLayout, VkImageAspectFlags aspects)
{
  if(currentLayout == newLayout)
    return;

  nvvk::ImageMemoryBarrierParams imageBarrier;
  imageBarrier.image                       = image;
  imageBarrier.oldLayout                   = currentLayout;
  imageBarrier.newLayout                   = newLayout;
  imageBarrier.subresourceRange.aspectMask = aspects;

  nvvk::cmdImageMemoryBarrier(cmd, imageBarrier);

  currentLayout = newLayout;
}
#endif

void Resources::cmdHBAO(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  auto timerSection = profiler.cmdFrameSection(cmd, "HBAO");

  // transition color to general
#if USE_DLSS
  if(m_frameBuffer.dlssMode == DlssMode::eSuperResolution)
  {
    cmdTransitionImageLayout(cmd, m_frameBuffer.dlssUpscaler.getGBuffer().getColorImage(DlssUpscaler::eDlssInputColor),
                             m_frameBuffer.dlssSrColorLayout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
  }
  else
#endif
  {
    cmdImageTransition(cmd, m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
  }

  // transition depth read optimal
  cmdImageTransition(cmd, m_frameBuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  m_hbaoPass.cmdCompute(cmd, m_hbaoFrame, frame.hbaoSettings);

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
}

bool Resources::compileShader(shaderc::SpvCompilationResult& compiled,
                              VkShaderStageFlagBits          shaderStage,
                              const std::filesystem::path&   filePath,
                              shaderc::CompileOptions*       options,
                              const std::filesystem::path&   dumpFileName)
{
  compiled = m_glslCompiler.compileFile(filePath, nvvkglsl::getShaderKind(shaderStage), options);
  if(compiled.GetCompilationStatus() == shaderc_compilation_status_success)
  {
    if(m_dumpSpirv)
    {
      // dump spirv files for improved aftermath debugging
      std::filesystem::path dumpFile = dumpFileName.empty() ? filePath.stem() : dumpFileName;
      dumpFile += ".spirv";

      nvutils::dumpSpirv(dumpFile, nvvkglsl::GlslCompiler::getSpirv(compiled), nvvkglsl::GlslCompiler::getSpirvSize(compiled));
    }
    return true;
  }
  else
  {
    std::string errorMessage = compiled.GetErrorMessage();
    if(!errorMessage.empty())
      nvutils::Logger::getInstance().log(nvutils::Logger::LogLevel::eWARNING, "%s", errorMessage.c_str());
    return false;
  }
}

bool Resources::compileShader(Shader&                      shader,
                              VkShaderStageFlagBits        shaderStage,
                              const std::filesystem::path& filePath,
                              shaderc::CompileOptions*     options,
                              const char*                  variant)
{
  shader.fileName = filePath.stem();
  if(variant)
  {
    shader.fileName += std::string(".") + variant;
  }
  return compileShader(shader.compiled, shaderStage, filePath, options, shader.fileName);
}

void Resources::dumpPipelineInternals(VkPipeline pipeline, const Shader& shader)
{
  if(!m_dumpInternal)
    return;

  nvvk::dumpPipelineInternals(m_device, pipeline, shader.fileName);
}

void Resources::createComputePipeline(VkComputePipelineCreateInfo& compInfo, const Shader& shader, VkPipeline& pipeline)
{
  VkShaderModuleCreateInfo shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(shader);

  compInfo.stage.pNext = &shaderInfo;
  compInfo.flags |= getPipelineCreateFlags();

  NVVK_CHECK(vkCreateComputePipelines(m_device, nullptr, 1, &compInfo, nullptr, &pipeline));

  dumpPipelineInternals(pipeline, shader);
}

VkCommandBuffer Resources::createTempCmdBuffer()
{
  VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool                 = m_tempCommandPool;
  allocInfo.commandBufferCount          = 1;

  VkCommandBuffer cmd;
  NVVK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, &cmd));

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  beginInfo.pInheritanceInfo         = nullptr;

  NVVK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

  return cmd;
}

void Resources::tempSyncSubmit(VkCommandBuffer cmd)
{
  vkEndCommandBuffer(cmd);

  VkCommandBufferSubmitInfo cmdInfo = {
      .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cmd,
  };

  VkSubmitInfo2 submitInfo2 = {
      .sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .flags                  = 0,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos    = &cmdInfo,
  };

  NVVK_CHECK(vkQueueSubmit2(m_queue.queue, 1, &submitInfo2, nullptr));
  NVVK_CHECK(vkQueueWaitIdle(m_queue.queue));

  vkFreeCommandBuffers(m_device, m_tempCommandPool, 1, &cmd);
}

void Resources::cmdBeginRendering(VkCommandBuffer cmd, bool hasSecondary, VkAttachmentLoadOp loadOpColor, VkAttachmentLoadOp loadOpDepth)
{
  VkClearValue colorClear{.color = {m_bgColor.x, m_bgColor.y, m_bgColor.z, m_bgColor.w}};
  VkClearValue depthClear{.depthStencil = {0.0F, 0}};

  cmdImageTransition(cmd, m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  cmdImageTransition(cmd, m_frameBuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

  VkRenderingAttachmentInfo colorAttachment = {
      .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView   = m_frameBuffer.imgColor.descriptor.imageView,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp      = loadOpColor,
      .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue  = colorClear,
  };

  // Shared depth attachment
  VkRenderingAttachmentInfo depthStencilAttachment{
      .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView   = m_frameBuffer.imgDepthStencil.descriptor.imageView,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp      = loadOpDepth,
      .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue  = depthClear,
  };

  // Dynamic rendering information: color and depth attachments
  VkRenderingInfo renderingInfo{
      .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .flags                = hasSecondary ? VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT : VkRenderingFlags(0),
      .renderArea           = m_frameBuffer.scissor,
      .layerCount           = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments    = &colorAttachment,
      .pDepthAttachment     = &depthStencilAttachment,
  };

  vkCmdBeginRendering(cmd, &renderingInfo);

  vkCmdSetViewportWithCount(cmd, 1, &m_frameBuffer.viewport);
  vkCmdSetScissorWithCount(cmd, 1, &m_frameBuffer.scissor);
}

#if USE_DLSS
void Resources::cmdBeginRenderingDlssSr(VkCommandBuffer cmd, VkAttachmentLoadOp loadOpColor, VkAttachmentLoadOp loadOpDepth, VkAttachmentLoadOp loadOpMotion)
{
  VkClearValue colorClear{.color = {m_bgColor.x, m_bgColor.y, m_bgColor.z, m_bgColor.w}};
  VkClearValue motionClear{.color = {0.0F, 0.0F, 0.0F, 0.0F}};
  VkClearValue depthClear{.depthStencil = {0.0F, 0}};

  auto& gbuffer = m_frameBuffer.dlssUpscaler.getGBuffer();

  cmdTransitionImageLayout(cmd, gbuffer.getColorImage(DlssUpscaler::eDlssInputColor), m_frameBuffer.dlssSrColorLayout,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
  cmdTransitionImageLayout(cmd, gbuffer.getColorImage(DlssUpscaler::eDlssMotion), m_frameBuffer.dlssSrMotionLayout,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
  cmdImageTransition(cmd, m_frameBuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

  VkRenderingAttachmentInfo colorAttachments[2] = {
      {
          .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
          .imageView   = gbuffer.getColorImageView(DlssUpscaler::eDlssInputColor),
          .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          .loadOp      = loadOpColor,
          .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
          .clearValue  = colorClear,
      },
      {
          .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
          .imageView   = gbuffer.getColorImageView(DlssUpscaler::eDlssMotion),
          .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          .loadOp      = loadOpMotion,
          .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
          .clearValue  = motionClear,
      },
  };

  VkRenderingAttachmentInfo depthStencilAttachment{
      .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView   = m_frameBuffer.imgDepthStencil.descriptor.imageView,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp      = loadOpDepth,
      .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue  = depthClear,
  };

  VkRenderingInfo renderingInfo{
      .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea           = m_frameBuffer.scissor,
      .layerCount           = 1,
      .colorAttachmentCount = 2,
      .pColorAttachments    = colorAttachments,
      .pDepthAttachment     = &depthStencilAttachment,
  };

  vkCmdBeginRendering(cmd, &renderingInfo);

  vkCmdSetViewportWithCount(cmd, 1, &m_frameBuffer.viewport);
  vkCmdSetScissorWithCount(cmd, 1, &m_frameBuffer.scissor);
}

void Resources::cmdPrepareDlssSrEvaluation(VkCommandBuffer cmd)
{
  auto& gbuffer = m_frameBuffer.dlssUpscaler.getGBuffer();

  cmdTransitionImageLayout(cmd, gbuffer.getColorImage(DlssUpscaler::eDlssInputColor), m_frameBuffer.dlssSrColorLayout,
                           VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
  cmdTransitionImageLayout(cmd, gbuffer.getColorImage(DlssUpscaler::eDlssMotion), m_frameBuffer.dlssSrMotionLayout,
                           VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
  cmdImageTransition(cmd, m_frameBuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, VK_IMAGE_LAYOUT_GENERAL);
  cmdImageTransition(cmd, m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
}

bool Resources::cmdEvaluateDlssSr(VkCommandBuffer cmd, glm::vec2 jitter, nvvk::ProfilerGpuTimer& profiler)
{
  auto timerSection = profiler.cmdFrameSection(cmd, "DLSS-SR");
  bool success      = m_frameBuffer.dlssUpscaler.upscale(cmd, jitter);
  if(!success && !m_dlssEvaluateFailed)
  {
    LOGE("DLSS-SR evaluation failed; falling back to native rendering.\n");
    m_dlssEvaluateFailed = true;
  }
  return success;
}
#endif

void Resources::cmdBeginRayTracing(VkCommandBuffer cmd)
{
  cmdImageTransition(cmd, m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
}

void Resources::cmdImageTransition(VkCommandBuffer cmd, nvvk::Image& rimg, VkImageAspectFlags aspects, VkImageLayout newLayout, bool needBarrier) const
{
  if(newLayout == rimg.descriptor.imageLayout && !needBarrier)
    return;

  nvvk::ImageMemoryBarrierParams imageBarrier;
  imageBarrier.image                       = rimg.image;
  imageBarrier.oldLayout                   = rimg.descriptor.imageLayout;
  imageBarrier.newLayout                   = newLayout;
  imageBarrier.subresourceRange.aspectMask = aspects;

  nvvk::cmdImageMemoryBarrier(cmd, imageBarrier);

  rimg.descriptor.imageLayout = newLayout;
}

}  // namespace tessellatedclusters
