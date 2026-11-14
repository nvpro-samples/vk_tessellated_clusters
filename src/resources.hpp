/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>

#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif

#include <glm/glm.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/alignment.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvkglsl/glsl.hpp>

#if VK_HEADER_VERSION < 309
#error Update Vulkan SDK >= 1.4.309.0
#endif

#include "hbao_pass.hpp"
#include "nvhiz_vk.hpp"
#include "../shaders/shaderio.h"

#if USE_DLSS
#include "dlss_denoiser.hpp"
#include "dlss_upscaler.hpp"
#endif

namespace tessellatedclusters {

struct Shader
{
  shaderc::SpvCompilationResult compiled;
  std::filesystem::path         fileName;

  operator shaderc::SpvCompilationResult&() { return compiled; }
  operator const shaderc::SpvCompilationResult&() const { return compiled; }
};

struct FrameConfig
{
  VkExtent2D windowSize;

  bool freezeCulling = false;
  bool hbaoActive    = true;

  HbaoPass::Settings hbaoSettings;

  // must be kept next to each other
  shaderio::FrameConstants frameConstants;
  shaderio::FrameConstants frameConstantsLast;
};

//////////////////////////////////////////////////////////////////////////

struct PhysicalDeviceInfo
{
  VkPhysicalDeviceProperties         properties10;
  VkPhysicalDeviceVulkan11Properties properties11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
  VkPhysicalDeviceVulkan12Properties properties12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
  VkPhysicalDeviceVulkan13Properties properties13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
  VkPhysicalDeviceVulkan14Properties properties14 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES};

  VkPhysicalDeviceFeatures         features10;
  VkPhysicalDeviceVulkan11Features features11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features features12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features features13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceVulkan14Features features14 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};

  void init(VkPhysicalDevice physicalDevice, uint32_t apiVersion = VK_API_VERSION_1_4)
  {
    assert(apiVersion >= VK_API_VERSION_1_2);

    VkPhysicalDeviceProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext                       = &properties11;
    properties11.pNext                = &properties12;
    if(apiVersion >= VK_API_VERSION_1_3)
    {
      properties12.pNext = &properties13;
    }
    if(apiVersion >= VK_API_VERSION_1_4)
    {
      properties13.pNext = &properties14;
    }
    vkGetPhysicalDeviceProperties2(physicalDevice, &props);
    properties10 = props.properties;

    VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext                     = &features11;
    features11.pNext                   = &features12;
    if(apiVersion >= VK_API_VERSION_1_3)
    {
      features12.pNext = &features13;
    }
    if(apiVersion >= VK_API_VERSION_1_4)
    {
      features13.pNext = &features14;
    }
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
    features10 = features.features;
  }
};

inline void cmdCopyBuffer(VkCommandBuffer cmd, const nvvk::Buffer& src, const nvvk::Buffer& dst)
{
  VkBufferCopy cpy = {0, 0, src.bufferSize};
  vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &cpy);
}

//////////////////////////////////////////////////////////////////////////

enum VisRtCompletMode
{
  VIS_RT_COMPLET_NONE,
  VIS_RT_COMPLET_TLAS,
  VIS_RT_COMPLET_BLAS,
  VIS_RT_COMPLET_CLUSTERS,
};

struct BufferRanges
{
  VkDeviceSize tempOffset = 0;

  VkDeviceSize beginOffset = 0;
  VkDeviceSize splitOffset = 0;

  VkDeviceSize append(VkDeviceSize size, VkDeviceSize alignment)
  {
    tempOffset = nvutils::align_up(tempOffset, alignment);

    VkDeviceSize offset = tempOffset;
    tempOffset += size;

    return offset;
  }

  void beginOverlap()
  {
    beginOffset = tempOffset;
    splitOffset = 0;
  }
  void splitOverlap()
  {
    splitOffset = std::max(splitOffset, tempOffset);
    tempOffset  = beginOffset;
  }
  void endOverlap() { tempOffset = std::max(splitOffset, tempOffset); }

  VkDeviceSize getSize(VkDeviceSize alignment = 4) { return nvutils::align_up(tempOffset, alignment); }
};


class Resources
{
public:
  static constexpr VkPipelineStageFlags2 ALL_SHADER_STAGES =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
      | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

#if USE_DLSS
  enum class DlssMode
  {
    eNone,
    eRayReconstruction,
    eSuperResolution,
  };
#endif

  struct FrameBuffer
  {
    static constexpr int MAX_SUPERSAMPLE = 4;

    // the resolution we render at, lowered when DLSS is active
    VkExtent2D renderSize{};
    // depends on supersample, the output of DLSS
    VkExtent2D targetSize{};
    VkExtent2D windowSize{};

    // 1 to MAX_SUPERSAMPLE: multiple of the window size along width and height
    // 540, 720, 1080, 1440, 2160: fixed 16:9 resolution with that height
    // 1024, 2048, 4096: fixed square resolution
    int  supersample    = 0;
    bool hbaoQuarterRes = true;
    bool useResolved    = false;
    // fixed target resolution, independent of the window: no resolve, the viewport letterboxes it
    bool  fixedSize  = false;
    float pixelScale = 1;

    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depthStencilFormat;

    VkViewport viewport;
    VkRect2D   scissor;

    nvvk::Image imgColor         = {};
    nvvk::Image imgColorResolved = {};
    nvvk::Image imgDepthStencil  = {};

    VkImageView viewDepth = VK_NULL_HANDLE;

    VkFormat    raytracingDepthFormat = VK_FORMAT_R32_SFLOAT;
    nvvk::Image imgRaytracingDepth    = {};

    nvvk::Image imgHizFar = {};

    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};

#if USE_DLSS
    DlssMode                    dlssMode           = DlssMode::eNone;
    DlssDenoiser                dlssDenoiser       = {};
    DlssUpscaler                dlssUpscaler       = {};
    NVSDK_NGX_PerfQuality_Value dlssQuality        = NVSDK_NGX_PerfQuality_Value(-1);
    VkImageLayout               dlssSrColorLayout  = VK_IMAGE_LAYOUT_GENERAL;
    VkImageLayout               dlssSrMotionLayout = VK_IMAGE_LAYOUT_GENERAL;
#endif
  };


  void init(VkDevice device, VkPhysicalDevice physicalDevice, VkInstance instance, nvvk::QueueInfo queue);
  void deinit();

  bool initFramebuffer(const VkExtent2D& windowSize, int supersample, bool hbaoQuarterRes);
  void deinitFramebuffer();

  glm::vec2 getFramebufferWindow2RenderScale() const
  {
    return glm::vec2(m_frameBuffer.renderSize.width, m_frameBuffer.renderSize.height)
           / glm::vec2(m_frameBuffer.windowSize.width, m_frameBuffer.windowSize.height);
  }
#if USE_DLSS
  // re-creates the framebuffer if mode or quality changed, falls back to `eNone` if DLSS fails to initialize
  // or, without `m_dlssWaitForInit`, while its NGX init is still running
  void setFramebufferDlss(DlssMode mode, NVSDK_NGX_PerfQuality_Value dlssQuality);
  void updateFramebufferDlss(VkCommandBuffer cmd);

  // NGX init and first feature creation take several seconds, they run on a worker thread
  void initDlssAsync(DlssMode mode, NVSDK_NGX_PerfQuality_Value dlssQuality);
  bool isDlssLoading() const;
  // returns true once after an async init finished
  bool pollDlssInit();

  bool m_dlssWaitForInit = true;
  // set when an evaluation failed, the app then disables DLSS at the next safe point
  bool m_dlssEvaluateFailed = false;
#endif

  void beginFrame(uint32_t cycleIndex);
  void postProcessFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  void emptyFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  void endFrame();

  void cmdBuildHiz(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  void cmdHBAO(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);

  void getReadbackData(shaderio::Readback& readback);

  //////////////////////////////////////////////////////////////////////////

  shaderc::CompileOptions makeCompilerOptions() { return shaderc::CompileOptions(m_glslCompiler.options()); }

  bool compileShader(shaderc::SpvCompilationResult& compiled,
                     VkShaderStageFlagBits          shader,
                     const std::filesystem::path&   filePath,
                     shaderc::CompileOptions*       options      = nullptr,
                     const std::filesystem::path&   dumpFileName = {});

  // `variant` must be provided when the same file is compiled more than once (different defines),
  // it keeps the shader's file name, and therefore the dumped files, unique
  bool compileShader(Shader&                      shader,
                     VkShaderStageFlagBits        shaderStage,
                     const std::filesystem::path& filePath,
                     shaderc::CompileOptions*     options = nullptr,
                     const char*                  variant = nullptr);

  // flags all pipelines must be created with, so that `dumpPipelineInternals` can work
  VkPipelineCreateFlags getPipelineCreateFlags() const
  {
    return m_dumpInternal ? VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR : 0;
  }

  // dumps the driver's internal representation (isa etc.) into the working directory,
  // named after the shader's file. does nothing unless `m_dumpInternal` is active.
  void dumpPipelineInternals(VkPipeline pipeline, const Shader& shader);

  // sets up compInfo's stage.pNext and flags, and dumps the pipeline internals if enabled
  void createComputePipeline(VkComputePipelineCreateInfo& compInfo, const Shader& shader, VkPipeline& pipeline);

  // tests if all shaders compiled well, returns false if not
  // also destroys all shaders if not all were successful.
  bool verifyShaders(size_t numShaders, const Shader* shaders)
  {
    for(size_t i = 0; i < numShaders; i++)
    {
      if(shaders[i].compiled.GetCompilationStatus() != shaderc_compilation_status_null_result_object
         && shaders[i].compiled.GetCompilationStatus() != shaderc_compilation_status_success)
        return false;
    }

    return true;
  }
  template <typename T>
  bool verifyShaders(T& container)
  {
    return verifyShaders(sizeof(T) / sizeof(Shader), (const Shader*)&container);
  }

  void destroyPipelines(size_t numPipelines, VkPipeline* pipelines)
  {
    for(size_t i = 0; i < numPipelines; i++)
    {
      vkDestroyPipeline(m_device, pipelines[i], nullptr);
      pipelines[i] = nullptr;
    }
  }
  template <typename T>
  void destroyPipelines(T& container)
  {
    destroyPipelines(sizeof(T) / sizeof(VkPipeline), (VkPipeline*)&container);
  }

  //////////////////////////////////////////////////////////////////////////

  VkCommandBuffer createTempCmdBuffer();
  void            tempSyncSubmit(VkCommandBuffer cmd);

  //////////////////////////////////////////////////////////////////////////

  void cmdBeginRendering(VkCommandBuffer    cmd,
                         bool               hasSecondary = false,
                         VkAttachmentLoadOp loadOpColor  = VK_ATTACHMENT_LOAD_OP_CLEAR,
                         VkAttachmentLoadOp loadOpDepth  = VK_ATTACHMENT_LOAD_OP_CLEAR);
#if USE_DLSS
  // renders into the DLSS-SR input color and motion vectors
  void cmdBeginRenderingDlssSr(VkCommandBuffer    cmd,
                               VkAttachmentLoadOp loadOpColor  = VK_ATTACHMENT_LOAD_OP_CLEAR,
                               VkAttachmentLoadOp loadOpDepth  = VK_ATTACHMENT_LOAD_OP_CLEAR,
                               VkAttachmentLoadOp loadOpMotion = VK_ATTACHMENT_LOAD_OP_CLEAR);
  void cmdPrepareDlssSrEvaluation(VkCommandBuffer cmd);
  bool cmdEvaluateDlssSr(VkCommandBuffer cmd, glm::vec2 jitter, nvvk::ProfilerGpuTimer& profiler);
#endif
  void cmdBeginRayTracing(VkCommandBuffer cmd);

  void cmdImageTransition(VkCommandBuffer cmd, nvvk::Image& rimg, VkImageAspectFlags aspects, VkImageLayout newLayout, bool needBarrier = false) const;

  //////////////////////////////////////////////////////////////////////////

  void simpleUploadBuffer(const nvvk::Buffer& buffer, void* data)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_uploader.appendBuffer(buffer, 0, buffer.bufferSize, data);
    m_uploader.cmdUploadAppended(cmd);
    tempSyncSubmit(cmd);
    m_uploader.releaseStaging();
  }

  void simpleUploadBuffer(const nvvk::Buffer& buffer, size_t offset, size_t sz, void* data)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_uploader.appendBuffer(buffer, offset, sz, data);
    m_uploader.cmdUploadAppended(cmd);
    tempSyncSubmit(cmd);
    m_uploader.releaseStaging();
  }

  enum FlushState
  {
    ALLOW_FLUSH,
    DONT_FLUSH,
  };

  class BatchedUploader
  {
  public:
    BatchedUploader(Resources& resources, VkDeviceSize maxBatchSize = 128 * 1024 * 1024)
        : m_resources(resources)
        , m_maxBatchSize(maxBatchSize)
    {
    }

    template <typename T>
    T* uploadBuffer(const nvvk::Buffer& dst, size_t offset, size_t sz, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
    {
      if(sz)
      {
        if(m_resources.m_uploader.checkAppendedSize(m_maxBatchSize, sz) && flushState == FlushState::ALLOW_FLUSH)
        {
          flush();
        }

        if(!m_cmd)
        {
          m_cmd = m_resources.createTempCmdBuffer();
        }
        T* mapping = nullptr;
        NVVK_CHECK(m_resources.m_uploader.appendBufferMapping(dst, offset, sz, mapping));

        if(src)
        {
          memcpy(mapping, src, sz);
        }

        return mapping;
      }
      return nullptr;
    }

    template <typename T>
    T* uploadBuffer(const nvvk::Buffer& dst, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
    {
      return uploadBuffer(dst, 0, dst.bufferSize, src, flushState);
    }

    void fillBuffer(const nvvk::Buffer& dst, uint32_t fillValue)
    {
      if(!m_cmd)
      {
        m_cmd = m_resources.createTempCmdBuffer();
      }
      vkCmdFillBuffer(m_cmd, dst.buffer, 0, dst.bufferSize, fillValue);
    }

    // must call flush at end of operations
    void flush()
    {
      if(m_cmd)
      {
        m_resources.m_uploader.cmdUploadAppended(m_cmd);
        m_resources.tempSyncSubmit(m_cmd);
        m_resources.m_uploader.releaseStaging();
        m_cmd = nullptr;
      }
    }

    ~BatchedUploader() { assert(!m_cmd && "must call flush at end"); }

  private:
    Resources&      m_resources;
    VkDeviceSize    m_maxBatchSize = 0;
    VkCommandBuffer m_cmd          = nullptr;
  };

  //////////////////////////////////////////////////////////////////////////

  static constexpr VkPipelineStageFlags2 s_supportedShaderStages =
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
      | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

  VkDevice         m_device          = {};
  VkPhysicalDevice m_physicalDevice  = {};
  nvvk::QueueInfo  m_queue           = {};
  VkCommandPool    m_tempCommandPool = {};

  nvvk::ResourceAllocator m_allocator     = {};
  nvvk::SamplerPool       m_samplerPool   = {};
  VkSampler               m_samplerLinear = {};
  nvvkglsl::GlslCompiler  m_glslCompiler  = {};
  nvvk::StagingUploader   m_uploader      = {};

  FrameBuffer m_frameBuffer;
  struct CommonBuffers
  {
    nvvk::BufferTyped<shaderio::FrameConstants> frameConstants;
    nvvk::BufferTyped<shaderio::Readback>       readBack;
    nvvk::BufferTyped<shaderio::Readback>       readBackHost;
  } m_commonBuffers;

  PhysicalDeviceInfo          m_physicalDeviceInfo = {};
  nvvk::GraphicsPipelineState m_basicGraphicsState = {};
  uint32_t                    m_cycleIndex         = 0;
  size_t                      m_fboChangeID        = ~0;
  glm::vec4                   m_bgColor            = {0.1, 0.13, 0.15, 1.0};

  VkPhysicalDeviceMeshShaderPropertiesEXT m_meshShaderPropsEXT = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};

  bool m_supportsClusters               = false;
  bool m_supportsMeshShaderNV           = false;
  bool m_supportsPipelineExecutableInfo = false;
  bool m_dumpSpirv                      = false;
  bool m_dumpInternal                   = false;

  HbaoPass        m_hbaoPass;
  HbaoPass::Frame m_hbaoFrame;

  NVHizVK                       m_hiz;
  NVHizVK::Update               m_hizUpdate;
  shaderc::SpvCompilationResult m_hizShaders[NVHizVK::SHADER_COUNT];

private:
};


}  // namespace tessellatedclusters
