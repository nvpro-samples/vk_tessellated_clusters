/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdlib>

#include <fmt/format.h>
#include <fmt/chrono.h>
#include <nvutils/file_operations.hpp>
#include <nvgui/camera.hpp>

#include "tessellatedclusters.hpp"
#if USE_DLSS
#include "../shaders/dlss_util.h"
#endif

bool g_verbose = false;

namespace tessellatedclusters {

TessellatedClusters::TessellatedClusters(const Info& info)
    : m_info(info)
{
  nvutils::ProfilerTimeline::CreateInfo createInfo;
  createInfo.name = "graphics";

  m_profilerTimeline = m_info.profilerManager->createTimeline(createInfo);

  m_screenshotTimeStamp =
      fmt::format("{:%Y_%m_%d_%H_%M_%S}", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));

  m_info.parameterRegistry->add({"scene"}, {".gltf", ".glb"}, &m_sceneFilePath);
  registerCameraPathParameters();
  m_info.parameterRegistry->add({"renderer"}, (int*)&m_tweak.renderer);
  m_info.parameterRegistry->add({"verbose"}, &g_verbose, true);
  m_info.parameterRegistry->add({"resetstats"}, &m_tweak.autoResetTimers);
  m_info.parameterRegistry->add({"hbaoblur"}, &m_frameConfig.hbaoSettings.blur);
  m_info.parameterRegistry->add({"hbaoquarter", "HBAO at half width and height, upsampled depth-aware. default true"},
                                &m_tweak.hbaoQuarter);
  m_info.parameterRegistry->add({"supersample", "1 to 4 times the window size, or fixed 540, 720, 1080, 1440, 2160 and square 1024, 2048, 4096"},
                                &m_tweak.supersample);
  m_info.parameterRegistry->add({"animation"}, &m_rendererConfig.doAnimation);
  m_info.parameterRegistry->add({"culling"}, &m_rendererConfig.doCulling);
  m_info.parameterRegistry->add({"tessrate"}, &m_tweak.tessRatePixels);
  m_info.parameterRegistry->add({"visualize", "0 material, 1 clusters, 2 tessellated clusters, 3 tessellated triangles, 4 triangles"},
                                &m_tweak.visualizeMode);
  m_info.parameterRegistry->add({"aosamples", "override the number of ambient occlusion rays, 0 disables them and yields noise free images"},
                                &m_aoSamplesOverride);
  m_info.parameterRegistry->add({"gridcopies"}, &m_tweak.gridCopies);
  m_info.parameterRegistry->add({"gridconfig"}, &m_tweak.gridConfig);
  m_info.parameterRegistry->add({"clusterconfig"}, (int*)&m_tweak.clusterConfig);
  //m_info.parameterRegistry->add({"nvclusterunderfill"}, &m_sceneConfig.clusterNvConfig.costUnderfill);
  //m_info.parameterRegistry->add({"nvclusteroverlap"}, &m_sceneConfig.clusterNvConfig.costOverlap);
  //m_info.parameterRegistry->add({"nvclusterunderfillvertices"}, &m_sceneConfig.clusterNvConfig.costUnderfillVertices);
  m_info.parameterRegistry->add({"overridetime"}, &m_tweak.overrideTime);
  m_info.parameterRegistry->add({"processingthreadpct", "float percentage of threads during initial file load and processing into lod clusters, default 0.5 == 50 %"},
                                &m_sceneConfig.processingThreadsPct);
  m_info.parameterRegistry->add({"dumpspirv", "dumps compiled spirv into working directory"}, &m_resources.m_dumpSpirv);
  m_info.parameterRegistry->add({"dumpinternal", "dumps the driver internal pipeline representation into working directory"},
                                &m_resources.m_dumpInternal);
  m_info.parameterRegistry->add({"splitfactor"}, &m_rendererConfig.splitFactor);
  m_info.parameterRegistry->add({"partbits", "max part triangles in bits, default 20"}, &m_rendererConfig.numPartTriangleBits);
  m_info.parameterRegistry->add({"splitbits", "max split triangles in bits, default 16"}, &m_rendererConfig.numSplitTriangleBits);
  m_info.parameterRegistry->add({"clustermegs", "max generated CLAS data in megabytes, default 1024"},
                                &m_rendererConfig.numGeneratedClusterMegs);
  m_info.parameterRegistry->add({"vertexbits", "max generated CLAS vertices in bits, default 24"},
                                &m_rendererConfig.numGeneratedVerticesBits);
  m_info.parameterRegistry->add({"lowtess1x", "optimize clusters with 1x1x1 tessellation factors"}, &m_rendererConfig.lowTessClusters1X);
  m_info.parameterRegistry->add({"lowtess2x", "optimize triangles with tessellation factors <= 2"}, &m_rendererConfig.lowTessClusters2X);
  m_info.parameterRegistry->add({"lowtessmode", "0 transient CLAS builds, 1 combined template instantiations. default 1"},
                                (int*)&m_rendererConfig.lowTessMode);
  m_info.parameterRegistry->add({"transient2xbatch", "max 2x2x2 patches per transient cluster build. default 32"},
                                &m_rendererConfig.transient2xBatchSize);
  m_info.parameterRegistry->add({"combined1xbatch", "max independent triangles per combined template. default 64"},
                                &m_rendererConfig.combined1xBatchSize);
  m_info.parameterRegistry->add({"combined2xbatch", "max independent 2x2x2 patches per combined template. default 16"},
                                &m_rendererConfig.combined2xBatchSize);
  m_info.parameterRegistry->add({"batchmeshlets", "rasterization batches low tessellated triangles via task shader. default 1"},
                                &m_rendererConfig.rasterBatchMeshlets);
  m_info.parameterRegistry->add({"extmeshshader", "use EXT instead of NV mesh shader, forced on without NV mesh shader support"},
                                &m_rendererConfig.useEXTmeshShader);
  m_info.parameterRegistry->add({"pndisplacement", "PN-triangles displacement, a shader permutation. default 1"},
                                &m_rendererConfig.pnDisplacement);
  m_info.parameterRegistry->add({"flipwinding", "flip the triangle winding"}, &m_rendererConfig.flipWinding);
  m_info.parameterRegistry->add({"persistentkernel", "use the persistent split kernel, a shader permutation. default 0"},
                                &m_rendererConfig.persistentKernel);
  m_info.parameterRegistry->add({"persistentthreads", "threads of the persistent split kernel, default derives it from the GPU"},
                                &m_rendererConfig.persistentThreads);
  m_info.parameterRegistry->add({"visibleclusterbits", "max visible clusters in bits, default 20"},
                                &m_rendererConfig.numVisibleClusterBits);
  m_info.parameterRegistry->add({"truncatebits", "ray tracing: CLAS position mantissa bits to drop, default 0"},
                                &m_rendererConfig.positionTruncateBits);
  m_info.parameterRegistry->add({"clusterbuildmode", "ray tracing: CLAS build flags, 0 default, 1 prefer fast build, 2 prefer fast trace"},
                                (int*)&m_tweak.clusterBuildMode);
  m_info.parameterRegistry->add({"facetshading", "facet shading, only without tessellation (tessrate 0). default 1"},
                                &m_tweak.facetShading);
  m_info.parameterRegistry->add({"hbao", "rasterization: ambient occlusion via HBAO. default 1"}, &m_tweak.hbaoActive);
  m_info.parameterRegistry->add({"hbaoradius", "HBAO radius as fraction of the scene size. default 0.05"}, &m_tweak.hbaoRadius);
  m_info.parameterRegistry->add({"renderstats", "extra atomic counters for statistics, costs performance. default false"},
                                &m_rendererConfig.useRenderStats);
  m_info.parameterRegistry->add({"dlss", "enable DLSS, super resolution in rasterization and ray reconstruction in ray tracing. default true"},
                                &m_rendererConfig.useDlss);
  m_info.parameterRegistry->add({"dlssquality", "0 max performance, 1 balanced, 2 max quality, 3 ultra performance, 5 DLAA. default 2"},
                                (int*)&m_rendererConfig.dlssQuality);
  m_info.parameterRegistry->add({"dlssdelayload", "render without DLSS until it is loaded in the background, ignored headless and with sequence files. default true"},
                                &m_dlssDelayLoad);
  m_info.parameterRegistry->add({"debugui", "enable the debug ui window, defaults to on in debug builds"}, &m_debugUI);
  m_info.parameterRegistry->add({"sequencescreenshot", "save screenshot at end of each sequence. 0 disabled (default), 1 full window, 2 rendered viewport"},
                                (int*)&m_sequenceScreenshotMode, true);
  m_info.parameterRegistry->add({"screenshotframegap", "save a screenshot every n frames, and after the last headless frame. 0 disabled (default)"},
                                &m_screenshotFrameInterval);
  m_info.parameterRegistry->add({"screenshotmode", "screenshot content of screenshotframegap: 1 full window, 2 rendered viewport (default)"},
                                (int*)&m_screenshotMode);

  m_frameConfig.frameConstants                         = {};
  m_frameConfig.frameConstants.ambientOcclusionSamples = 2;
  m_frameConfig.frameConstants.facetShading            = 1;
  m_frameConfig.frameConstants.doShadow                = 1;
  m_frameConfig.frameConstants.ambientOcclusionRadius  = 0.1f;

  m_frameConfig.frameConstants                    = {};
  m_frameConfig.frameConstants.wireThickness      = 2.f;
  m_frameConfig.frameConstants.wireSmoothing      = 1.f;
  m_frameConfig.frameConstants.wireColor          = {118.f / 255.f, 185.f / 255.f, 0.f};
  m_frameConfig.frameConstants.wireStipple        = 0;
  m_frameConfig.frameConstants.wireBackfaceColor  = {0.5f, 0.5f, 0.5f};
  m_frameConfig.frameConstants.wireStippleRepeats = 5;
  m_frameConfig.frameConstants.wireStippleLength  = 0.5f;
  m_frameConfig.frameConstants.doWireframe        = 0;
  m_frameConfig.frameConstants.doShadow           = 1;

  m_frameConfig.frameConstants.animationRippleEnabled   = 1;
  m_frameConfig.frameConstants.animationRippleFrequency = 100.f;
  m_frameConfig.frameConstants.animationRippleAmplitude = 0.005f;
  m_frameConfig.frameConstants.animationRippleSpeed     = 3.7f;

  m_frameConfig.frameConstants.ambientOcclusionRadius  = 0.1f;
  m_frameConfig.frameConstants.ambientOcclusionSamples = 2;

  m_frameConfig.frameConstants.displacementOffset = 0.0f;
  m_frameConfig.frameConstants.displacementScale  = 1.0f;

  m_frameConfig.frameConstants.lightMixer = 0.5f;
}


bool TessellatedClusters::initScene(const std::filesystem::path& filePath)
{
  deinitScene();

  std::string fileName = nvutils::utf8FromPath(filePath);

  LOGI("Loading scene: %s\n", fileName.c_str());

  m_scene = std::make_unique<Scene>();
  if(!m_scene->init(filePath, m_sceneConfig, m_resources))
  {
    LOGW("Loading scene failed\n");

    m_scene = nullptr;
    return false;
  }
  else
  {
    adjustSceneClusterConfig();
  }

  m_sceneFilePath = filePath;

  return true;
}

void TessellatedClusters::deinitScene()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  if(m_scene)
  {
    m_scene->deinit(m_resources);
    m_scene = nullptr;
  }
}

void TessellatedClusters::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  m_windowSize = size;
  m_resources.initFramebuffer(m_windowSize, m_tweak.supersample, m_tweak.hbaoQuarter);
  updateImguiImage();
  if(m_renderer)
  {
    m_renderer->updatedFrameBuffer(m_resources);
  }
}

void TessellatedClusters::updateImguiImage()
{
  if(m_imguiTexture)
  {
    ImGui_ImplVulkan_RemoveTexture(m_imguiTexture);
    m_imguiTexture = nullptr;
  }

  VkImageView imageView = m_resources.m_frameBuffer.useResolved ? m_resources.m_frameBuffer.imgColorResolved.descriptor.imageView :
                                                                  m_resources.m_frameBuffer.imgColor.descriptor.imageView;

  assert(imageView);

  m_imguiTexture     = ImGui_ImplVulkan_AddTexture(m_imguiSampler, imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  m_imguiFboChangeID = m_resources.m_fboChangeID;
}

void TessellatedClusters::onPreRender()
{
  m_profilerTimeline->frameAdvance();

  if(m_screenshotFrameInterval > 0 && m_frames > 0 && (m_frames % m_screenshotFrameInterval) == 0)
  {
    saveScreenshot(m_screenshotMode, fmt::format("screenshot_{}_{:08}.jpg", m_screenshotTimeStamp, m_frames - 1));
  }
}

void TessellatedClusters::onLastHeadlessFrame()
{
  // the interval never covers the final frame, which is typically the one of interest
  if(m_screenshotFrameInterval > 0 && m_frames > 0)
  {
    saveScreenshot(m_screenshotMode, fmt::format("screenshot_{}_{:08}.jpg", m_screenshotTimeStamp, m_frames - 1));
  }
}

void TessellatedClusters::saveScreenshot(ScreenshotMode mode, const std::string& filename)
{
  if(mode == SCREENSHOT_OFF)
    return;

  // headless has no window content
  if(m_app->isHeadless())
  {
    mode = SCREENSHOT_VIEWPORT;
  }

  if(mode == SCREENSHOT_WINDOW)
  {
    m_app->saveScreenShot(std::filesystem::path(filename), 100);
  }
  else if(mode == SCREENSHOT_VIEWPORT)
  {
    // the resolved image is window sized, the render target is supersampled
    bool useResolved = m_resources.m_frameBuffer.useResolved;
    VkImage image = useResolved ? m_resources.m_frameBuffer.imgColorResolved.image : m_resources.m_frameBuffer.imgColor.image;
    VkExtent2D extent = useResolved ? m_windowSize : m_resources.m_frameBuffer.targetSize;

    m_app->saveImageToFile(image, extent, filename, 100, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
}

void TessellatedClusters::parameterSequenceCallback(const nvutils::ParameterSequencer::State& state)
{
  std::string message;
  message += fmt::format("MemoryReport {} \"{}\" = {{ \n", state.index, state.description);
  if(m_renderer)
  {
    Renderer::ResourceUsageInfo resourceActual   = m_renderer->getResourceUsage(false);
    Renderer::ResourceUsageInfo resourceReserved = m_renderer->getResourceUsage(true);

    message += fmt::format("Memory; Actual; Reserved;\n");
    message += fmt::format("Geometry; {}; {};\n", resourceActual.geometryMemBytes, resourceReserved.geometryMemBytes);
    if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_TESS)
    {
      message += fmt::format("Templates; {}; {};\n", resourceActual.rtTemplateMemBytes, resourceReserved.rtTemplateMemBytes);
      message += fmt::format("CLAS; {}; {};\n", resourceActual.rtClasMemBytes, resourceReserved.rtClasMemBytes);
      message += fmt::format("BLAS; {}; {};\n", resourceActual.rtBlasMemBytes, resourceReserved.rtBlasMemBytes);
      message += fmt::format("TLAS; {}; {};\n", resourceActual.rtTlasMemBytes, resourceReserved.rtTlasMemBytes);
    }
    message += fmt::format("Operations; {}; {};\n", resourceActual.operationsMemBytes, resourceReserved.operationsMemBytes);
    message += fmt::format("Total; {}; {};\n", resourceActual.getTotalSum(), resourceReserved.getTotalSum());

    {
      const bool isRaytrace = m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_TESS;

      shaderio::Readback readback;
      m_resources.getReadbackData(readback);

      message += fmt::format("Clusters; Count;\n");
      // ray tracing only counts full clusters as part of the render stats
      if(!isRaytrace || m_rendererConfig.useRenderStats)
      {
        message += fmt::format("Full clusters; {};\n", readback.numFullClusters);
      }
      if(isRaytrace)
      {
        message += fmt::format("Template instantiations; {};\n", readback.numTempInstantiations);
        message += fmt::format("Transient builds; {};\n", readback.numTransBuilds);
      }
      message += fmt::format("Blas clusters; {};\n", readback.numBlasClusters);
      if(m_rendererConfig.useRenderStats)
      {
        message += fmt::format("Cluster triangles; {};\n", readback.numTotalTriangles);
      }
      if(isRaytrace)
      {
        message += fmt::format("Cluster vertices; {};\n", readback.numGenVertices);
      }
      message += fmt::format("Part triangles; {};\n", readback.numPartTriangles);
      if(isRaytrace)
      {
        message += fmt::format("Part lowtess entries; {};\n", readback.numTransPartTriangles);
      }
      message += fmt::format("Split triangles; {};\n", readback.numSplitTriangles);
    }
  }
  message += "}\n";

  LOGI("%s", message.c_str());

  saveScreenshot(m_sequenceScreenshotMode, fmt::format("screenshot_{}_{}.jpg", state.index, state.description));
}

void TessellatedClusters::deinitRenderer()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  if(m_renderer)
  {
    m_renderer->deinit(m_resources);
    m_renderer = nullptr;
  }
}

static VkBuildAccelerationStructureFlagsKHR getBuildFlags(TessellatedClusters::BuildMode mode)
{
  switch(mode)
  {
    case TessellatedClusters::BUILD_DEFAULT:
      return 0;
    case TessellatedClusters::BUILD_FAST_BUILD:
      return VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    case TessellatedClusters::BUILD_FAST_TRACE:
      return VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    default:
      return 0;
  }
}

void TessellatedClusters::initRenderer(RendererType rtype)
{
  LOGI("Initializing renderer and compiling shaders\n");
  deinitRenderer();
  if(!m_scene)
    return;

  printf("init renderer %d\n", rtype);

  switch(rtype)
  {
    case RENDERER_RASTER_CLUSTERS_TESS:
      m_renderer = makeRendererRasterClustersTess();
      break;
    case RENDERER_RAYTRACE_CLUSTERS_TESS:
      m_renderer = makeRendererRayTraceClustersTess();
      break;
  }

  m_rendererConfig.gridConfig     = m_tweak.gridConfig;
  m_rendererConfig.numSceneCopies = m_tweak.gridCopies;

  m_rendererConfig.clusterBuildFlags = getBuildFlags(m_tweak.clusterBuildMode);

  if(m_renderer && !m_renderer->init(m_resources, *m_scene, m_rendererConfig))
  {
    m_renderer = nullptr;
    LOGE("Renderer init failed\n");
  }

  m_rendererFboChangeID = m_resources.m_fboChangeID;
}

void TessellatedClusters::postInitNewScene()
{
  assert(m_scene);

  float     sceneDimension = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo);
  glm::vec3 center         = (m_scene->m_bbox.hi + m_scene->m_bbox.lo) * 0.5f;

  m_frameConfig.frameConstants.wLightPos = center + sceneDimension;
  m_frameConfig.frameConstants.sceneSize = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo);

  setSceneCamera(m_sceneFilePath);

  m_frames = 0;

  {
    // Re-adjusting camera to fit the new scene
    m_info.cameraManipulator->fit(m_scene->m_bbox.lo, m_scene->m_bbox.hi, true);
    nvgui::SetHomeCamera(m_info.cameraManipulator->getCamera());
  }

  float radius = sceneDimension * 0.5f;
  m_info.cameraManipulator->setClipPlanes(glm::vec2(0.01F * radius, 100.0F * radius));
  m_info.cameraManipulator->setSpeed(radius);

  m_frameConfig.frameConstants.animationRippleEnabled   = 1;
  m_frameConfig.frameConstants.animationRippleFrequency = 50.f;
  m_frameConfig.frameConstants.animationRippleAmplitude = 0.004f;
  m_frameConfig.frameConstants.animationRippleSpeed     = 5.f;
  m_frameConfig.frameConstants.doShadow                 = 1;
  m_frameConfig.frameConstants.ambientOcclusionRadius   = 0.1f;
  m_frameConfig.frameConstants.ambientOcclusionSamples  = 2;
  m_frameConfig.frameConstants.lightMixer               = 0.5f;
  m_frameConfig.frameConstants.skyParams                = {};

  m_frameConfig.frameConstants.displacementOffset = 0.0f;
  m_frameConfig.frameConstants.displacementScale  = 1.0f;
}

void TessellatedClusters::onAttach(nvapp::Application* app)
{
  m_app = app;
  {
    VkPhysicalDeviceProperties2 physicalProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceShaderSMBuiltinsPropertiesNV smProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV};
    physicalProperties.pNext = &smProperties;
    vkGetPhysicalDeviceProperties2(app->getPhysicalDevice(), &physicalProperties);
    // pseudo heuristic
    m_rendererConfig.persistentThreads = smProperties.shaderSMCount * smProperties.shaderWarpsPerSM * 8;
  }


  {
    m_ui.enumAdd(GUI_RENDERER, RENDERER_RASTER_CLUSTERS_TESS, "Rasterization");

    m_ui.enumAdd(GUI_LOWTESSMODE, int(LOWTESS_MODE_TRANSIENT_BUILDS), "transient builds");
    m_ui.enumAdd(GUI_LOWTESSMODE, int(LOWTESS_MODE_COMBINED_TEMPLATES), "combined templates");

    m_ui.enumAdd(GUI_BUILDMODE, BUILD_DEFAULT, "default");
    m_ui.enumAdd(GUI_BUILDMODE, BUILD_FAST_BUILD, "fast build");
    m_ui.enumAdd(GUI_BUILDMODE, BUILD_FAST_TRACE, "fast trace");

    if(m_resources.m_supportsClusters)
    {
      m_ui.enumAdd(GUI_RENDERER, RENDERER_RAYTRACE_CLUSTERS_TESS, "Ray tracing");
    }
    else
    {
      LOGW("WARNING: Cluster raytracing extension not found\n");
      if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_TESS)
      {
        m_tweak.renderer = RENDERER_RASTER_CLUSTERS_TESS;
      }
    }
    if(!m_resources.m_supportsMeshShaderNV)
    {
      m_rendererConfig.useEXTmeshShader = true;
    }
    {
      for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
      {
        std::string enumStr = fmt::format("{}T_{}V", s_clusterInfos[i].tris, s_clusterInfos[i].verts);
        m_ui.enumAdd(GUI_MESHLET, s_clusterInfos[i].cfg, enumStr.c_str());
      }
    }

    m_ui.enumAdd(GUI_SUPERSAMPLE, 1, "none");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 2, "4x");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 540, "540p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 720, "720p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 1080, "1080p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 1440, "1440p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 2160, "2160p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 1024, "1024 sq");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 2048, "2048 sq");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 4096, "4096 sq");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_NONE, "Material");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_CLUSTER, "Clusters");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TESSELLATED_CLUSTER, "Tessellated clusters");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TESSELLATED_TRIANGLES, "Tessellated triangles");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TRIANGLES, "Triangles");
  }

  // Initialize core components

  m_profilerGpuTimer.init(m_profilerTimeline, app->getDevice(), app->getPhysicalDevice(), app->getQueue(0).familyIndex, true);
  m_resources.init(app->getDevice(), app->getPhysicalDevice(), app->getInstance(), app->getQueue(0));

#if USE_DLSS
  m_resources.m_dlssWaitForInit = m_resources.m_dlssWaitForInit || !m_dlssDelayLoad;

  // NGX init takes seconds, overlap it with scene loading
  if(m_rendererConfig.useDlss)
  {
    m_resources.initDlssAsync(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_TESS ? Resources::DlssMode::eRayReconstruction :
                                                                                    Resources::DlssMode::eSuperResolution,
                              m_rendererConfig.dlssQuality);
  }
#endif

  {
    NVVK_CHECK(m_resources.m_samplerPool.acquireSampler(m_imguiSampler));
    NVVK_DBG_NAME(m_imguiSampler);
  }

  m_resources.initFramebuffer({128, 128}, m_tweak.supersample, m_tweak.hbaoQuarter);
  updateImguiImage();

  updatedClusterConfig();

  // Search for default scene if none was provided on the command line
  if(m_sceneFilePath.empty())
  {
    const std::filesystem::path              exeDirectoryPath   = nvutils::getExecutablePath().parent_path();
    const std::vector<std::filesystem::path> defaultSearchPaths = {
        // regular build
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_DOWNLOAD_DIRECTORY),
        // install build
        std::filesystem::absolute(exeDirectoryPath / "resources"),
    };

    m_sceneFilePath = nvutils::findFile("bunny_v2/bunny.gltf", defaultSearchPaths);
    if(m_tweak.gridCopies == 1)
    {
      m_tweak.gridCopies = 121;  // 11x11 grid
    }
  }

  if(!m_sceneFilePath.empty())
  {
    if(initScene(m_sceneFilePath))
    {
      postInitNewScene();

      initRenderer(m_tweak.renderer);
    }
  }

  m_tweakLast          = m_tweak;
  m_sceneConfigLast    = m_sceneConfig;
  m_rendererConfigLast = m_rendererConfig;
}

void TessellatedClusters::onDetach()
{
#if USE_DLSS
  // NGX calls can't be interrupted, the first feature creation alone takes seconds
  if(m_resources.isDlssLoading())
  {
    LOGI("DLSS load still in progress on window close, exiting immediately\n");
    std::_Exit(0);
  }
#endif

  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  deinitRenderer();
  deinitScene();

  m_resources.m_samplerPool.releaseSampler(m_imguiSampler);
  ImGui_ImplVulkan_RemoveTexture(m_imguiTexture);

  m_resources.deinit();

  m_profilerGpuTimer.deinit();
}


void TessellatedClusters::onFileDrop(const std::filesystem::path& filePath)
{
  if(filePath.empty())
    return;

  // reset grid parameter (in case scene is too large to be replicated)
  m_tweak.gridCopies = 1;
  LOGI("Loading model: %s\n", nvutils::utf8FromPath(filePath).c_str());
  deinitRenderer();

  if(initScene(filePath))
  {
    postInitNewScene();
    initRenderer(m_tweak.renderer);
    m_tweakLast       = m_tweak;
    m_sceneConfigLast = m_sceneConfig;
  }
}

const TessellatedClusters::ClusterInfo TessellatedClusters::s_clusterInfos[NUM_CLUSTER_CONFIGS] = {
    {64, 64, CLUSTER_64T_64V},     {64, 128, CLUSTER_64T_128V},   {64, 192, CLUSTER_64T_192V},
    {96, 96, CLUSTER_96T_96V},     {128, 128, CLUSTER_128T_128V}, {128, 256, CLUSTER_128T_256V},
    {256, 256, CLUSTER_256T_256V},
};

void TessellatedClusters::adjustSceneClusterConfig()
{
#if 0
  m_scene->m_clusterTriangleHistogram.resize(m_scene->m_maxClusterTriangles + 1);
  m_scene->m_clusterVertexHistogram.resize(m_scene->m_maxClusterVertices + 1);

  // adjust the ui and settings based on actual data
  for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
  {
    const ClusterInfo& entry = s_clusterInfos[i];
    if(m_scene->m_maxClusterTriangles <= entry.tris && m_scene->m_maxClusterVertices <= entry.verts)
    {
      m_tweak.clusterConfig = entry.cfg;

      m_scene->m_config.clusterTriangles = entry.tris;
      m_sceneConfig.clusterTriangles     = entry.tris;
      m_sceneConfigLast.clusterTriangles = entry.tris;

      m_scene->m_config.clusterVertices = entry.verts;
      m_sceneConfig.clusterVertices     = entry.verts;
      m_sceneConfigLast.clusterVertices = entry.verts;
      return;
    }
  }
#endif
}

void TessellatedClusters::updatedClusterConfig()
{
  for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
  {
    if(s_clusterInfos[i].cfg == m_tweak.clusterConfig)
    {
      m_sceneConfig.clusterTriangles = s_clusterInfos[i].tris;
      m_sceneConfig.clusterVertices  = s_clusterInfos[i].verts;
      return;
    }
  }
}

void TessellatedClusters::handleChanges()
{
  if(m_tweak.clusterConfig != m_tweakLast.clusterConfig)
  {
    updatedClusterConfig();
  }

  if(!m_resources.m_supportsMeshShaderNV)
  {
    m_rendererConfig.useEXTmeshShader = true;
  }

  bool sceneChanged = false;
  if(memcmp(&m_sceneConfig, &m_sceneConfigLast, sizeof(m_sceneConfig)))
  {
    sceneChanged = true;

    deinitRenderer();
    initScene(m_sceneFilePath);
  }

  // Load the per-scene camera paths file once a (new) scene has finished loading.
  // Done here on the main thread (not in the loader thread's setSceneCamera) so
  // it does not race the UI / playback reads of m_cameraPaths. Explicitly provided
  // paths (--addcamerapath/--loadcamerapaths, via command line, config, or sequence)
  // are global and take precedence, so the per-scene file is only used without them.
  if(m_scene && m_sceneFilePath != m_cameraPathsLoadedScene)
  {
    m_cameraPathsLoadedScene = m_sceneFilePath;

    if(!m_cameraPathsExternal)
    {
      m_cameraPaths.clear();
      m_cameraPathActive   = -1;
      m_cameraPathEditKey  = -1;
      m_cameraPathPlayback = CAMERA_PATH_STOP;

      std::filesystem::path pathsFile = sceneCameraPathsFile(m_sceneFilePath);
      if(std::filesystem::exists(pathsFile) && loadCameraPaths(pathsFile, m_cameraPaths))
      {
        m_cameraPathActive = m_cameraPaths.empty() ? -1 : 0;
        LOGI("Loaded %zu camera path(s) from %s\n", m_cameraPaths.size(), nvutils::utf8FromPath(pathsFile).c_str());
      }
    }
  }

  bool shaderChanged = false;
  if(m_reloadShaders)
  {
    shaderChanged   = true;
    m_reloadShaders = false;
  }

  bool frameBufferChanged = false;
  // a change of only the HBAO resolution leaves the change id alone
  const bool supersampleChanged = tweakChanged(m_tweak.supersample);
  const bool hbaoQuarterChanged = tweakChanged(m_tweak.hbaoQuarter);
  if(supersampleChanged || hbaoQuarterChanged)
  {
    const size_t fboChangeID = m_resources.m_fboChangeID;
    m_resources.initFramebuffer(m_windowSize, m_tweak.supersample, m_tweak.hbaoQuarter);
    if(fboChangeID != m_resources.m_fboChangeID)
    {
      updateImguiImage();
      frameBufferChanged = true;
    }
  }

  bool dlssLoaded = false;
#if USE_DLSS
  // switch to DLSS once its async init finished
  dlssLoaded = m_resources.pollDlssInit() && m_rendererConfig.useDlss;

  if(m_resources.m_dlssEvaluateFailed)
  {
    // re-initializes the renderer natively
    m_rendererConfig.useDlss         = false;
    m_resources.m_dlssEvaluateFailed = false;
  }
#endif

  bool rendererChanged = false;
  if(sceneChanged || shaderChanged || tweakChanged(m_tweak.renderer) || tweakChanged(m_tweak.gridCopies)
     || tweakChanged(m_tweak.gridConfig) || tweakChanged(m_tweak.clusterBuildMode)
     || rendererCfgChanged(m_rendererConfig.flipWinding) || rendererCfgChanged(m_rendererConfig.doAnimation)
     || rendererCfgChanged(m_rendererConfig.doCulling) || rendererCfgChanged(m_rendererConfig.persistentThreads)
     || rendererCfgChanged(m_rendererConfig.persistentKernel) || rendererCfgChanged(m_rendererConfig.positionTruncateBits)
     || rendererCfgChanged(m_rendererConfig.pnDisplacement) || rendererCfgChanged(m_rendererConfig.numVisibleClusterBits)
     || rendererCfgChanged(m_rendererConfig.numSplitTriangleBits) || rendererCfgChanged(m_rendererConfig.numGeneratedVerticesBits)
     || rendererCfgChanged(m_rendererConfig.numPartTriangleBits) || rendererCfgChanged(m_rendererConfig.numGeneratedClusterMegs)
     || rendererCfgChanged(m_rendererConfig.lowTessClusters1X) || rendererCfgChanged(m_rendererConfig.lowTessClusters2X)
     || rendererCfgChanged(m_rendererConfig.lowTessMode) || rendererCfgChanged(m_rendererConfig.transient2xBatchSize)
     || rendererCfgChanged(m_rendererConfig.combined1xBatchSize) || rendererCfgChanged(m_rendererConfig.combined2xBatchSize)
     || rendererCfgChanged(m_rendererConfig.rasterBatchMeshlets) || rendererCfgChanged(m_rendererConfig.useEXTmeshShader)
     || rendererCfgChanged(m_rendererConfig.splitFactor) || rendererCfgChanged(m_rendererConfig.debugVisualization)
     || rendererCfgChanged(m_rendererConfig.generateHistogram) || rendererCfgChanged(m_rendererConfig.useRenderStats)
     || rendererCfgChanged(m_rendererConfig.useDlss) || rendererCfgChanged(m_rendererConfig.dlssQuality) || dlssLoaded)
  {
    rendererChanged = true;

    initRenderer(m_tweak.renderer);

    if(shaderChanged)
    {
      m_resources.m_hbaoPass.reloadShaders();
    }
  }
  else if(m_renderer && frameBufferChanged)
  {
    m_renderer->updatedFrameBuffer(m_resources);
  }

  // renderer init may re-create the framebuffer, e.g. for DLSS
  if(m_imguiFboChangeID != m_resources.m_fboChangeID)
  {
    updateImguiImage();
  }

  bool hadChange = memcmp(&m_tweakLast, &m_tweak, sizeof(m_tweak))
                   || memcmp(&m_rendererConfigLast, &m_rendererConfig, sizeof(m_rendererConfig))
                   || memcmp(&m_sceneConfigLast, &m_sceneConfig, sizeof(m_sceneConfig));

  m_tweakLast          = m_tweak;
  m_rendererConfigLast = m_rendererConfig;
  m_sceneConfigLast    = m_sceneConfig;

  if(hadChange && m_tweak.autoResetTimers)
  {
    m_info.profilerManager->resetFrameSections(8);
  }
}

void TessellatedClusters::registerCameraPathParameters()
{
  // `--addcamerapath "<string>"` appends a path definition; the index is the
  // order in which the paths were added. Repeatable.
  m_info.parameterRegistry->addCustom(
      {"addcamerapath", "append a camera path definition string (repeatable); the path index equals the order added"}, 1,
      [this](const nvutils::ParameterBase* const, std::span<char const* const> args, const std::filesystem::path&) -> bool {
        CameraPath path;
        if(!path.setFromString(args[0]))
        {
          LOGE("addcamerapath: could not parse camera path string\n");
          return false;
        }
        if(!m_cameraPathsExternal)
        {
          // the first explicit path replaces the ones loaded with the scene, so the index is the order added
          m_cameraPaths.clear();
          m_cameraPathActive   = -1;
          m_cameraPathEditKey  = -1;
          m_cameraPathPlayback = CAMERA_PATH_STOP;
          m_cameraPathStarted  = false;
          m_cameraPathU        = 0.0;
        }
        m_cameraPaths.push_back(std::move(path));
        m_cameraPathsExternal = true;
        if(m_cameraPathActive < 0)
          m_cameraPathActive = 0;
        LOGI("addcamerapath: added path %d with %zu keyframes\n", int(m_cameraPaths.size()) - 1, m_cameraPaths.back().size());
        return true;
      });

  // `--loadcamerapaths "<filename>"` replaces all path definitions with the
  // paths from a simple text file (one path per line). Full overwrite.
  m_info.parameterRegistry->addCustom(
      {"loadcamerapaths", "load all camera path definitions from a text file (full overwrite)"}, 1,
      [this](const nvutils::ParameterBase* const, std::span<char const* const> args, const std::filesystem::path& basePath) -> bool {
        std::filesystem::path file = std::filesystem::path(args[0]);
        if(file.is_relative() && !basePath.empty())
          file = basePath / file;
        if(!loadCameraPaths(file, m_cameraPaths))
        {
          LOGE("loadcamerapaths: could not open %s\n", nvutils::utf8FromPath(file).c_str());
          return false;
        }
        m_cameraPathActive    = m_cameraPaths.empty() ? -1 : 0;
        m_cameraPathPlayback  = CAMERA_PATH_STOP;
        m_cameraPathsExternal = true;
        LOGI("loadcamerapaths: loaded %zu path(s) from %s\n", m_cameraPaths.size(), nvutils::utf8FromPath(file).c_str());
        return true;
      });

  // `--runcamerapath <index> <framecount>` plays back a path over a fixed number
  // of frames (deterministic), one keyframe-step per rendered frame. Meant to be
  // used within benchmark sequences.
  m_info.parameterRegistry->addCustom(
      {"runcamerapath", "deterministic fixed-step playback: <index> <framecount>, spreads the given path across framecount frames"},
      2, [this](const nvutils::ParameterBase* const, std::span<char const* const> args, const std::filesystem::path&) -> bool {
        int index      = atoi(args[0]);
        int frameCount = atoi(args[1]);
        if(index < 0 || index >= int(m_cameraPaths.size()))
        {
          LOGE("runcamerapath: path index %d out of range (%d paths defined)\n", index, int(m_cameraPaths.size()));
          return false;
        }
        m_cameraPathActive   = index;
        m_cameraPathFrames   = std::max(frameCount, 1);
        m_cameraPathFrame    = 0;
        m_cameraPathPlayback = CAMERA_PATH_FIXED;
        return true;
      });
}

std::filesystem::path TessellatedClusters::sceneCameraPathsFile(const std::filesystem::path& scenePath) const
{
  if(scenePath.empty())
    return {};

  // stored next to the model file, named after it, e.g. ".../bunny.camerapaths.txt"
  std::filesystem::path file = scenePath;
  file.replace_extension(".camerapaths.txt");
  return file;
}

bool TessellatedClusters::applyCameraPathSample(double u)
{
  if(m_cameraPathActive < 0 || m_cameraPathActive >= int(m_cameraPaths.size()))
    return false;

  CameraPathKey key;
  if(!m_cameraPaths[m_cameraPathActive].sample(u, key))
    return false;

  // keep the manipulator's other camera state (clip planes, projection, ...)
  nvutils::CameraManipulator::Camera cam = m_info.cameraManipulator->getCamera();
  cam.eye                                = key.eye;
  cam.ctr                                = key.ctr;
  cam.up                                 = key.up;
  cam.fov                                = key.fov;
  m_info.cameraManipulator->setCamera(cam, true);
  return true;
}

void TessellatedClusters::updateCameraPath(double time)
{
  if(m_cameraPathPlayback == CAMERA_PATH_STOP)
    return;

  if(m_cameraPathActive < 0 || m_cameraPathActive >= int(m_cameraPaths.size()) || m_cameraPaths[m_cameraPathActive].empty())
  {
    m_cameraPathPlayback = CAMERA_PATH_STOP;
    return;
  }

  const CameraPath& path = m_cameraPaths[m_cameraPathActive];

  if(m_cameraPathPlayback == CAMERA_PATH_FIXED)
  {
    // deterministic: frame f maps to u = f / (frames - 1)
    int    denom = std::max(m_cameraPathFrames - 1, 1);
    double u     = double(std::min(m_cameraPathFrame, denom)) / double(denom);
    applyCameraPathSample(u);
    m_cameraPathU = u;

    // advance for the next frame, hold at the end (the sequence/headless run
    // stops on its own frame count)
    if(m_cameraPathFrame < m_cameraPathFrames - 1)
      m_cameraPathFrame++;
  }
  else  // CAMERA_PATH_REALTIME
  {
    double dt           = m_cameraPathStarted ? std::max(time - m_lastTime, 0.0) : 0.0;
    m_cameraPathStarted = true;
    if(path.duration > 0.0)
      m_cameraPathU += dt / path.duration;

    if(m_cameraPathU >= 1.0)
    {
      if(path.loop)
        m_cameraPathU = std::fmod(m_cameraPathU, 1.0);
      else
      {
        m_cameraPathU        = 1.0;
        m_cameraPathPlayback = CAMERA_PATH_STOP;
      }
    }

    // sample the advanced position, so a non-looping path renders its last key before stopping
    applyCameraPathSample(m_cameraPathU);
  }
}

void TessellatedClusters::onRender(VkCommandBuffer cmd)
{
  double time = m_clock.getSeconds();

  // drive the camera along the active path (overrides the manipulator for the
  // rendered frame); runs after nvapp::ElementCamera's per-frame input update
  updateCameraPath(time);

  m_resources.beginFrame(m_app->getFrameCycleIndex());

  m_frameConfig.windowSize = m_windowSize;
  m_frameConfig.hbaoActive = false;

  if(m_renderer)
  {
    uint32_t width  = m_windowSize.width;
    uint32_t height = m_windowSize.height;

    if(m_rendererFboChangeID != m_resources.m_fboChangeID)
    {
      m_renderer->updatedFrameBuffer(m_resources);
      m_rendererFboChangeID = m_resources.m_fboChangeID;
    }

    m_frameConfig.hbaoActive = m_tweak.hbaoActive && m_tweak.renderer == RENDERER_RASTER_CLUSTERS_TESS;

    bool rasterDlssSrActive = false;
#if USE_DLSS
    rasterDlssSrActive = m_tweak.renderer == RENDERER_RASTER_CLUSTERS_TESS
                         && m_resources.m_frameBuffer.dlssMode == Resources::DlssMode::eSuperResolution;
#endif

    shaderio::FrameConstants& frameConstants = m_frameConfig.frameConstants;

    // for motion always use last
    frameConstants.viewProjMatrixPrev = frameConstants.viewProjMatrix;

    if(m_frames && !m_frameConfig.freezeCulling)
    {
      m_frameConfig.frameConstantsLast      = m_frameConfig.frameConstants;
      m_frameConfig.frameConstantsLast.time = 0;
    }

    int supersample = m_tweak.supersample;

    // may be lower than window size times supersample due to DLSS
    uint32_t renderWidth  = m_resources.m_frameBuffer.renderSize.width;
    uint32_t renderHeight = m_resources.m_frameBuffer.renderSize.height;
    frameConstants.time   = float(time);
    frameConstants.frame  = m_frames;

    frameConstants.facetShading = m_tweak.facetShading ? 1 : 0;
    frameConstants.doAnimation  = m_rendererConfig.doAnimation ? 1 : 0;
    frameConstants.visualize    = m_tweak.visualizeMode;

    // the scene setup provides a default, the command-line can override it
    if(m_aoSamplesOverride >= 0)
    {
      frameConstants.ambientOcclusionSamples = m_aoSamplesOverride;
    }

    {
      frameConstants.visFilterClusterID  = ~0;
      frameConstants.visFilterInstanceID = ~0;
    }
    // for motion vectors of the animated surface
    m_frameConfig.frameConstants.animationStatePrev = m_frameConfig.frameConstants.animationState;
    if(m_rendererConfig.doAnimation)
    {
      if(m_tweak.overrideTime)
      {
        m_frameConfig.frameConstants.animationState = m_tweak.overrideTime;
        m_animTime                                  = m_tweak.overrideTime;
      }
      else
      {
        m_animTime += (time - m_lastTime) * 0.5;
        m_frameConfig.frameConstants.animationState = float(m_animTime);
      }
    }
    frameConstants.bgColor     = m_resources.m_bgColor;
    frameConstants.flipWinding = m_rendererConfig.flipWinding ? 1 : 0;

    frameConstants.viewport    = glm::ivec2(renderWidth, renderHeight);
    frameConstants.viewportf   = glm::vec2(renderWidth, renderHeight);
    frameConstants.supersample = m_tweak.supersample;
    frameConstants.nearPlane   = float(m_info.cameraManipulator->getClipPlanes().x);
    frameConstants.farPlane    = float(m_info.cameraManipulator->getClipPlanes().y);
    frameConstants.wUpDir      = glm::vec3(m_info.cameraManipulator->getUp());
#if USE_DLSS
    frameConstants.jitter = shaderio::dlssJitter(m_frames);
#endif

    // fixed resolutions may not match the window's aspect ratio
    float aspect = float(m_resources.m_frameBuffer.targetSize.width) / float(m_resources.m_frameBuffer.targetSize.height);

    // reversed-Z: near plane at depth 1, far plane at 0
    glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(float(m_info.cameraManipulator->getFov())), aspect,
                                                 frameConstants.farPlane, frameConstants.nearPlane);
    projection[1][1] *= -1;

    // rasterization applies the jitter through the projection, ray tracing offsets the ray within the pixel
    glm::mat4 projectionRender = projection;
    if(rasterDlssSrActive)
    {
      glm::mat4 jitterMatrix(1.0f);
      jitterMatrix[3][0] = -frameConstants.jitter.x * 2.0f / std::max(frameConstants.viewportf.x, 1.0f);
      jitterMatrix[3][1] = -frameConstants.jitter.y * 2.0f / std::max(frameConstants.viewportf.y, 1.0f);
      projectionRender   = jitterMatrix * projection;
    }

    glm::mat4 view  = m_info.cameraManipulator->getViewMatrix();
    glm::mat4 viewI = glm::inverse(view);

    frameConstants.viewProjMatrixRender = projectionRender * view;
    frameConstants.viewProjMatrix       = projection * view;
    frameConstants.viewProjMatrixI      = glm::inverse(frameConstants.viewProjMatrix);
    frameConstants.viewMatrix           = view;
    frameConstants.viewMatrixI          = viewI;
    frameConstants.projMatrix           = projection;
    frameConstants.projMatrixI          = glm::inverse(projection);

    glm::mat4 viewNoTrans         = view;
    viewNoTrans[3]                = {0.0f, 0.0f, 0.0f, 1.0f};
    frameConstants.skyProjMatrixI = glm::inverse(projection * viewNoTrans);

    frameConstants.tessRate = m_tweak.tessRatePixels ? 1.0f / float(m_tweak.tessRatePixels) : 0.0f;
    // the rate is in pixels of the target, DLSS renders at lower resolution
    frameConstants.tessRate *= float(m_resources.m_frameBuffer.targetSize.height) / float(renderHeight);

    glm::vec4 hPos   = projection * glm::vec4(1.0f, 1.0f, -frameConstants.farPlane, 1.0f);
    glm::vec2 hCoord = glm::vec2(hPos.x / hPos.w, hPos.y / hPos.w);
    glm::vec2 dim    = glm::abs(hCoord);

    // helper to quickly get footprint of a point at a given distance
    //
    // __.__hPos (far plane is width x height)
    // \ | /
    //  \|/
    //   x camera
    //
    // here: viewPixelSize / point.w = size of point in pixels
    // * 0.5f because renderWidth/renderHeight represents [-1,1] but we need half of frustum
    frameConstants.viewPixelSize = dim * (glm::vec2(float(renderWidth), float(renderHeight)) * 0.5f) * frameConstants.farPlane;
    // here: viewClipSize / point.w = size of point in clip-space units
    // no extra scale as half clip space is 1.0 in extent
    frameConstants.viewClipSize = dim * frameConstants.farPlane;

    frameConstants.viewPos = frameConstants.viewMatrixI[3];  // position of eye in the world
    frameConstants.viewDir = -viewI[2];

    frameConstants.viewPlane   = frameConstants.viewDir;
    frameConstants.viewPlane.w = -glm::dot(glm::vec3(frameConstants.viewPos), glm::vec3(frameConstants.viewDir));

    frameConstants.wLightPos = frameConstants.viewMatrixI[3];  // place light at position of eye in the world

    {
      // hiz
      m_resources.m_hizUpdate.farInfo.getShaderFactors((float*)&frameConstants.hizSizeFactors);
      frameConstants.hizSizeMax = m_resources.m_hizUpdate.farInfo.getSizeMax();
    }

    {
      // hbao setup
      auto& hbaoView                    = m_frameConfig.hbaoSettings.view;
      hbaoView.farPlane                 = frameConstants.farPlane;
      hbaoView.nearPlane                = frameConstants.nearPlane;
      hbaoView.projectionMatrix         = projection;
      hbaoView.viewMatrix               = view;
      hbaoView.tanFovy                  = tanf(glm::radians(float(m_info.cameraManipulator->getFov())) * 0.5f);
      m_frameConfig.hbaoSettings.radius = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo) * m_tweak.hbaoRadius;
    }

    if(!m_frames)
    {
      // no motion on first frame
      frameConstants.viewProjMatrixPrev = frameConstants.viewProjMatrix;

      m_frameConfig.frameConstantsLast      = m_frameConfig.frameConstants;
      m_frameConfig.frameConstantsLast.time = 0;
    }

    if(m_frames)
    {
      shaderio::FrameConstants frameCurrent = m_frameConfig.frameConstants;
      frameCurrent.time                     = 0;
    }

    m_renderer->render(cmd, m_resources, *m_scene, m_frameConfig, m_profilerGpuTimer);
  }
  else
  {
    m_resources.emptyFrame(cmd, m_frameConfig, m_profilerGpuTimer);
  }

  {
    m_resources.postProcessFrame(cmd, m_frameConfig, m_profilerGpuTimer);
  }

  m_resources.endFrame();

  m_lastTime = time;
  m_frames++;
}

void TessellatedClusters::setSceneCamera(const std::filesystem::path& filePath)
{
  nvgui::SetCameraJsonFile(filePath);

  float     radius = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo) * 0.5f;
  glm::vec3 center = (m_scene->m_bbox.hi + m_scene->m_bbox.lo) * 0.5f;

  if(!m_scene->m_cameras.empty())
  {
    auto& c = m_scene->m_cameras[0];
    m_info.cameraManipulator->setFov(c.fovy);


    c.eye              = glm::vec3(c.worldMatrix[3]);
    float     distance = glm::length(center - c.eye);
    glm::mat3 rotMat   = glm::mat3(c.worldMatrix);
    c.center           = {0, 0, -distance};
    c.center           = c.eye + (rotMat * c.center);
    c.up               = {0, 1, 0};

    m_info.cameraManipulator->setCamera({c.eye, c.center, c.up, static_cast<float>(glm::degrees(c.fovy))});

    nvgui::SetHomeCamera({c.eye, c.center, c.up, static_cast<float>(glm::degrees(c.fovy))});
    for(auto& cam : m_scene->m_cameras)
    {
      cam.eye            = glm::vec3(cam.worldMatrix[3]);
      float     distance = glm::length(center - cam.eye);
      glm::mat3 rotMat   = glm::mat3(cam.worldMatrix);
      cam.center         = {0, 0, -distance};
      cam.center         = cam.eye + (rotMat * cam.center);
      cam.up             = {0, 1, 0};


      nvgui::AddCamera({cam.eye, cam.center, cam.up, static_cast<float>(glm::degrees(cam.fovy))});
    }
  }
  else
  {
    // Re-adjusting camera to fit the new scene
    m_info.cameraManipulator->fit(m_scene->m_bbox.lo, m_scene->m_bbox.hi, true);
    nvgui::SetHomeCamera(m_info.cameraManipulator->getCamera());
  }

  m_info.cameraManipulator->setClipPlanes(glm::vec2(0.01F * radius, 100.0F * radius));
}

float TessellatedClusters::decodePickingDepth(const shaderio::Readback& readback)
{
  if(!isPickingValid(readback))
  {
    return 0.f;
  }
  uint32_t bits = readback._packedDepth0;
  bits ^= ~(int(bits) >> 31) | 0x80000000u;
  float res = *(float*)&bits;
  // reversed-Z depth as written by the shaders (1 = near plane, 0 = far plane)
  return res;
}

bool TessellatedClusters::isPickingValid(const shaderio::Readback& readback)
{
  return readback._packedDepth0 != 0u;
}

}  // namespace tessellatedclusters
