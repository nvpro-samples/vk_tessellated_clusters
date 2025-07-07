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

#include <fmt/format.h>
#include <nvutils/file_operations.hpp>
#include <nvgui/camera.hpp>

#include "tessellatedclusters.hpp"

bool g_verbose = false;

namespace tessellatedclusters {

TessellatedClusters::TessellatedClusters(const Info& info)
    : m_info(info)
{
  nvutils::ProfilerTimeline::CreateInfo createInfo;
  createInfo.name = "graphics";

  m_profilerTimeline = m_info.profilerManager->createTimeline(createInfo);

  m_info.parameterRegistry->add({"scene"}, {".gltf", ".glb"}, &m_sceneFilePath);
  m_info.parameterRegistry->add({"renderer"}, (int*)&m_tweak.renderer);
  m_info.parameterRegistry->add({"verbose"}, &g_verbose, true);
  m_info.parameterRegistry->add({"resetstats"}, &m_tweak.autoResetTimers);
  m_info.parameterRegistry->add({"supersample"}, &m_tweak.supersample);
  m_info.parameterRegistry->add({"animation"}, &m_rendererConfig.doAnimation);
  m_info.parameterRegistry->add({"culling"}, &m_rendererConfig.doCulling);
  m_info.parameterRegistry->add({"tessrate"}, &m_tweak.tessRatePixels);
  m_info.parameterRegistry->add({"gridcopies"}, &m_tweak.gridCopies);
  m_info.parameterRegistry->add({"gridconfig"}, &m_tweak.gridConfig);
  m_info.parameterRegistry->add({"clusterconfig"}, (int*)&m_tweak.clusterConfig);
  m_info.parameterRegistry->add({"nvcluster"}, &m_sceneConfig.clusterNvLibrary);
  //m_info.parameterRegistry->add({"nvclusterunderfill"}, &m_sceneConfig.clusterNvConfig.costUnderfill);
  //m_info.parameterRegistry->add({"nvclusteroverlap"}, &m_sceneConfig.clusterNvConfig.costOverlap);
  //m_info.parameterRegistry->add({"nvclusterunderfillvertices"}, &m_sceneConfig.clusterNvConfig.costUnderfillVertices);
  m_info.parameterRegistry->add({"overridetime"}, &m_tweak.overrideTime);
  m_info.parameterRegistry->add({"processingthreadpct", "float percentage of threads during initial file load and processing into lod clusters, default 0.5 == 50 %"},
                                &m_sceneConfig.processingThreadsPct);

  m_frameConfig.frameConstants                         = {};
  m_frameConfig.frameConstants.ambientOcclusionSamples = 1;
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
  m_frameConfig.frameConstants.ambientOcclusionSamples = 16;

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
  m_resources.initFramebuffer(m_windowSize, m_tweak.supersample, m_tweak.hbaoFullRes);
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

  m_imguiTexture = ImGui_ImplVulkan_AddTexture(m_imguiSampler, imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void TessellatedClusters::onPreRender()
{
  m_profilerTimeline->frameAdvance();
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
  m_info.cameraManipulator->setSceneSize(sceneDimension);

  m_frameConfig.frameConstants.animationRippleEnabled   = 1;
  m_frameConfig.frameConstants.animationRippleFrequency = 50.f;
  m_frameConfig.frameConstants.animationRippleAmplitude = 0.004f;
  m_frameConfig.frameConstants.animationRippleSpeed     = 5.f;
  m_frameConfig.frameConstants.doShadow                 = 1;
  m_frameConfig.frameConstants.ambientOcclusionRadius   = 0.1f;
  m_frameConfig.frameConstants.ambientOcclusionSamples  = 16;
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
    {
      for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
      {
        std::string enumStr = fmt::format("{}T_{}V", s_clusterInfos[i].tris, s_clusterInfos[i].verts);
        m_ui.enumAdd(GUI_MESHLET, s_clusterInfos[i].cfg, enumStr.c_str());
      }
    }

    m_ui.enumAdd(GUI_SUPERSAMPLE, 1, "none");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 2, "4x");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_NONE, "Material");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_CLUSTER, "Clusters");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TESSELLATED_CLUSTER, "Tessellated clusters");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TESSELLATED_TRIANGLES, "Tessellated triangles");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TRIANGLES, "Triangles");
  }

  // Initialize core components

  m_profilerGpuTimer.init(m_profilerTimeline, app->getDevice(), app->getPhysicalDevice(), app->getQueue(0).familyIndex, true);
  m_resources.init(app->getDevice(), app->getPhysicalDevice(), app->getInstance(), app->getQueue(0));

  {
    NVVK_CHECK(m_resources.m_samplerPool.acquireSampler(m_imguiSampler));
    NVVK_DBG_NAME(m_imguiSampler);
  }

  m_resources.initFramebuffer({128, 128}, m_tweak.supersample, m_tweak.hbaoFullRes);
  updateImguiImage();

  updatedClusterConfig();

  // Search for default scene if none was provided on the command line
  if(m_sceneFilePath.empty())
  {
    const std::filesystem::path              exeDirectoryPath   = nvutils::getExecutablePath().parent_path();
    const std::vector<std::filesystem::path> defaultSearchPaths = {
        // regular build
        std::filesystem::absolute(exeDirectoryPath / PROJECT_EXE_TO_DOWNLOAD_DIRECTORY),
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
    {32, 32, CLUSTER_32T_32V},     {64, 64, CLUSTER_64T_64V},     {64, 128, CLUSTER_64T_128V},
    {64, 192, CLUSTER_64T_192V},   {96, 96, CLUSTER_96T_96V},     {128, 128, CLUSTER_128T_128V},
    {128, 256, CLUSTER_128T_256V}, {256, 256, CLUSTER_256T_256V},
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

  bool sceneChanged = false;
  if(memcmp(&m_sceneConfig, &m_sceneConfigLast, sizeof(m_sceneConfig)))
  {
    sceneChanged = true;

    deinitRenderer();
    initScene(m_sceneFilePath);
  }

  bool shaderChanged = false;
  if(m_reloadShaders)
  {
    shaderChanged   = true;
    m_reloadShaders = false;
  }

  bool frameBufferChanged = false;
  if(tweakChanged(m_tweak.supersample) || tweakChanged(m_tweak.hbaoFullRes))
  {
    m_resources.initFramebuffer(m_windowSize, m_tweak.supersample, m_tweak.hbaoFullRes);
    updateImguiImage();

    frameBufferChanged = true;
  }

  bool rendererChanged = false;
  if(sceneChanged || shaderChanged || tweakChanged(m_tweak.renderer) || tweakChanged(m_tweak.gridCopies)
     || tweakChanged(m_tweak.gridConfig) || rendererCfgChanged(m_rendererConfig.flipWinding)
     || rendererCfgChanged(m_rendererConfig.doAnimation) || rendererCfgChanged(m_rendererConfig.doCulling)
     || rendererCfgChanged(m_rendererConfig.persistentThreads)
     || (m_renderer
         && (rendererCfgChanged(m_rendererConfig.positionTruncateBits) || rendererCfgChanged(m_rendererConfig.pnDisplacement)
             || rendererCfgChanged(m_rendererConfig.numVisibleClusterBits) || rendererCfgChanged(m_rendererConfig.numSplitTriangleBits)
             || rendererCfgChanged(m_rendererConfig.numGeneratedVerticesBits)
             || rendererCfgChanged(m_rendererConfig.numPartTriangleBits) || rendererCfgChanged(m_rendererConfig.numGeneratedClusterMegs)
             || rendererCfgChanged(m_rendererConfig.transientClusters1X) || rendererCfgChanged(m_rendererConfig.transientClusters2X)
             || tweakChanged(m_tweak.clusterBuildMode) || rendererCfgChanged(m_rendererConfig.rasterBatchMeshlets)))
     || rendererCfgChanged(m_rendererConfig.debugVisualization))
  {
    rendererChanged = true;

    initRenderer(m_tweak.renderer);
  }
  else if(m_renderer && frameBufferChanged)
  {
    m_renderer->updatedFrameBuffer(m_resources);
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

void TessellatedClusters::onRender(VkCommandBuffer cmd)
{
  double time = m_clock.getSeconds();

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

    shaderio::FrameConstants& frameConstants = m_frameConfig.frameConstants;
    if(m_frames && !m_frameConfig.freezeCulling)
    {
      m_frameConfig.frameConstantsLast      = m_frameConfig.frameConstants;
      m_frameConfig.frameConstantsLast.time = 0;
    }

    int supersample = m_tweak.supersample;

    uint32_t renderWidth  = width * m_tweak.supersample;
    uint32_t renderHeight = height * m_tweak.supersample;
    frameConstants.time   = float(time);

    frameConstants.facetShading = m_tweak.facetShading ? 1 : 0;
    frameConstants.doAnimation  = m_rendererConfig.doAnimation ? 1 : 0;
    frameConstants.visualize    = m_tweak.visualizeMode;

    {
      frameConstants.visFilterClusterID  = ~0;
      frameConstants.visFilterInstanceID = ~0;
    }
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
    frameConstants.nearPlane   = m_info.cameraManipulator->getClipPlanes().x;
    frameConstants.farPlane    = m_info.cameraManipulator->getClipPlanes().y;
    frameConstants.wUpDir      = m_info.cameraManipulator->getUp();

    glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(m_info.cameraManipulator->getFov()), float(width) / float(height),
                                                 frameConstants.nearPlane, frameConstants.farPlane);
    projection[1][1] *= -1;

    glm::mat4 view  = m_info.cameraManipulator->getViewMatrix();
    glm::mat4 viewI = glm::inverse(view);

    frameConstants.viewProjMatrix  = projection * view;
    frameConstants.viewProjMatrixI = glm::inverse(frameConstants.viewProjMatrix);
    frameConstants.viewMatrix      = view;
    frameConstants.viewMatrixI     = viewI;
    frameConstants.projMatrix      = projection;
    frameConstants.projMatrixI     = glm::inverse(projection);

    glm::mat4 viewNoTrans         = view;
    viewNoTrans[3]                = {0.0f, 0.0f, 0.0f, 1.0f};
    frameConstants.skyProjMatrixI = glm::inverse(projection * viewNoTrans);

    frameConstants.tessRate = m_tweak.tessRatePixels ? 1.0f / float(m_tweak.tessRatePixels) : 0.0f;

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
      hbaoView.isOrtho                  = false;
      hbaoView.projectionMatrix         = projection;
      m_frameConfig.hbaoSettings.radius = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo) * m_tweak.hbaoRadius;

      glm::vec4 hi = frameConstants.projMatrixI * glm::vec4(1, 1, -0.9, 1);
      hi /= hi.w;
      float tanx           = hi.x / fabsf(hi.z);
      float tany           = hi.y / fabsf(hi.z);
      hbaoView.halfFovyTan = tany;
    }

    if(!m_frames)
    {
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
  return 1.f - res;
}

bool TessellatedClusters::isPickingValid(const shaderio::Readback& readback)
{
  return readback._packedDepth0 != 0u;
}

}  // namespace tessellatedclusters
