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

#include <filesystem>

#include <imgui/backends/imgui_vk_extra.h>
#include <imgui/imgui_camera_widget.h>
#include <imgui/imgui_orient.h>
#include <implot.h>
#include <nvh/fileoperations.hpp>
#include <nvh/misc.hpp>
#include <nvh/cameramanipulator.hpp>
#include <nvvkhl/shaders/dh_sky.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>

#include "tessellatedclusters.hpp"
#include "vk_nv_cluster_acc.h"

bool g_verbose = false;

namespace tessellatedclusters {

std::string TessellatedClusters::getShaderPrepend()
{
  std::string prepend = m_customShaderPrepend;
  if(!prepend.empty())
  {
    char* test  = &prepend[0];
    char* found = nullptr;
    while((found = strstr(test, "\\n")) != nullptr)
    {
      found[0] = ' ';
      found[1] = '\n';
      test += 2;
    }
  }


  prepend += nvh::stringFormat("#define DEBUG_VISUALIZATION %d\n", m_tweak.useDebugVisualization ? 1 : 0);
  prepend += nvh::stringFormat("#define DO_ANIMATION %d\n", m_tweak.doAnimation ? 1 : 0);
  prepend += nvh::stringFormat("#define DO_CULLING %d\n", m_tweak.doCulling ? 1 : 0);

#ifdef _DEBUG
  printf(prepend.c_str());
#endif

  return prepend;
}

bool TessellatedClusters::initScene(const char* filename)
{
  deinitScene();

  if(filename)
  {
    LOGI("Loading scene %s\n", filename);

    m_scene = std::make_unique<Scene>();
    if(!m_scene->init(filename, m_resources, m_sceneConfig))
    {
      m_scene = nullptr;
      LOGW("Loading scene failed\n");
    }
    else
    {
      // a scene may come with loaded config, that can differ from what user asked for
      m_lastSceneConfig = m_sceneConfig = m_scene->m_config;
      m_tweak.clusterConfig = m_lastTweak.clusterConfig = getClusterConfig(m_scene->m_config);
    }
    return m_scene != nullptr;
  }

  return true;
}

void TessellatedClusters::deinitScene()
{
  if(m_scene)
  {
    m_scene->deinit(m_resources);
  }
}

bool TessellatedClusters::initFramebuffers(int width, int height)
{
  bool result = m_resources.initFramebuffer(width, height, m_tweak.supersample, m_tweak.hbaoFullRes);
  updateTargetImage();

  return result;
}

void TessellatedClusters::updateTargetImage()
{
  if(m_resources.m_framebuffer.useResolved)
  {
    m_targetImage.image = m_resources.m_framebuffer.imgColorResolved.image;
    m_targetImage.view  = m_resources.m_framebuffer.viewColorResolved;
  }
  else
  {
    m_targetImage.image = m_resources.m_framebuffer.imgColor.image;
    m_targetImage.view  = m_resources.m_framebuffer.viewColor;
  }
}

void TessellatedClusters::deinitRenderer()
{
  if(m_renderer)
  {
    m_resources.synchronize("sync deinitRenderer");
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

  m_rendererConfig.flipWinding       = m_tweak.flipWinding;
  m_rendererConfig.clusterBuildFlags = getBuildFlags(m_tweak.clusterBuildMode);

  m_resources.m_shaderManager.m_prepend = getShaderPrepend();
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

  setCameraFromScene(m_modelFilename.c_str());

  float sceneDimension                   = glm::length((m_scene->m_bbox.hi - m_scene->m_bbox.lo));
  m_frameConfig.frameConstants.wLightPos = (m_scene->m_bbox.hi + m_scene->m_bbox.lo) * 0.5f + sceneDimension;
  m_frameConfig.frameConstants.sceneSize = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo);

  m_frames = 0;
}

bool TessellatedClusters::initCore(nvvk::Context& context, int winWidth, int winHeight, const std::string& exePath)
{
  m_clusterSupport = context.hasDeviceExtension(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME)
                     && load_VK_NV_cluster_accleration_structure(context.m_instance, context.m_device);

  context.ignoreDebugMessage(0x8c548bfd);  // rayquery missing
  context.ignoreDebugMessage(0x901f59ec);  // unknown enum
  context.ignoreDebugMessage(0x79de34d4);  // unsupported ext
  context.ignoreDebugMessage(0x23e43bb7);  // attachment not written
  context.ignoreDebugMessage(0x6bbb14);    // spir-v location mesh shader
  context.ignoreDebugMessage(0x5d6b67e2);  // shader module
  context.ignoreDebugMessage(0x1292ada1);

  m_renderer = nullptr;

  {
    VkPhysicalDeviceProperties2 physicalProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceShaderSMBuiltinsPropertiesNV smProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV};
    physicalProperties.pNext = &smProperties;
    vkGetPhysicalDeviceProperties2(context.m_physicalDevice, &physicalProperties);
    // ommit * 32 here, gives much better perf
    m_rendererConfig.persistentThreads = smProperties.shaderSMCount * smProperties.shaderWarpsPerSM;
  }

  {
    std::vector<std::string> shaderSearchPaths;
    std::string              path = NVPSystem::exePath();
    shaderSearchPaths.push_back(NVPSystem::exePath());
    shaderSearchPaths.push_back(NVPSystem::exePath() + "shaders");
    shaderSearchPaths.push_back(std::string("GLSL_" PROJECT_NAME));
    shaderSearchPaths.push_back(NVPSystem::exePath() + std::string("GLSL_" PROJECT_NAME));
    shaderSearchPaths.push_back(NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY));
    shaderSearchPaths.push_back(NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY) + "shaders");
    shaderSearchPaths.push_back(NVPSystem::exePath() + std::string(PROJECT_NVPRO_CORE_RELDIRECTORY) + "nvvkhl/shaders");

    bool valid = m_resources.init(&context, shaderSearchPaths);
    valid      = valid && m_resources.initFramebuffer(winWidth, winHeight, m_tweak.supersample, m_tweak.hbaoFullRes);

    updateTargetImage();

    if(!valid)
    {
      LOGE("failed to initialize resources\n");
      exit(-1);
    }
  }

  {
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
    m_frameConfig.frameConstants.skyParams  = nvvkhl_shaders::initSimpleSkyParameters();
  }

  {
    m_ui.enumAdd(GUI_RENDERER, RENDERER_RASTER_CLUSTERS_TESS, "Rasterization");

    m_ui.enumAdd(GUI_BUILDMODE, BUILD_DEFAULT, "default");
    m_ui.enumAdd(GUI_BUILDMODE, BUILD_FAST_BUILD, "fast build");
    m_ui.enumAdd(GUI_BUILDMODE, BUILD_FAST_TRACE, "fast trace");

    if(m_clusterSupport)
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
    m_ui.enumAdd(GUI_MESHLET, CLUSTER_64T_64V, "64T_64V");
    m_ui.enumAdd(GUI_MESHLET, CLUSTER_64T_128V, "64T_128V");
    m_ui.enumAdd(GUI_MESHLET, CLUSTER_96T_96V, "96T_96V");
    m_ui.enumAdd(GUI_MESHLET, CLUSTER_96T_192V, "96T_192V");
    m_ui.enumAdd(GUI_MESHLET, CLUSTER_128T_128V, "128T_128V");
    m_ui.enumAdd(GUI_MESHLET, CLUSTER_128T_256V, "128T_256V");
    m_ui.enumAdd(GUI_MESHLET, CLUSTER_256T_256V, "256T_256V");
    m_ui.enumAdd(GUI_MESHLET, CLUSTER_CUSTOM, "CUSTOM");

    m_ui.enumAdd(GUI_SUPERSAMPLE, 1, "none");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 2, "4x");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_NONE, "Material");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_CLUSTER, "Clusters");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TESSELLATED_CLUSTER, "Tessellated clusters");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TESSELLATED_TRIANGLES, "Tessellated triangles");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TRIANGLES, "Triangles");
  }

  updatedClusterConfig();

  // Search for default scene if none was provided on the command line
  if(m_modelFilename.empty())
  {
    const std::vector<std::string> defaultSearchPaths = {NVPSystem::exePath() + PROJECT_DOWNLOAD_RELDIRECTORY,
                                                         NVPSystem::exePath() + "media"};  // for INSTALL search path
    m_modelFilename                                   = nvh::findFile("bunny_v2/bunny.gltf", defaultSearchPaths, true);
    if(m_tweak.gridCopies == 1)
    {
      m_tweak.gridCopies = 121;  // 11x11 grid
    }
  }


  if(initScene(!m_modelFilename.empty() ? m_modelFilename.c_str() : nullptr))
  {
    postInitNewScene();
    initRenderer(m_tweak.renderer);
  }
  else
  {
    return false;
  }

  m_lastTweak               = m_tweak;
  m_lastSceneConfig         = m_sceneConfig;
  m_lastRendererConfig      = m_rendererConfig;
  m_lastCustomShaderPrepend = m_customShaderPrepend;

  m_mouseButtonHandler.init();

  return true;
}

void TessellatedClusters::deinit(nvvk::Context& context)
{
  deinitRenderer();
  deinitScene();
  m_resources.deinit();
}

void TessellatedClusters::loadFile(std::string& filename)
{
  // reset grid parameter (in case scene is too large to be replicated)
  m_tweak.gridCopies = 1;
  //
  if(filename.empty())
    return;
  m_modelFilename = filename;
  LOGI("Loading model: %s\n", filename.c_str());
  deinitRenderer();
  m_resources.synchronize("open file");

  if(initScene(m_modelFilename.c_str()))
  {
    postInitNewScene();
    initRenderer(m_tweak.renderer);
    m_lastTweak       = m_tweak;
    m_lastSceneConfig = m_sceneConfig;
  }
}


void TessellatedClusters::onSceneChanged()
{
  m_resources.synchronize("sync sceneChanged");
  deinitRenderer();
  initScene(m_modelFilename.c_str());
}

tessellatedclusters::TessellatedClusters::ClusterConfig TessellatedClusters::getClusterConfig(const SceneConfig& sceneConfig)
{
  struct Entry
  {
    uint32_t      tris;
    uint32_t      verts;
    ClusterConfig cfg;
  };

  Entry entries[] = {
      {64, 64, CLUSTER_64T_64V},     {64, 128, CLUSTER_64T_128V},   {96, 96, CLUSTER_96T_96V},
      {96, 192, CLUSTER_96T_192V},   {128, 128, CLUSTER_128T_128V}, {128, 256, CLUSTER_128T_256V},
      {256, 256, CLUSTER_256T_256V},
  };

  for(size_t i = 0; i < NV_ARRAY_SIZE(entries); i++)
  {
    const Entry& entry = entries[i];
    if(sceneConfig.clusterTriangles <= entry.tris && sceneConfig.clusterVertices <= entry.verts)
      return entry.cfg;
  }

  return CLUSTER_CUSTOM;
}

void TessellatedClusters::updatedClusterConfig()
{
  switch(m_tweak.clusterConfig)
  {
    case CLUSTER_64T_64V:
      m_sceneConfig.clusterTriangles = 64;
      m_sceneConfig.clusterVertices  = 64;
      break;
    case CLUSTER_64T_128V:
      m_sceneConfig.clusterTriangles = 64;
      m_sceneConfig.clusterVertices  = 128;
      break;
    case CLUSTER_96T_96V:
      m_sceneConfig.clusterTriangles = 96;
      m_sceneConfig.clusterVertices  = 96;
      break;
    case CLUSTER_96T_192V:
      m_sceneConfig.clusterTriangles = 96;
      m_sceneConfig.clusterVertices  = 192;
      break;
    case CLUSTER_128T_128V:
      m_sceneConfig.clusterTriangles = 128;
      m_sceneConfig.clusterVertices  = 128;
      break;
    case CLUSTER_128T_256V:
      m_sceneConfig.clusterTriangles = 128;
      m_sceneConfig.clusterVertices  = 256;
      break;
    case CLUSTER_256T_256V:
      m_sceneConfig.clusterTriangles = 256;
      m_sceneConfig.clusterVertices  = 256;
      break;
  }
}

void TessellatedClusters::applyConfigFile(nvh::ParameterList& parameterList, const char* filename)
{
  std::string result = nvh::loadFile(filename, false);
  if(result.empty())
  {
    LOGW("file not found: %s\n", filename);
    return;
  }
  std::vector<const char*> args;
  nvh::ParameterList::tokenizeString(result, args);

  std::string path = nvh::getFilePath(filename);

  parameterList.applyTokens(uint32_t(args.size()), args.data(), "-", path.c_str());
}

TessellatedClusters::ChangeStates TessellatedClusters::handleChanges(uint32_t width, uint32_t height, const EventStates& states)
{
  ChangeStates changes = {};
  changes.targetImage  = 0;
  changes.timerReset   = 0;

  if(m_tweak.clusterConfig != m_lastTweak.clusterConfig)
  {
    updatedClusterConfig();
  }

  bool sceneChanged = false;
  if(memcmp(&m_sceneConfig, &m_lastSceneConfig, sizeof(m_sceneConfig)))
  {
    sceneChanged = true;

    onSceneChanged();
  }

  bool shaderChanged = false;
  if(states.reloadShaders || m_customShaderPrepend != m_lastCustomShaderPrepend || tweakChanged(m_tweak.doAnimation)
     || tweakChanged(m_tweak.doCulling))
  {
    shaderChanged = true;
  }

  if(tweakChanged(m_tweak.supersample) || tweakChanged(m_tweak.hbaoFullRes))
  {
    m_resources.initFramebuffer(width, height, m_tweak.supersample, m_tweak.hbaoFullRes);
    updateTargetImage();
    changes.targetImage = 1;
  }

  bool rendererChanged = false;
  if(sceneChanged || shaderChanged || tweakChanged(m_tweak.renderer) || tweakChanged(m_tweak.gridCopies)
     || tweakChanged(m_tweak.gridConfig) || tweakChanged(m_tweak.flipWinding) || rendererCfgChanged(m_rendererConfig.persistentThreads)
     || (m_renderer && m_renderer->supportsClusters()
         && (rendererCfgChanged(m_rendererConfig.positionTruncateBits) || rendererCfgChanged(m_rendererConfig.pnDisplacement)
             || rendererCfgChanged(m_rendererConfig.numVisibleClusterBits) || rendererCfgChanged(m_rendererConfig.numSplitTriangleBits)
             || rendererCfgChanged(m_rendererConfig.numGeneratedVerticesBits)
             || rendererCfgChanged(m_rendererConfig.numPartTriangleBits) || rendererCfgChanged(m_rendererConfig.numGeneratedClusterMegs)
             || rendererCfgChanged(m_rendererConfig.transientClusters1X) || rendererCfgChanged(m_rendererConfig.transientClusters2X)
             || tweakChanged(m_tweak.clusterBuildMode) || rendererCfgChanged(m_rendererConfig.rasterBatchMeshlets)))
     || tweakChanged(m_tweak.useDebugVisualization))
  {
    rendererChanged = true;

    m_resources.synchronize("sync rendererChanged");
    initRenderer(m_tweak.renderer);
    m_resources.m_hbaoPass.reloadShaders();
  }

  bool hadChange = shaderChanged || memcmp(&m_lastTweak, &m_tweak, sizeof(m_tweak))
                   || memcmp(&m_lastRendererConfig, &m_rendererConfig, sizeof(m_rendererConfig))
                   || memcmp(&m_lastSceneConfig, &m_sceneConfig, sizeof(m_sceneConfig));

  if(hadChange)
  {
    m_equalFrames = 0;
  }

  m_lastTweak               = m_tweak;
  m_lastRendererConfig      = m_rendererConfig;
  m_lastSceneConfig         = m_sceneConfig;
  m_lastCustomShaderPrepend = m_customShaderPrepend;

  changes.timerReset = hadChange && m_tweak.autoResetTimers;

  return changes;
}

void TessellatedClusters::renderFrame(VkCommandBuffer cmd, uint32_t width, uint32_t height, double time, nvvk::ProfilerVK& profilerVK, uint32_t cycleIndex)
{
  m_resources.beginFrame(cycleIndex);

  m_frameConfig.winWidth  = width;
  m_frameConfig.winHeight = height;

  if(m_renderer)
  {
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
    frameConstants.doAnimation  = m_tweak.doAnimation ? 1 : 0;
    frameConstants.visualize    = m_tweak.visualizeMode;

    // this is set systematically since selection is removed from the class
    // TODO: might need to remove some code in the shaders.
    // if(!m_hasSelection)
    {
      frameConstants.visFilterClusterID  = ~0;
      frameConstants.visFilterInstanceID = ~0;
    }
    if(m_tweak.doAnimation)
    {
      if(m_tweak.overrideTime)
      {
        m_frameConfig.frameConstants.animationState = m_tweak.overrideTime;
        m_animTime                                  = m_tweak.overrideTime;
      }
      else
      {
        m_animTime += (time - m_lastTime) * 0.5;
        m_frameConfig.frameConstants.animationState = (m_animTime);
      }
    }
    frameConstants.bgColor     = m_resources.m_bgColor;
    frameConstants.flipWinding = m_tweak.flipWinding ? 1 : 0;

    frameConstants.viewport    = glm::ivec2(renderWidth, renderHeight);
    frameConstants.viewportf   = glm::vec2(renderWidth, renderHeight);
    frameConstants.supersample = m_tweak.supersample;
    frameConstants.nearPlane   = CameraManip.getClipPlanes().x;
    frameConstants.farPlane    = CameraManip.getClipPlanes().y;
    frameConstants.wUpDir      = CameraManip.getUp();

    glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), float(width) / float(height),
                                                 frameConstants.nearPlane, frameConstants.farPlane);
    projection[1][1] *= -1;

    glm::mat4 view  = CameraManip.getMatrix();
    glm::mat4 viewI = glm::inverse(view);

    frameConstants.viewProjMatrix  = projection * view;
    frameConstants.viewProjMatrixI = glm::inverse(frameConstants.viewProjMatrix);
    frameConstants.viewMatrix      = view;
    frameConstants.viewMatrixI     = viewI;
    frameConstants.projMatrix      = projection;
    frameConstants.projMatrixI     = glm::inverse(projection);
    frameConstants.tessRate        = m_tweak.tessRatePixels ? 1.0f / float(m_tweak.tessRatePixels) : 0.0f;

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

      if(memcmp(&frameCurrent, &m_frameConfig.frameConstantsLast, sizeof(shaderio::FrameConstants)))
        m_equalFrames = 0;
      else
        m_equalFrames++;
    }

    m_renderer->render(cmd, m_resources, *m_scene, m_frameConfig, profilerVK);
  }
  else
  {
    m_resources.emptyFrame(cmd, m_frameConfig, profilerVK);
  }

  {
    m_resources.postProcessFrame(cmd, m_frameConfig, profilerVK);
  }

  m_resources.endFrame();

  m_lastTime = time;
  m_frames++;
}


void TessellatedClusters::setCameraFromScene(const char* filename)
{
  ImGuiH::SetCameraJsonFile(std::filesystem::path(filename).stem().string());

  float     radius = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo) * 0.5f;
  glm::vec3 center = (m_scene->m_bbox.hi + m_scene->m_bbox.lo) * 0.5f;

  if(!m_scene->m_cameras.empty())
  {
    auto& c = m_scene->m_cameras[0];
    //CameraManip.setMatrix(c.worldMatrix);
    CameraManip.setFov(c.fovy);


    c.eye              = glm::vec3(c.worldMatrix[3]);
    float     distance = glm::length(center - c.eye);
    glm::mat3 rotMat   = glm::mat3(c.worldMatrix);
    c.center           = {0, 0, -distance};
    c.center           = c.eye + (rotMat * c.center);
    c.up               = {0, 1, 0};

    CameraManip.setCamera({c.eye, c.center, c.up, static_cast<float>(glm::degrees(c.fovy))});

    ImGuiH::SetHomeCamera({c.eye, c.center, c.up, static_cast<float>(glm::degrees(c.fovy))});
    for(auto& cam : m_scene->m_cameras)
    {
      cam.eye            = glm::vec3(cam.worldMatrix[3]);
      float     distance = glm::length(center - cam.eye);
      glm::mat3 rotMat   = glm::mat3(cam.worldMatrix);
      cam.center         = {0, 0, -distance};
      cam.center         = cam.eye + (rotMat * cam.center);
      cam.up             = {0, 1, 0};


      ImGuiH::AddCamera({cam.eye, cam.center, cam.up, static_cast<float>(glm::degrees(cam.fovy))});
    }
  }
  else
  {
    // Re-adjusting camera to fit the new scene
    CameraManip.fit(m_scene->m_bbox.lo, m_scene->m_bbox.hi, true);
    ImGuiH::SetHomeCamera(CameraManip.getCamera());
  }

  CameraManip.setClipPlanes(glm::vec2(0.01F * radius, 100.0F * radius));
}

float TessellatedClusters::decodePickingDepth(const shaderio::Readback& readback)
{
  if(!isReadbackValid(readback))
  {
    return 0.f;
  }
  uint32_t bits = readback._packedDepth0;
  bits ^= ~(int(bits) >> 31) | 0x80000000u;
  float res = *(float*)&bits;
  return 1.f - res;
}

bool TessellatedClusters::isReadbackValid(const shaderio::Readback& readback)
{
  return readback._packedDepth0 != 0u;
}

void TessellatedClusters::setupConfigParameters(nvh::ParameterList& parameterList)
{
  parameterList.add("verbose", &g_verbose, true);

  parameterList.add("resetstats", &m_tweak.autoResetTimers);

  parameterList.add("renderer", (uint32_t*)&m_tweak.renderer);
  parameterList.add("supersample", &m_tweak.supersample);
  parameterList.add("gridcopies", &m_tweak.gridCopies);
  parameterList.add("gridconfig", &m_tweak.gridConfig);
  parameterList.add("clusterconfig", (int*)&m_tweak.clusterConfig);
  parameterList.add("tessrate", &m_tweak.tessRatePixels);
  parameterList.add("animation", &m_tweak.doAnimation);
  parameterList.add("overridetime", &m_tweak.overrideTime);
  parameterList.add("culling", &m_tweak.doCulling);

  parameterList.addFilename(".gltf", &m_modelFilename);
  parameterList.addFilename(".glb", &m_modelFilename);
}

void TessellatedClusters::setupContextInfo(nvvk::ContextCreateInfo& contextInfo)
{
  static VkPhysicalDeviceMeshShaderFeaturesNV meshFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV};

  static VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR execPropertiesFeatures = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  static VkPhysicalDeviceImagelessFramebufferFeaturesKHR imagelessFeatures = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES_KHR};
  static VkPhysicalDeviceShaderClockFeaturesKHR clockFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};

  static VkPhysicalDeviceAccelerationStructureFeaturesKHR accFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  static VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  static VkPhysicalDeviceRayQueryFeaturesKHR queryFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  static VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clusterFeatures = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV};

  static VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};

  static VkPhysicalDeviceFragmentShadingRateFeaturesKHR shadingRateFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR};

  static VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentricFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};

  static VkPhysicalDeviceShaderSMBuiltinsFeaturesNV shaderBuiltinFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV};

  static VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR shaderRayPosFetchFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};

  contextInfo.apiMajor = 1;
  contextInfo.apiMinor = 3;

  contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

#if defined(_DEBUG) && 0
  // enable debugPrintf
  contextInfo.addDeviceExtension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME, false);
  static VkValidationFeaturesEXT      validationInfo    = {VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
  static VkValidationFeatureEnableEXT enabledFeatures[] = {VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
  validationInfo.enabledValidationFeatureCount          = NV_ARRAY_SIZE(enabledFeatures);
  validationInfo.pEnabledValidationFeatures             = enabledFeatures;
  contextInfo.instanceCreateInfoExt                     = &validationInfo;
#ifdef _WIN32
  _putenv_s("DEBUG_PRINTF_TO_STDOUT", "1");
#else
  putenv("DEBUG_PRINTF_TO_STDOUT=1");
#endif
#endif  // _DEBUG

  contextInfo.addDeviceExtension(VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME, false);
  contextInfo.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, false);
  contextInfo.addDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, false, &clockFeatures);
  contextInfo.addDeviceExtension(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, true, &shadingRateFeatures);
  contextInfo.addDeviceExtension(VK_NV_MESH_SHADER_EXTENSION_NAME, false, &meshFeatures);

  contextInfo.addDeviceExtension(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, true, &execPropertiesFeatures);

  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, false);
  contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accFeatures);
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rayFeatures);
  contextInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &queryFeatures);

  contextInfo.addDeviceExtension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, false, &atomicFloatFeatures);
  contextInfo.addDeviceExtension(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, false, &barycentricFeatures);
  contextInfo.addDeviceExtension(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME, false, &shaderBuiltinFeatures);
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, false, &shaderRayPosFetchFeatures);

  contextInfo.addDeviceExtension(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME, true, &clusterFeatures,
                                 VK_NV_CLUSTER_ACCELERATION_STRUCTURE_SPEC_VERSION);
}
}  // namespace tessellatedclusters
