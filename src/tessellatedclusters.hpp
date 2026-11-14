/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <backends/imgui_impl_vulkan.h>
#include <nvapp/application.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/parameter_registry.hpp>
#include <nvutils/parameter_sequencer.hpp>
#include <nvvk/context.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvgui/enum_registry.hpp>

#include "camera_path.hpp"
#include "renderer.hpp"


namespace tessellatedclusters {

class TessellatedClusters : public nvapp::IAppElement
{
public:
  enum RendererType
  {
    RENDERER_RASTER_CLUSTERS_TESS,
    RENDERER_RAYTRACE_CLUSTERS_TESS,
  };

  enum ClusterConfig
  {
    CLUSTER_64T_64V,
    CLUSTER_64T_128V,
    CLUSTER_64T_192V,
    CLUSTER_96T_96V,
    CLUSTER_128T_128V,
    CLUSTER_128T_256V,
    CLUSTER_256T_256V,
    NUM_CLUSTER_CONFIGS,
  };

  struct ClusterInfo
  {
    uint32_t      tris;
    uint32_t      verts;
    ClusterConfig cfg;
  };

  static const ClusterInfo s_clusterInfos[NUM_CLUSTER_CONFIGS];

  enum BuildMode
  {
    BUILD_DEFAULT,
    BUILD_FAST_BUILD,
    BUILD_FAST_TRACE,
  };

  enum GuiEnums
  {
    GUI_RENDERER,
    GUI_BUILDMODE,
    GUI_LOWTESSMODE,
    GUI_SUPERSAMPLE,
    GUI_MESHLET,
    GUI_VISUALIZE,
  };

  struct Tweak
  {
    ClusterConfig clusterConfig = CLUSTER_64T_64V;

    RendererType renderer    = RENDERER_RAYTRACE_CLUSTERS_TESS;
    int          supersample = 2;
    bool         hbaoQuarter = true;

    uint32_t visualizeMode = VISUALIZE_TESSELLATED_CLUSTER;

    float overrideTime = 0.0f;
    bool  facetShading = true;

    uint32_t tessRatePixels = 4;

    uint32_t gridCopies = 1;
    uint32_t gridConfig = 13;

    bool  hbaoActive = true;
    float hbaoRadius = 0.05f;

    bool autoResetTimers = false;

    BuildMode clusterBuildMode = BUILD_DEFAULT;
  };


  struct ViewPoint
  {
    std::string name;
    glm::mat4   mat;
    float       sceneScale;
    float       fov;
  };

  enum ScreenshotMode
  {
    SCREENSHOT_OFF,
    SCREENSHOT_WINDOW,
    SCREENSHOT_VIEWPORT
  };

  struct TargetImage
  {
    VkImage     image;
    VkImageView view;
    VkFormat    format;
  };

  struct Info
  {
    nvutils::ProfilerManager*                   profilerManager{};
    nvutils::ParameterRegistry*                 parameterRegistry{};
    std::shared_ptr<nvutils::CameraManipulator> cameraManipulator;
  };

  TessellatedClusters(const Info& info);

  ~TessellatedClusters() override { m_info.profilerManager->destroyTimeline(m_profilerTimeline); }

  void onAttach(nvapp::Application* app) override;
  void onDetach() override;
  void onUIRender() override;
  void onUIMenu() override;
  void onPreRender() override;
  void onRender(VkCommandBuffer cmd) override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;
  void onFileDrop(const std::filesystem::path& filename) override;
  void onLastHeadlessFrame() override;

  // appends memory statistics to the benchmark output of a parameter sequence
  void parameterSequenceCallback(const nvutils::ParameterSequencer::State& state);

  void setSupportsClusters(bool supported) { m_resources.m_supportsClusters = supported; }
  void setSupportsMeshShaderNV(bool supported) { m_resources.m_supportsMeshShaderNV = supported; }
  void setSupportsPipelineExecutableInfo(bool supported) { m_resources.m_supportsPipelineExecutableInfo = supported; }
#if USE_DLSS
  // otherwise renders without DLSS until its NGX init finished
  void setDlssWaitForInit(bool wait) { m_resources.m_dlssWaitForInit = wait; }
#endif

  bool isDebugUI() const { return m_debugUI; }

private:
  VkExtent2D                 m_windowSize;
  Info                       m_info;
  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer{};
  nvapp::Application*        m_app{};

  //////////////////////////////////////////////////////////////////////////

  // key components

  Resources                 m_resources;
  FrameConfig               m_frameConfig;
  double                    m_lastTime = 0;
  VkDescriptorSet           m_imguiTexture{};
  VkSampler                 m_imguiSampler{};
  nvgui::EnumRegistry       m_ui;
  nvutils::PerformanceTimer m_clock;

  void saveScreenshot(ScreenshotMode mode, const std::string& filename);

  ScreenshotMode m_sequenceScreenshotMode  = SCREENSHOT_OFF;
  ScreenshotMode m_screenshotMode          = SCREENSHOT_VIEWPORT;
  int            m_screenshotFrameInterval = 0;  // save a screenshot every n frames, 0 disables
  std::string    m_screenshotTimeStamp;          // session startup time, keeps screenshots of multiple runs apart

  int32_t m_aoSamplesOverride = -1;  // negative keeps the scene's default

  bool   m_reloadShaders         = false;
  bool   m_requestCameraRecenter = false;
  int    m_frames                = 0;
  double m_animTime              = 0;

  bool m_dlssDelayLoad = true;

#ifndef NDEBUG
  bool m_debugUI = true;
#else
  bool m_debugUI = false;
#endif

  Tweak m_tweak;
  Tweak m_tweakLast;

  std::unique_ptr<Scene> m_scene;
  std::filesystem::path  m_sceneFilePath;
  SceneConfig            m_sceneConfig;
  SceneConfig            m_sceneConfigLast;
  glm::vec3              m_sceneUpVector = glm::vec3(0, 1, 0);

  // Fixed camera paths (fly-through), see camera_path.hpp.
  // Defined via `--addcamerapath` (order == index) and/or the UI, and
  // played back either in real-time (UI) or with a fixed number of frames
  // (`--runcamerapath <index> <framecount>`) for deterministic benchmarking.
  enum CameraPathPlayback
  {
    CAMERA_PATH_STOP,      // not playing
    CAMERA_PATH_REALTIME,  // advance by wall-clock time over `duration` seconds
    CAMERA_PATH_FIXED,     // advance one keyframe-step per rendered frame (deterministic)
  };

  std::vector<CameraPath> m_cameraPaths;  // path definitions, index == order added
  bool m_cameraPathsExternal = false;  // paths set via --addcamerapath/--loadcamerapaths (command line, config, or sequence); global, take precedence over per-scene files
  std::filesystem::path m_cameraPathsLoadedScene;  // scene for which the per-scene paths file was last loaded
  int                   m_cameraPathActive   = -1;
  CameraPathPlayback    m_cameraPathPlayback = CAMERA_PATH_STOP;
  bool                  m_cameraPathStarted  = false;  // real-time: first frame has zero delta
  double                m_cameraPathU        = 0.0;    // normalized playback position [0,1]
  int                   m_cameraPathFrames   = 128;    // fixed: total frames for a full traversal
  int                   m_cameraPathFrame    = 0;      // fixed: current frame
  int                   m_cameraPathEditKey  = -1;     // UI: selected keyframe

  std::unique_ptr<Renderer> m_renderer;
  uint64_t                  m_rendererFboChangeID{};
  uint64_t                  m_imguiFboChangeID{};
  RendererConfig            m_rendererConfig;
  RendererConfig            m_rendererConfigLast;

  bool initScene(const std::filesystem::path& filePath);
  void setSceneCamera(const std::filesystem::path& filePath);
  void deinitScene();
  void postInitNewScene();

  // Camera path (see camera_path.hpp)
  void registerCameraPathParameters();   // `--addcamerapath`, `--loadcamerapaths`, `--runcamerapath`
  void updateCameraPath(double time);    // drive the manipulator during playback (called from onRender)
  bool applyCameraPathSample(double u);  // sample active path at `u` and set the camera; false if unavailable
  void cameraPathUI();
  // Per-scene camera paths file, stored next to the model file and named after
  // it (e.g. "<modeldir>/<scene>.camerapaths.txt")
  std::filesystem::path sceneCameraPathsFile(const std::filesystem::path& scenePath) const;

  void initRenderer(RendererType rtype);
  void deinitRenderer();

  void updateImguiImage();

  void adjustSceneClusterConfig();
  void updatedClusterConfig();

  void handleChanges();

  float decodePickingDepth(const shaderio::Readback& readback);
  bool  isPickingValid(const shaderio::Readback& readback);

  void viewportUI(ImVec2 corner, ImVec2 imageSize);

  template <typename T>
  bool sceneChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_sceneConfig);
    assert(offset < sizeof(m_sceneConfig));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_sceneConfigLast) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool tweakChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_tweak);
    assert(offset < sizeof(m_tweak));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_tweakLast) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool rendererCfgChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_rendererConfig);
    assert(offset < sizeof(m_rendererConfig));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_rendererConfigLast) + offset, sizeof(T)) != 0;
  }
};
}  // namespace tessellatedclusters
