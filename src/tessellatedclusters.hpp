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
#pragma once

#include <memory>

#include <nvvk/context_vk.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvh/timesampler.hpp>
#include <nvh/parametertools.hpp>
#include <vulkan/vulkan_core.h>
#include <imgui/imgui_helper.h>
#include <implot.h>
#include <glm/glm.hpp>

#include "resources.hpp"
#include "scene.hpp"
#include "renderer.hpp"
#include "shaders/shaderio.h"


namespace tessellatedclusters {
int const SAMPLE_SIZE_WIDTH(1024);
int const SAMPLE_SIZE_HEIGHT(1024);
class TessellatedClusters
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
    CLUSTER_96T_96V,
    CLUSTER_96T_192V,
    CLUSTER_128T_128V,
    CLUSTER_128T_256V,
    CLUSTER_256T_256V,
    CLUSTER_CUSTOM,
  };

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
    GUI_SUPERSAMPLE,
    GUI_MESHLET,
    GUI_VISUALIZE,
  };

  struct Tweak
  {
    ClusterConfig clusterConfig = CLUSTER_64T_64V;

    RendererType renderer    = RENDERER_RAYTRACE_CLUSTERS_TESS;
    int          supersample = 2;

    uint32_t visualizeMode = VISUALIZE_TESSELLATED_CLUSTER;
    bool     doAnimation   = false;
    bool     doCulling     = false;

    float overrideTime = 0.0f;
    bool  flipWinding  = false;
    bool  facetShading = true;

    uint32_t tessRatePixels = 4;

    uint32_t gridCopies = 1;
    uint32_t gridConfig = 13;

    bool  hbaoFullRes = false;
    bool  hbaoActive  = true;
    float hbaoRadius  = 0.05f;

    bool autoResetTimers       = false;
    bool useDebugVisualization = true;

    BuildMode clusterBuildMode = BUILD_DEFAULT;
  };

  struct MouseButtonHandler
  {
    enum InternalState
    {
      eInternalNone,
      eInternalFirstDown,
      eInternalFirstUp,
      eInternalDrag,
      eInternalFirstClick,
      eInternalSecondDown,
      eInternalSecondUp,
      eInternalSecondClick
    };


    struct ButtonClick
    {
      glm::uvec2     pos;
      InternalState  internal = eInternalNone;
      double         firstUpTime;
      nvh::Stopwatch internalTime;
    };
    inline void init()
    {
      for(auto& p : mouseClickPos)
      {
        p.pos = glm::uvec2(~0u, ~0u);
      }
    }

    inline void update(glm::uvec2 mousePos) { currentPos = mousePos; }


    enum ButtonState
    {
      eNone,
      eSingleClick,
      eDoubleClick,
      eDrag
    };

    ButtonState getButtonState(ImGuiMouseButton button)
    {
      ButtonClick& b = mouseClickPos[button];

      bool isDown  = ImGui::IsMouseDown(button);
      bool isUp    = ImGui::IsMouseReleased(button);
      bool isMoved = (b.pos != currentPos);

      switch(b.internal)
      {
        case eInternalNone:
          if(isDown)
          {
            b.internal = eInternalFirstDown;
            b.pos      = currentPos;
          }
          break;

        case eInternalFirstDown:
          if(isUp)
          {
            b.internal    = eInternalFirstUp;
            b.firstUpTime = b.internalTime.elapsed();
            break;
          }
          if(isMoved)
          {
            b.internal = eInternalDrag;
          }
          break;

        case eInternalFirstUp: {
          double elapsed = b.internalTime.elapsed() - b.firstUpTime;
          if(isMoved || elapsed > doubleClickThreshold)
          {
            b.internal = eInternalFirstClick;
            break;
          }
          if(isDown)
          {
            b.internal = eInternalSecondDown;
          }
          break;
        }
        case eInternalFirstClick:
          b.internal = eInternalNone;
          break;

        case eInternalDrag:
          if(isUp)
          {
            b.internal = eInternalNone;
          }
          break;

        case eInternalSecondDown:
          if(isMoved)
          {
            b.internal = eInternalDrag;
            break;
          }
          if(isUp)
          {
            b.internal = eInternalSecondUp;
          }
          break;
        case eInternalSecondUp:
          b.internal = eInternalNone;
          break;
      }


      switch(b.internal)
      {
        case eInternalFirstClick:
          return eSingleClick;
        case eInternalDrag:
          return eDrag;
        case eInternalSecondUp:
          return eDoubleClick;
        default:
          return eNone;
      }
    }

    std::array<ButtonClick, ImGuiMouseButton_COUNT> mouseClickPos;
    glm::uvec2                                      currentPos;
    const double                                    doubleClickThreshold = 100.0;
  };

  struct ViewPoint
  {
    std::string name;
    glm::mat4   mat;
    float       sceneScale;
    float       fov;
  };

  struct TargetImage
  {
    VkImage     image;
    VkImageView view;
    VkFormat    format;
  };

  bool m_clusterSupport = false;

  ImGuiH::Registry m_ui;
  double           m_uiTime = 0;

  Tweak m_tweak;
  Tweak m_lastTweak;

  FrameConfig m_frameConfig;
  uint32_t    m_equalFrames = 0;

  std::unique_ptr<Scene> m_scene;
  SceneConfig            m_sceneConfig;
  SceneConfig            m_lastSceneConfig;
  RendererConfig         m_rendererConfig;
  RendererConfig         m_lastRendererConfig;


  std::unique_ptr<Renderer> m_renderer;
  Resources                 m_resources;
  std::string               m_rendererShaderPrepend;
  std::string               m_rendererLastShaderPrepend;
  TargetImage               m_targetImage;
  size_t                    m_rendererFboChangeID;

  std::string m_customShaderPrepend;
  std::string m_lastCustomShaderPrepend;

  std::string m_modelFilename;
  glm::vec3   m_modelUpVector = glm::vec3(0, 1, 0);

  std::string m_dragDropName;

  int    m_frames        = 0;
  double m_lastFrameTime = 0;
  double m_statsCpuTime  = 0;
  double m_statsGpuTime  = 0;
  double m_statsDrwTime  = 0;
  double m_statsRdrTime  = 0;
  double m_statsBldTime  = 0;
  double m_statsAnmTime  = 0;

  double m_animTime = 0;
  double m_lastTime = 0;

  bool m_requestCameraRecenter = false;
  /* JEM dactivates selection, TODO remove code once validated with CK
  bool m_requestSelection      = false;
  bool               m_hasSelection  = false;
  shaderio::Readback m_selectionData = {};
  */

  static void setupContextInfo(nvvk::ContextCreateInfo& info);

  bool initCore(nvvk::Context& context, int winWidth, int winHeight, const std::string& exePath);
  void deinit(nvvk::Context& context);

  void loadFile(std::string& filename);
  bool initScene(const char* filename);
  void deinitScene();
  void postInitNewScene();

  bool initFramebuffers(int width, int height);
  void updateTargetImage();
  void initRenderer(RendererType rtype);
  void deinitRenderer();

  void setupConfigParameters(nvh::ParameterList& parameterList);

  void onSceneChanged();

  ClusterConfig getClusterConfig(const SceneConfig& sceneConfig);
  void          updatedClusterConfig();

  void applyConfigFile(nvh::ParameterList& parameterList, const char* filename);

  std::string getShaderPrepend();

  struct CallBacks
  {
    std::function<std::string(const char* what, const char* files)> openFile;
    nvh::ParameterList*                                             parameterList;
  };

  void processUI(double time, nvh::Profiler& profiler, const CallBacks& callbacks);
  void viewportUI(ImVec2 corner);

  struct EventStates
  {
    glm::ivec2 winSize          = {};
    glm::ivec2 mouse            = {};
    int        mouseButtonFlags = 0;
    int        mouseWheel       = 0;

    bool reloadShaders = false;
    bool alignView     = false;
    bool saveView      = false;
  };

  struct ChangeStates
  {
    uint32_t timerReset : 1;
    uint32_t targetImage : 1;
  };
  ChangeStates handleChanges(uint32_t width, uint32_t height, const EventStates& states);

  void renderFrame(VkCommandBuffer cmd, uint32_t width, uint32_t height, double time, nvvk::ProfilerVK& profilerVK, uint32_t cycleIndex);

  template <typename T>
  bool sceneChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_sceneConfig);
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_lastSceneConfig) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool tweakChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_tweak);
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_lastTweak) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool rendererCfgChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_rendererConfig);
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_lastRendererConfig) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool tweakChangedNonZero(const T& val) const
  {
    size_t   offset  = size_t(&val) - size_t(&m_tweak);
    const T* lastVal = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(&m_lastTweak) + offset);
    bool     state   = (val != 0) != (*lastVal != 0);
    return state;
  }

  template <typename T>
  bool tweakChangedPositive(const T& val) const
  {
    size_t   offset  = size_t(&val) - size_t(&m_tweak);
    const T* lastVal = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(&m_lastTweak) + offset);
    bool     state   = (val >= 0) != (*lastVal >= 0);
    return state;
  }

  template <typename T>
  void uiPlot(const std::string& plotName, const std::string& tooltipFormat, const std::vector<T>& data, const T& maxValue)
  {
    ImVec2 plotSize = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2);

    // Ensure minimum height to avoid overly squished graphics
    plotSize.y = std::max(plotSize.y, ImGui::GetTextLineHeight() * 20);

    const ImPlotFlags     plotFlags = ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText | ImPlotFlags_Crosshairs;
    const ImPlotAxisFlags axesFlags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel;
    const ImColor         plotColor = ImColor(0.07f, 0.9f, 0.06f, 1.0f);

    if(ImPlot::BeginPlot(plotName.c_str(), plotSize, plotFlags))
    {
      ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_NoButtons);
      ImPlot::SetupAxes(nullptr, "Count", axesFlags, axesFlags);
      ImPlot::SetupAxesLimits(0, data.size(), 0, static_cast<double>(maxValue), ImPlotCond_Always);

      ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
      ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
      ImPlot::SetNextFillStyle(plotColor);
      ImPlot::PlotShaded(plotName.c_str(), data.data(), (int)data.size(), -INFINITY, 1.0, 0.0, 0, 0);
      ImPlot::PopStyleVar();

      if(ImPlot::IsPlotHovered())
      {
        ImPlotPoint mouse       = ImPlot::GetPlotMousePos();
        int         mouseOffset = (int(mouse.x)) % (int)data.size();
        ImGui::BeginTooltip();
        ImGui::Text(tooltipFormat.c_str(), mouseOffset, data[mouseOffset]);
        ImGui::EndTooltip();
      }

      ImPlot::EndPlot();
    }
  }
  void setCameraFromScene(const char* filename);

  float decodePickingDepth(const shaderio::Readback& readback);
  bool  isReadbackValid(const shaderio::Readback& readback);

  MouseButtonHandler m_mouseButtonHandler;
};
}  // namespace tessellatedclusters
