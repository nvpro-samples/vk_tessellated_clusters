/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <span>

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <implot/implot.h>
#include <nvgui/camera.hpp>
#include <nvutils/file_operations.hpp>
#include <nvgui/sky.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/window.hpp>
#include <nvgui/file_dialog.hpp>

#include "tessellatedclusters.hpp"

namespace tessellatedclusters {


std::string formatMemorySize(size_t sizeInBytes)
{
  static const std::string units[]     = {"B", "KB", "MB", "GB"};
  static const size_t      unitSizes[] = {1, 1000, 1000 * 1000, 1000 * 1000 * 1000};

  uint32_t currentUnit = 0;
  for(uint32_t i = 1; i < 4; i++)
  {
    if(sizeInBytes < unitSizes[i])
    {
      break;
    }
    currentUnit++;
  }

  float size = float(sizeInBytes) / float(unitSizes[currentUnit]);

  return fmt::format("{:.3} {}", size, units[currentUnit]);
}

std::string formatMetric(size_t size)
{
  static const std::string units[]     = {"", "K", "M", "G"};
  static const size_t      unitSizes[] = {1, 1000, 1000 * 1000, 1000 * 1000 * 1000};

  uint32_t currentUnit = 0;
  for(uint32_t i = 1; i < 4; i++)
  {
    if(size < unitSizes[i])
    {
      break;
    }
    currentUnit++;
  }

  float fsize = float(size) / float(unitSizes[currentUnit]);

  return fmt::format("{:.3} {}", fsize, units[currentUnit]);
}

template <typename T>
void uiPlot(std::string plotName, std::string tooltipFormat, std::span<const T> data, const T& maxValue)
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
    ImPlot::SetupAxesLimits(0, static_cast<double>(data.size()), 0, static_cast<double>(maxValue), ImPlotCond_Always);

    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
    ImPlot::PlotShaded("", data.data(), (int)data.size(), -INFINITY, 1.0, 0.0,
                       ImPlotSpec(ImPlotProp_FillColor, (ImU32)plotColor, ImPlotProp_FillAlpha, 0.75f));

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

struct UsagePercentages
{
  uint32_t pctVisible = 0;

  uint32_t pctFull  = 0;
  uint32_t pctPart  = 0;
  uint32_t pctSplit = 0;

  uint32_t pctInst            = 0;
  uint32_t pctClusterReserved = 0;
  uint32_t pctClusterActual   = 0;
  uint32_t pctVertices        = 0;
  uint32_t pctBlas            = 0;

  static uint32_t getUsagePct(uint64_t requested, uint64_t reserved)
  {
    bool     exceeds = requested > reserved;
    uint32_t pct     = (reserved == 0) ? 100 : uint32_t(double(requested) * 100.0 / double(reserved));
    // artificially raise pct over 100 to trigger warning
    if(exceeds && pct < 101)
      pct = 101;
    return pct;
  }

  void setupPercentages(shaderio::Readback& readback, const RendererConfig& rendererConfig)
  {
    pctVisible = getUsagePct(readback.numVisibleClusters, 1ull << rendererConfig.numVisibleClusterBits);

    pctFull  = getUsagePct(readback.numFullClusters, (1ull << rendererConfig.numVisibleClusterBits));
    pctPart  = getUsagePct(readback.numPartTriangles, (1ull << rendererConfig.numPartTriangleBits));
    pctSplit = getUsagePct(readback.numSplitTriangles, (1ull << rendererConfig.numSplitTriangleBits));

    pctInst = getUsagePct(readback.numBlasClusters,
                          (1ull << rendererConfig.numVisibleClusterBits) + (1ull << rendererConfig.numPartTriangleBits));
    pctClusterReserved = getUsagePct(readback.numGenDatas, size_t(rendererConfig.numGeneratedClusterMegs) * 1024 * 1024);
    pctClusterActual = getUsagePct(readback.numGenActualDatas, size_t(rendererConfig.numGeneratedClusterMegs) * 1024 * 1024);

    pctVertices = getUsagePct(readback.numGenVertices, 1ull << rendererConfig.numGeneratedVerticesBits);
    pctBlas     = getUsagePct(readback.numBlasActualSizes, std::max(readback.numBlasReservedSizes, 1u));
  }

  const char* getWarning()
  {
    if(pctVisible > 100)
      return "WARNING: Scene: Visible clusters limit exceeded";
    if(pctFull > 100)
      return "WARNING: Tessellation: Full clusters limit exceeded";
    if(pctPart > 100)
      return "WARNING: Tessellation: Partial triangles limit exceeded";
    if(pctSplit > 100)
      return "WARNING: Tessellation: Split triangles limit exceeded";
    if(pctInst > 100)
      return "WARNING: Ray Tracing: Clusters limit exceeded";
    if(pctClusterReserved > 100)
      return "WARNING: Ray Tracing: Cluster reserved bytes exceeded";
    if(pctVertices > 100)
      return "WARNING: Ray Tracing: Vertices limit exceeded";
    if(pctBlas > 100)
      return "WARNING: Ray Tracing: BLAS reserved bytes exceeded";

    return nullptr;
  }
};

void TessellatedClusters::cameraPathUI()
{
  namespace PE = nvgui::PropertyEditor;

  // File: Save / Load the per-scene paths file stored next to the executable
  const std::filesystem::path pathsFile = sceneCameraPathsFile(m_sceneFilePath);
  ImGui::TextUnformatted("File:");
  ImGui::SameLine();
  ImGui::BeginDisabled(pathsFile.empty());
  if(ImGui::SmallButton("Save"))
  {
    if(saveCameraPaths(pathsFile, m_cameraPaths))
      LOGI("Saved %zu camera path(s) to %s\n", m_cameraPaths.size(), nvutils::utf8FromPath(pathsFile).c_str());
    else
      LOGE("Could not write camera paths to %s\n", nvutils::utf8FromPath(pathsFile).c_str());
  }
  ImGui::SameLine();
  if(ImGui::SmallButton("Load") && loadCameraPaths(pathsFile, m_cameraPaths))
  {
    m_cameraPathActive = m_cameraPaths.empty() ? -1 : std::min(std::max(m_cameraPathActive, 0), int(m_cameraPaths.size()) - 1);
    m_cameraPathEditKey  = -1;
    m_cameraPathPlayback = CAMERA_PATH_STOP;
  }
  ImGui::EndDisabled();
  if(pathsFile.empty())
    ImGui::TextDisabled("(load a scene to save/load its paths file)");
  else
    ImGui::TextDisabled("%s", nvutils::utf8FromPath(pathsFile.filename()).c_str());

  ImGui::Spacing();

  // Path: New / Delete / Copy / Paste (Copy/Paste exchange the path string via the clipboard)
  ImGui::TextUnformatted("Path:");
  ImGui::SameLine();
  if(ImGui::SmallButton("New##path"))
  {
    m_cameraPaths.emplace_back();
    m_cameraPathActive   = int(m_cameraPaths.size()) - 1;
    m_cameraPathEditKey  = -1;
    m_cameraPathPlayback = CAMERA_PATH_STOP;
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(m_cameraPathActive < 0 || m_cameraPathActive >= int(m_cameraPaths.size()));
  if(ImGui::SmallButton("Delete##path"))
  {
    m_cameraPaths.erase(m_cameraPaths.begin() + m_cameraPathActive);
    m_cameraPathActive   = m_cameraPaths.empty() ? -1 : std::min(m_cameraPathActive, int(m_cameraPaths.size()) - 1);
    m_cameraPathEditKey  = -1;
    m_cameraPathPlayback = CAMERA_PATH_STOP;
  }
  ImGui::SameLine();
  if(ImGui::SmallButton("Copy") && m_cameraPathActive >= 0 && m_cameraPathActive < int(m_cameraPaths.size()))
    ImGui::SetClipboardText(m_cameraPaths[m_cameraPathActive].getString().c_str());
  ImGui::EndDisabled();
  ImGui::SameLine();
  if(ImGui::SmallButton("Paste"))
  {
    const char* clip = ImGui::GetClipboardText();
    CameraPath  pasted;
    if(clip && pasted.setFromString(clip))
    {
      m_cameraPaths.push_back(std::move(pasted));
      m_cameraPathActive   = int(m_cameraPaths.size()) - 1;
      m_cameraPathEditKey  = -1;
      m_cameraPathPlayback = CAMERA_PATH_STOP;
    }
  }

  // active path selector
  {
    std::string preview = (m_cameraPathActive >= 0 && m_cameraPathActive < int(m_cameraPaths.size())) ?
                              fmt::format("Path {} ({} keys)", m_cameraPathActive, m_cameraPaths[m_cameraPathActive].size()) :
                              std::string("<none>");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if(ImGui::BeginCombo("##camerapathsel", preview.c_str()))
    {
      for(int i = 0; i < int(m_cameraPaths.size()); i++)
      {
        if(ImGui::Selectable(fmt::format("Path {} ({} keys)", i, m_cameraPaths[i].size()).c_str(), i == m_cameraPathActive))
        {
          m_cameraPathActive   = i;
          m_cameraPathEditKey  = -1;
          m_cameraPathPlayback = CAMERA_PATH_STOP;
          m_cameraPathU        = 0.0;
        }
      }
      ImGui::EndCombo();
    }
  }

  const bool hasPath = m_cameraPathActive >= 0 && m_cameraPathActive < int(m_cameraPaths.size());
  const nvutils::CameraManipulator::Camera& cam = m_info.cameraManipulator->getCamera();

  // Key: New / Update / Delete (captured from the current camera)
  ImGui::TextUnformatted("Key:");
  ImGui::SameLine();
  ImGui::BeginDisabled(!hasPath);
  if(ImGui::SmallButton("New##key") && hasPath)
  {
    CameraPath& keyPath = m_cameraPaths[m_cameraPathActive];
    int insertAt = (m_cameraPathEditKey >= 0 && m_cameraPathEditKey < int(keyPath.keys.size())) ? m_cameraPathEditKey + 1 :
                                                                                                  int(keyPath.keys.size());
    keyPath.keys.insert(keyPath.keys.begin() + insertAt, CameraPathKey{cam.eye, cam.ctr, cam.up, cam.fov});
    m_cameraPathEditKey = insertAt;
  }
  ImGui::SameLine();
  if(ImGui::SmallButton("Update##key") && hasPath && m_cameraPathEditKey >= 0
     && m_cameraPathEditKey < int(m_cameraPaths[m_cameraPathActive].keys.size()))
  {
    m_cameraPaths[m_cameraPathActive].keys[m_cameraPathEditKey] = CameraPathKey{cam.eye, cam.ctr, cam.up, cam.fov};
  }
  ImGui::SameLine();
  if(ImGui::SmallButton("Delete##key") && hasPath && m_cameraPathEditKey >= 0
     && m_cameraPathEditKey < int(m_cameraPaths[m_cameraPathActive].keys.size()))
  {
    CameraPath& keyPath = m_cameraPaths[m_cameraPathActive];
    keyPath.keys.erase(keyPath.keys.begin() + m_cameraPathEditKey);
    m_cameraPathEditKey = std::min(m_cameraPathEditKey, int(keyPath.keys.size()) - 1);
  }
  ImGui::EndDisabled();

  // keyframe list (always shown, even when empty; selecting previews the camera)
  if(ImGui::BeginListBox("##keys", ImVec2(-FLT_MIN, ImGui::GetTextLineHeightWithSpacing() * 4.5f)))
  {
    if(hasPath)
    {
      const CameraPath& selPath = m_cameraPaths[m_cameraPathActive];
      for(int i = 0; i < int(selPath.keys.size()); i++)
      {
        const CameraPathKey& k = selPath.keys[i];
        if(ImGui::Selectable(
               fmt::format("{:2d}: eye {:.1f} {:.1f} {:.1f}  fov {:.0f}", i, k.eye.x, k.eye.y, k.eye.z, k.fov).c_str(),
               i == m_cameraPathEditKey))
        {
          m_cameraPathEditKey                  = i;
          m_cameraPathPlayback                 = CAMERA_PATH_STOP;
          nvutils::CameraManipulator::Camera c = cam;
          c.eye = k.eye, c.ctr = k.ctr, c.up = k.up, c.fov = k.fov;
          m_info.cameraManipulator->setCamera(c, false);  // animate for a nicer preview
        }
      }
    }
    ImGui::EndListBox();
  }

  if(!hasPath)
    return;

  CameraPath& path = m_cameraPaths[m_cameraPathActive];

  PE::begin("camerapath", ImGuiTableFlags_Resizable);
  PE::Checkbox("Smooth", &path.smooth, "Catmull-Rom interpolation, otherwise piecewise linear");
  PE::Checkbox("Loop", &path.loop, "Real-time playback wraps around");
  {
    float dur = float(path.duration);
    if(PE::InputFloat("Duration (s)", &dur, 0.1f, 1.0f, "%.2f", 0, "Seconds for a full real-time playback"))
      path.duration = std::max(double(dur), 0.0);
  }
  PE::end();

  // real-time playback + scrub
  const bool active = m_cameraPathPlayback != CAMERA_PATH_STOP;
  if(ImGui::Button(active ? "Stop" : "Play"))
  {
    if(active)
    {
      m_cameraPathPlayback = CAMERA_PATH_STOP;
    }
    else
    {
      m_cameraPathPlayback = CAMERA_PATH_REALTIME;
      m_cameraPathStarted  = false;
      if(m_cameraPathU >= 1.0)
        m_cameraPathU = 0.0;
    }
  }
  ImGui::SameLine();
  if(ImGui::Button("Restart"))
  {
    m_cameraPathU       = 0.0;
    m_cameraPathStarted = false;
  }
  ImGui::SameLine();
  {
    float u = float(m_cameraPathU);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if(ImGui::SliderFloat("##scrub", &u, 0.0f, 1.0f, "t = %.3f"))
    {
      m_cameraPathU        = u;
      m_cameraPathPlayback = CAMERA_PATH_STOP;
      applyCameraPathSample(m_cameraPathU);
    }
  }

  // deterministic (fixed-step) playback, same behaviour as --runcamerapath
  PE::begin("camerapathfixed", ImGuiTableFlags_Resizable);
  PE::InputInt("Fixed frames", &m_cameraPathFrames, 1, 16, 0,
               "Number of frames for a full fixed-step (deterministic) traversal, as used by --runcamerapath");
  PE::end();
  m_cameraPathFrames = std::max(m_cameraPathFrames, 1);
  if(ImGui::Button("Run fixed"))
  {
    m_cameraPathFrame    = 0;
    m_cameraPathPlayback = CAMERA_PATH_FIXED;
  }
  ImGui::SameLine();
  if(m_cameraPathPlayback == CAMERA_PATH_FIXED)
    ImGui::Text("frame %d / %d", m_cameraPathFrame, m_cameraPathFrames);
  else
    ImGui::TextDisabled("cmd: --runcamerapath %d %d", m_cameraPathActive, m_cameraPathFrames);
}

void TessellatedClusters::viewportUI(ImVec2 corner, ImVec2 imageSize)
{
  ImVec2     mouseAbsPos = ImGui::GetMousePos();
  glm::uvec2 mousePos    = {std::max(mouseAbsPos.x - corner.x, 0.0f), std::max(mouseAbsPos.y - corner.y, 0.0f)};

  {
    const Resources::FrameBuffer& fb = m_resources.m_frameBuffer;
    // fixed resolutions are letterboxed, so map through the displayed rect
    glm::vec2  scale                           = fb.fixedSize ?
                                                     glm::vec2(fb.renderSize.width, fb.renderSize.height) / glm::vec2(imageSize.x, imageSize.y) :
                                                     m_resources.getFramebufferWindow2RenderScale();
    glm::uvec2 maxPos                          = {fb.renderSize.width - 1, fb.renderSize.height - 1};
    m_frameConfig.frameConstants.mousePosition = glm::min(glm::uvec2(glm::vec2(mousePos) * scale), maxPos);
  }

  if(m_renderer)
  {
    shaderio::Readback readback;
    m_resources.getReadbackData(readback);

    UsagePercentages pct;
    pct.setupPercentages(readback, m_rendererConfig);

    const char* warning = pct.getWarning();

    if(warning)
    {
      ImVec4 warn_color = {0.75f, 0.2f, 0.2f, 1};
      ImVec4 hi_color   = {0.85f, 0.3f, 0.3f, 1};
      ImVec4 lo_color   = {0, 0, 0, 1};

      ImGui::SetWindowFontScale(2.0);

      // poor man's outline
      ImGui::SetCursorPos({7, 7});
      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({9, 9});
      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({9, 7});
      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({7, 9});
      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({8, 8});
      ImGui::TextColored(hi_color, "%s", warning);

      ImGui::SetWindowFontScale(1.0);
    }
  }
}

void TessellatedClusters::onUIRender()
{
  ImGuiWindow* viewport = ImGui::FindWindowByName("Viewport");

  if(viewport)
  {
    if(nvgui::isWindowHovered(viewport))
    {
      if(ImGui::IsKeyDown(ImGuiKey_R))
      {
        m_reloadShaders = true;
      }
      if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
      {
        m_requestCameraRecenter = true;
      }
    }
  }

  bool earlyOut = !m_scene;

  if(earlyOut)
  {
    return;
  }

  shaderio::Readback readback;
  m_resources.getReadbackData(readback);

  // camera control, recenter
  if(m_requestCameraRecenter && isPickingValid(readback))
  {

    // render pixels, which also holds for letterboxed fixed resolutions
    glm::vec2 mousePos = glm::vec2(m_frameConfig.frameConstants.mousePosition);

    const glm::mat4 view = m_info.cameraManipulator->getViewMatrix();
    const glm::mat4 proj = m_frameConfig.frameConstants.projMatrix;

    float d = decodePickingDepth(readback);

    if(d > 0.0F)  // Ignore infinite
    {
      glm::vec4 win_norm = {0, 0, m_resources.m_frameBuffer.renderSize.width, m_resources.m_frameBuffer.renderSize.height};
      const glm::vec3 hitPos = glm::unProjectZO({mousePos.x, mousePos.y, d}, view, proj, win_norm);

      // Set the interest position
      glm::dvec3 eye, center, up;
      m_info.cameraManipulator->getLookat(eye, center, up);
      m_info.cameraManipulator->setLookat(eye, hitPos, up, false);
    }

    m_requestCameraRecenter = false;
  }

  ImVec4 textColor = ImGui::GetStyleColorVec4(ImGuiCol_Text);
  // for emphasized parameter we want to recommend to the user
  const ImVec4 recommendedColor = ImVec4(0.0, 1.0, 0.0, 1.0);
  // for warnings
  const ImVec4 warnColor = ImVec4(1.0f, 0.7f, 0.3f, 1.0f);

  UsagePercentages pct;
  if(m_renderer)
  {
    pct.setupPercentages(readback, m_rendererConfig);
  }

  ImGui::Begin("Settings");
  ImGui::PushItemWidth(170 * ImGui::GetWindowDpiScale());

  namespace PE = nvgui::PropertyEditor;

  if(ImGui::CollapsingHeader("Scene Modifiers"))  //, nullptr, ImGuiTreeNodeFlags_DefaultOpen ))
  {
    PE::begin("##Scene Complexity");
    PE::Checkbox("Flip faces winding", &m_rendererConfig.flipWinding);
    PE::InputInt("Render grid copies", (int*)&m_tweak.gridCopies, 1, 16, ImGuiInputTextFlags_EnterReturnsTrue,
                 "Instances the entire scene on a grid");
    PE::InputInt("Render grid bits", (int*)&m_tweak.gridConfig, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue,
                 "Instance grid config encoded in 6 bits: 0..2 bit enabled axis, 3..5 bit enabled rotation");
    PE::end();
  }

  if(ImGui::CollapsingHeader("Rendering", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Rendering");
    PE::entry("Renderer", [&]() { return m_ui.enumCombobox(GUI_RENDERER, "renderer", &m_tweak.renderer); });
    PE::entry("Super sampling",
              [&]() { return m_ui.enumCombobox(GUI_SUPERSAMPLE, "##Supersample", &m_tweak.supersample); });
#if USE_DLSS
    bool        dlssAvailable = false;
    const char* dlssLabel     = "DLSS";
    if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_TESS)
    {
      // NGX is only initialized once DLSS is requested
      dlssAvailable = !m_resources.m_frameBuffer.dlssDenoiser.isInitialized()
                      || m_resources.m_frameBuffer.dlssDenoiser.isAvailable();
      dlssLabel = "DLSS - RR";
    }
    else if(m_tweak.renderer == RENDERER_RASTER_CLUSTERS_TESS)
    {
      dlssAvailable = !m_resources.m_frameBuffer.dlssUpscaler.isInitialized()
                      || m_resources.m_frameBuffer.dlssUpscaler.isAvailable();
      dlssLabel = "DLSS - SR";
    }
    // no changes while NGX loads in the background, DLSS is switched on once done
    const bool  dlssLoading = m_resources.isDlssLoading();
    std::string dlssEntry   = dlssLoading ? fmt::format("{} (loading)", dlssLabel) : dlssLabel;
    ImGui::BeginDisabled(!dlssAvailable || dlssLoading);
    {
      PE::entry(dlssEntry, [&]() {
        bool changed = ImGui::Checkbox("Enabled", &m_rendererConfig.useDlss);
        ImGui::SameLine();

        const char* labels[] = {"DLAA", "Quality", "Balanced", "Performance", "Ultra Performance"};
        const NVSDK_NGX_PerfQuality_Value values[] = {
            NVSDK_NGX_PerfQuality_Value_DLAA,
            NVSDK_NGX_PerfQuality_Value_MaxQuality,
            NVSDK_NGX_PerfQuality_Value_Balanced,
            NVSDK_NGX_PerfQuality_Value_MaxPerf,
            NVSDK_NGX_PerfQuality_Value_UltraPerformance,
        };

        int current = 1;
        for(int i = 0; i < IM_ARRAYSIZE(values); i++)
        {
          if(m_rendererConfig.dlssQuality == values[i])
          {
            current = i;
            break;
          }
        }
        if(ImGui::Combo("Quality", &current, labels, IM_ARRAYSIZE(labels)))
        {
          m_rendererConfig.dlssQuality = values[current];
          changed                      = true;
        }
        return changed;
      });
    }
    ImGui::EndDisabled();
#endif
    PE::Text("Render Resolution:", "%d x %d", m_resources.m_frameBuffer.renderSize.width,
             m_resources.m_frameBuffer.renderSize.height);

    ImGui::BeginDisabled(m_tweak.tessRatePixels != 0);
    m_tweak.facetShading = m_tweak.facetShading || (m_tweak.tessRatePixels != 0);
    PE::Checkbox("Facet shading", &m_tweak.facetShading, "Forced to enabled if tesselation is on (see readme.md)");
    ImGui::EndDisabled();
    PE::Checkbox("Wireframe", (bool*)&m_frameConfig.frameConstants.doWireframe);

    // conditional UI, declutters the UI, prevents presenting many sections in disabled state
    if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_TESS)
    {
      PE::Checkbox("Cast shadow rays", (bool*)&m_frameConfig.frameConstants.doShadow);
      PE::SliderFloat("Ambient occlusion radius", &m_frameConfig.frameConstants.ambientOcclusionRadius, 0.001f, 1.f);
      PE::SliderInt("Ambient occlusion rays", &m_frameConfig.frameConstants.ambientOcclusionSamples, 0, 64);
    }
    if(m_tweak.renderer == RENDERER_RASTER_CLUSTERS_TESS)
    {
      PE::Checkbox("Ambient occlusion (HBAO)", &m_tweak.hbaoActive);
      if(PE::treeNode("HBAO settings"))
      {
        PE::Checkbox("Blur", &m_frameConfig.hbaoSettings.blur);
        PE::Checkbox("Quarter resolution", &m_tweak.hbaoQuarter, "Half width and height, upsampled depth-aware");
        PE::InputFloat("Radius", &m_tweak.hbaoRadius, 0.01f);
        PE::InputFloat("Sharpness", &m_frameConfig.hbaoSettings.powerExponent, 1.0f);
        PE::InputFloat("Intensity", &m_frameConfig.hbaoSettings.intensity, 0.1f);
        PE::InputFloat("Bias", &m_frameConfig.hbaoSettings.bias, 0.01f);
        PE::treePop();
      }
    }
    PE::end();
    PE::begin("##RenderingSpecific");

    ImGui::PushStyleColor(ImGuiCol_Text, recommendedColor);
    PE::entry("Visualize", [&]() {
      ImGui::PopStyleColor();  // pop text color here so it only applies to the label
      return m_ui.enumCombobox(GUI_VISUALIZE, "visualize", &m_tweak.visualizeMode);
    });

    PE::InputIntClamped("Max visible clusters in bits", (int*)&m_rendererConfig.numVisibleClusterBits, 16, 25, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue);

    PE::Checkbox("Culling (Occlusion & Frustum)", &m_rendererConfig.doCulling);

    PE::Checkbox("Freeze Cull / LoD", &m_frameConfig.freezeCulling);
    PE::Checkbox("Render statistics", &m_rendererConfig.useRenderStats,
                 "Adds additional atomic counters for statistics, impacts performance");
    PE::end();

    if(ImGui::BeginTable("Rendering stats", 3, ImGuiTableFlags_BordersOuter))
    {
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 150.0f);
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      //ImGui::TableHeadersRow(); // we do not show the header, it is not usefull
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Visible Clusters");
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctVisible > 100 ? warnColor : textColor, "%s (%d%%)",
                         formatMetric(readback.numVisibleClusters).c_str(), pct.pctVisible);
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctVisible > 100 ? warnColor : textColor, "%d", readback.numVisibleClusters);
      if(m_rendererConfig.useRenderStats)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Rendered triangles");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(readback.numTotalTriangles).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%d", readback.numTotalTriangles);
      }
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Rendered clusters");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(readback.numBlasClusters).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%d", readback.numBlasClusters);
      if(m_rendererConfig.useRenderStats)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Rendered tri/cluster");
        ImGui::TableNextColumn();
        ImGui::Text("%.1f", float(readback.numTotalTriangles) / float(readback.numBlasClusters));
        ImGui::TableNextColumn();
        ImGui::Text("%s", "");
      }
      ImGui::TableNextRow();
      ImGui::EndTable();
    }
  }

  if(m_renderer && m_tweak.renderer == RENDERER_RASTER_CLUSTERS_TESS
     && ImGui::CollapsingHeader("Rasterization", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Rasterization");
    PE::Checkbox("Batch meshlets", &m_rendererConfig.rasterBatchMeshlets);
    ImGui::BeginDisabled(!m_resources.m_supportsMeshShaderNV);
    PE::Checkbox("Use EXT mesh shader", &m_rendererConfig.useEXTmeshShader,
                 "Use VK_EXT_mesh_shader instead of VK_NV_mesh_shader, forced on without NV support.");
    ImGui::EndDisabled();
    PE::end();
  }

  if(m_renderer && m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_TESS
     && ImGui::CollapsingHeader("Ray Tracing", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Raytracing");
    PE::entry(
        "Low tessellation mode",
        [&]() { return m_ui.enumCombobox(GUI_LOWTESSMODE, "##LowTessMode", &m_rendererConfig.lowTessMode); },
        "How clusters/triangles with low tessellation factors are turned into CLAS.\n"
        "Transient builds create a CLAS from scratch, combined templates instantiate\n"
        "pre-built templates that store a batch of independent triangles / 2x2x2 patches.");
    PE::Checkbox("Optimize == 1x1x1", &m_rendererConfig.lowTessClusters1X);
    PE::Checkbox("Optimize <= 2x2x2", &m_rendererConfig.lowTessClusters2X);
    if(m_rendererConfig.lowTessMode == LOWTESS_MODE_COMBINED_TEMPLATES)
    {
      PE::InputIntClamped("Combined 1x1x1 batch size", (int*)&m_rendererConfig.combined1xBatchSize, 1, 85, 1, 8,
                          ImGuiInputTextFlags_EnterReturnsTrue,
                          "Number of independent triangles a combined template stores at most.");
      PE::InputIntClamped("Combined 2x2x2 batch size", (int*)&m_rendererConfig.combined2xBatchSize, 1, 32, 1, 2,
                          ImGuiInputTextFlags_EnterReturnsTrue,
                          "Number of independent 2x2x2 patches a combined template stores at most (rounded down to a power of two).");
    }
    PE::InputIntClamped("Max CLAS data in MB", (int*)&m_rendererConfig.numGeneratedClusterMegs, 16, 3 * 1024, 16, 128,
                        ImGuiInputTextFlags_EnterReturnsTrue);
    PE::InputIntClamped("Max CLAS vertices in bits", (int*)&m_rendererConfig.numGeneratedVerticesBits, 15, 27, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue);
    PE::end();

    if(ImGui::BeginTable("RT stats", 3, ImGuiTableFlags_BordersOuter))
    {
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 150.0f);
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      //ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Template instantiations");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(readback.numTempInstantiations).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%d", readback.numTempInstantiations);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Transient clusters");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(readback.numTransBuilds).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%d", readback.numTransBuilds);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("CLAS vertex count");
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctVertices > 100 ? warnColor : textColor, "%s", formatMetric(readback.numGenVertices).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctVertices > 100 ? warnColor : textColor, "%d%%", pct.pctVertices);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("CLAS vertex bytes");
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctVertices > 100 ? warnColor : textColor, "%s",
                         formatMemorySize(sizeof(glm::vec3) * readback.numGenVertices).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctVertices > 100 ? warnColor : textColor, "%d%%", pct.pctVertices);
      ImGui::EndTable();
    }
  }

  if(m_renderer && ImGui::CollapsingHeader("Tessellation", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Tessellation");
    ImGui::PushStyleColor(ImGuiCol_Text, recommendedColor);
    PE::InputIntClamped("Pixels per segment", (int*)&m_tweak.tessRatePixels, 0, 128, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue, "Set to 0 to disable tessellation");
    ImGui::PopStyleColor();
    PE::InputIntClamped("Max split factor", (int*)&m_rendererConfig.splitFactor, 2, TESSTABLE_SIZE, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue);
    PE::Checkbox("PN-Triangles displacement", &m_rendererConfig.pnDisplacement);
    PE::InputIntClamped("Max part triangles in bits", (int*)&m_rendererConfig.numPartTriangleBits, 4, 24, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue);
    PE::InputIntClamped("Max split triangles in bits", (int*)&m_rendererConfig.numSplitTriangleBits, 16, 24, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue);
    PE::Checkbox("Persistent Split Kernel", (bool*)&m_rendererConfig.persistentKernel);
    PE::end();

    if(ImGui::BeginTable("Tessellation stats", 3, ImGuiTableFlags_BordersOuter))
    {
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 150.0f);
      ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Percentage", ImGuiTableColumnFlags_WidthStretch);
      //ImGui::TableHeadersRow(); // we do not show the header, it is not visually usefull
      // ray tracing only counts full clusters as part of the render stats
      if(m_tweak.renderer != RENDERER_RAYTRACE_CLUSTERS_TESS || m_rendererConfig.useRenderStats)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Full clusters");
        ImGui::TableNextColumn();
        ImGui::TextColored(pct.pctFull > 100 ? warnColor : textColor, "%d", readback.numFullClusters);
        ImGui::TableNextColumn();
        ImGui::TextColored(pct.pctFull > 100 ? warnColor : textColor, "%d%%", pct.pctFull);
      }
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Part triangles");
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctPart > 100 ? warnColor : textColor, "%d", readback.numPartTriangles);
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctPart > 100 ? warnColor : textColor, "%d%%", pct.pctPart);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Split triangles");
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctSplit > 100 ? warnColor : textColor, "%d", readback.numSplitTriangles);
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctSplit > 100 ? warnColor : textColor, "%d%%", pct.pctSplit);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::EndTable();
    }
  }

  if(ImGui::CollapsingHeader("Animation & Displacement", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Animation");

    ImGui::PushStyleColor(ImGuiCol_Text, recommendedColor);
    PE::Checkbox("Enable animation", &m_rendererConfig.doAnimation);
    ImGui::PopStyleColor();

    ImGui::BeginDisabled(!m_rendererConfig.doAnimation);
    PE::SliderFloat("Override time value", &m_tweak.overrideTime, 0, 10.0f, "%0.3f", 0, "Set to 0 disables override");
    PE::SliderFloat("Ripple frequency", &m_frameConfig.frameConstants.animationRippleFrequency, 0.001f, 200.f);
    float amplitude = m_frameConfig.frameConstants.animationRippleAmplitude * 100;
    PE::SliderFloat("Ripple amplitude", &amplitude, 0.f, 2.0f, "%.3f");
    m_frameConfig.frameConstants.animationRippleAmplitude = amplitude * 0.01f;
    PE::SliderFloat("Ripple speed", &m_frameConfig.frameConstants.animationRippleSpeed, 0.f, 10.f);
    ImGui::EndDisabled();
    if(m_scene && !m_scene->m_textureImages.empty())
    {
      PE::InputFloat("Displacement scale", &m_frameConfig.frameConstants.displacementScale, 0.01f, 1.0f, "%.5f",
                     ImGuiInputTextFlags_EnterReturnsTrue);
      PE::InputFloat("Displacement offset", &m_frameConfig.frameConstants.displacementOffset, 0.01f, 1.0f, "%.5f",
                     ImGuiInputTextFlags_EnterReturnsTrue);
    }
    PE::end();
  }

  if(ImGui::CollapsingHeader("Clusters & CLAS", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Clusters");
    PE::entry("Cluster/meshlet size",
              [&]() { return m_ui.enumCombobox(GUI_MESHLET, "##ClusterSize", &m_tweak.clusterConfig); });
    PE::InputIntClamped("CLAS Mantissa drop bits", (int*)&m_rendererConfig.positionTruncateBits, 0, 22, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue);
    PE::entry(
        "Transient CLAS build mode",
        [&]() { return m_ui.enumCombobox(GUI_BUILDMODE, "##BuildMode", &m_tweak.clusterBuildMode); }, "Transient CLAS build mode");
    PE::end();
  }

  ImGui::End();

  ImGui::Begin("Statistics");

  if(ImGui::CollapsingHeader("Scene", nullptr, ImGuiTreeNodeFlags_DefaultOpen) && m_renderer)
  {
    if(ImGui::BeginTable("Scene stats", 3, ImGuiTableFlags_None))
    {
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Scene", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Triangles");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_numTriangles * m_tweak.gridCopies).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_numTriangles).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Clusters");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_numClusters * m_tweak.gridCopies).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_numClusters).c_str());
      ImGui::EndTable();
    }
  }

  if(ImGui::CollapsingHeader("Memory", nullptr, ImGuiTreeNodeFlags_DefaultOpen) && m_renderer)
  {

    Renderer::ResourceUsageInfo resourceReserved = m_renderer->getResourceUsage(true);
    Renderer::ResourceUsageInfo resourceActual   = m_renderer->getResourceUsage(false);

    if(ImGui::BeginTable("Memory stats", 3, ImGuiTableFlags_RowBg))
    {
      ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Actual", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Reserved", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Geometry");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceActual.geometryMemBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("==");
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Template");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceActual.rtTemplateMemBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("==");
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("TLAS");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceActual.rtTlasMemBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("==");
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("BLAS");
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctBlas > 100 ? warnColor : textColor, "%s (%d%% used)",
                         formatMemorySize(resourceActual.rtBlasMemBytes).c_str(), pct.pctBlas);
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctBlas > 100 ? warnColor : textColor, "%s",
                         formatMemorySize(resourceReserved.rtBlasMemBytes).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("CLAS");
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctClusterActual > 100 ? warnColor : textColor, "%s (%d%% used)",
                         formatMemorySize(resourceActual.rtClasMemBytes).c_str(), pct.pctClusterActual);
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctClusterReserved > 100 ? warnColor : textColor, "%s (%d%% requested)",
                         formatMemorySize(resourceReserved.rtClasMemBytes).c_str(), pct.pctClusterReserved);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Operations");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceActual.operationsMemBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("==");
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::EndTable();
    }
  }

  if(m_renderer && ImGui::CollapsingHeader("Generated Clusters"))
  {
    // histogram[i] holds the number of generated clusters that have i triangles.
    // scan for the tallest bin (used to scale the plot's y-axis), the highest occupied
    // bin (the largest triangle count that actually occurred) and the total count.
    uint32_t maxValue     = 0;  // largest number of clusters found in any single bin
    uint32_t maxTriangles = 0;  // highest per-cluster triangle count that occurred
    uint32_t count        = 0;  // total number of generated clusters
    for(uint32_t i = 0; i < std::size(readback.histogram); ++i)
    {
      uint32_t bin = readback.histogram[i];
      count += bin;
      if(bin > 0)
        maxTriangles = i;
      if(bin > maxValue)
        maxValue = bin;
    }

    ImGui::Text("Cluster count: %d", count);
    ImGui::Text("Cluster max triangles: %d", maxTriangles);

    // limit the plotted range to the largest triangle count a cluster can reach:
    // the peak of the model's base cluster size and the tessellation table's max,
    // plus one because histogram[i] holds clusters with exactly i triangles.
    uint32_t plotTriangles =
        std::max(m_scene ? m_scene->m_maxClusterTriangles : 0u, m_renderer->getTessellationMaxTriangles()) + 1;
    plotTriangles = std::min(plotTriangles, uint32_t(std::size(readback.histogram)));

    uiPlot<uint32_t>(std::string("G.Cluster Triangle Histogram"), std::string("Cluster count with %d triangles: %d"),
                     std::span<const uint32_t>(readback.histogram, plotTriangles), maxValue);

    m_rendererConfig.generateHistogram = true;
  }
  else
  {
    m_rendererConfig.generateHistogram = false;
  }

  if(m_scene && ImGui::CollapsingHeader("Model Clusters"))
  {
    ImGui::Text("Cluster max triangles: %d", m_scene->m_maxClusterTriangles);
    ImGui::Text("Cluster max vertices: %d", m_scene->m_maxClusterVertices);
    ImGui::Text("Cluster count: %d", m_scene->m_numClusters);
    ImGui::Text("Clusters with config (%d) triangles: %d (%.1f%%)", m_scene->m_config.clusterTriangles,
                m_scene->m_clusterTriangleHistogram.back(),
                float(m_scene->m_clusterTriangleHistogram.back()) * 100.f / float(m_scene->m_numClusters));

    uiPlot<uint32_t>(std::string("M.Cluster Triangle Histogram"), std::string("Cluster count with %d triangles: %d"),
                     m_scene->m_clusterTriangleHistogram, m_scene->m_clusterTriangleHistogramMax);
    uiPlot<uint32_t>(std::string("M.Cluster Vertex Histogram"), std::string("Cluster count with %d vertices: %d"),
                     m_scene->m_clusterVertexHistogram, m_scene->m_clusterVertexHistogramMax);
  }
  ImGui::End();

  ImGui::Begin("Misc Settings");

  if(ImGui::CollapsingHeader("Camera", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    nvgui::CameraWidget(m_info.cameraManipulator);
  }
  if(ImGui::CollapsingHeader("Camera Paths", nullptr))
  {
    cameraPathUI();
  }

  if(ImGui::CollapsingHeader("Lighting", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    namespace PE = nvgui::PropertyEditor;
    PE::begin();
    PE::SliderFloat("Light Mixer", &m_frameConfig.frameConstants.lightMixer, 0.0f, 1.0f, "%.3f", 0,
                    "Mix between flashlight and sun light");
    PE::end();
    ImGui::TextDisabled("Sun & Sky");
    nvgui::skySimpleParametersUI(m_frameConfig.frameConstants.skyParams);
  }

  ImGui::End();

  if(m_debugUI)
  {
    ImGui::Begin("Debug");
    if(ImGui::CollapsingHeader("Misc settings", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      PE::begin("##DebugMisc");
      PE::InputInt("Colorize xor", (int*)&m_frameConfig.frameConstants.colorXor);
      PE::Checkbox("Auto reset timer", &m_tweak.autoResetTimers);
      PE::InputInt("Persistent threads", (int*)&m_rendererConfig.persistentThreads, 1, 128, ImGuiInputTextFlags_EnterReturnsTrue);
      PE::end();
    }

    if(ImGui::CollapsingHeader("Debug Shader Values", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      PE::begin("##DebugShader");
      PE::InputInt("dbgInt", (int*)&m_frameConfig.frameConstants.dbgUint, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue);
      PE::InputFloat("dbgFloat", &m_frameConfig.frameConstants.dbgFloat, 0.1f, 1.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
      PE::end();

      ImGui::Text(" debugI :  %10d", readback.debugI);
      ImGui::Text(" debugUI:  %10u", readback.debugUI);
      ImGui::Text(" debugU64:  %llX", readback.debugU64);
      static bool debugFloat  = false;
      static bool debugHex    = false;
      static bool debugAll    = false;
      static bool debugSingle = false;
      ImGui::Checkbox(" as float", &debugFloat);
      ImGui::SameLine();
      ImGui::Checkbox("hex", &debugHex);
      ImGui::SameLine();
      ImGui::Checkbox("all", &debugAll);
      ImGui::SameLine();
      ImGui::Checkbox("single", &debugSingle);
      ImGui::SameLine();
      bool doPrint = ImGui::Button("print");

      // renders right-justified text into the current table cell
      auto textRight = [](const char* fmt, ...) {
        char    buf[64];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        float textWidth  = ImGui::CalcTextSize(buf).x;
        float availwidth = ImGui::GetContentRegionAvail().x;
        if(availwidth > textWidth)
          ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (availwidth - textWidth));
        ImGui::TextUnformatted(buf);
      };

      if(debugSingle)
      {
        // debugA, debugB and debugC are contiguous in memory, so indexing debugA
        // beyond its bounds reads them as one flat array.
        uint32_t count = debugAll ? 128 : 64;

        if(ImGui::BeginTable("debugValues", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame))
        {
          ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed);
          ImGui::TableSetupColumn("value");
          ImGui::TableHeadersRow();

          for(uint32_t i = 0; i < count; i++)
          {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            textRight("%3d", i);
            ImGui::TableNextColumn();
            if(debugFloat)
            {
              textRight("%f", *(float*)&readback.debugA[i]);
              if(doPrint)
                LOGI("%3d; %f;\n", i, *(float*)&readback.debugA[i]);
            }
            else if(debugHex)
            {
              textRight("%8X", readback.debugA[i]);
              if(doPrint)
                LOGI("%3d; %8X;\n", i, readback.debugA[i]);
            }
            else
            {
              textRight("%10u", readback.debugA[i]);
              if(doPrint)
                LOGI("%3d; %10u;\n", i, readback.debugA[i]);
            }
          }
          ImGui::EndTable();
        }
      }
      else
      {
        uint32_t count = debugAll ? 64 : 32;

        if(ImGui::BeginTable("debugValues", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame))
        {
          ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed);
          ImGui::TableSetupColumn("A");
          ImGui::TableSetupColumn("B");
          ImGui::TableSetupColumn("C");
          ImGui::TableHeadersRow();

          for(uint32_t i = 0; i < count; i++)
          {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            textRight("%2d", i);
            if(debugFloat)
            {
              ImGui::TableNextColumn();
              textRight("%f", *(float*)&readback.debugA[i]);
              ImGui::TableNextColumn();
              textRight("%f", *(float*)&readback.debugB[i]);
              ImGui::TableNextColumn();
              textRight("%f", *(float*)&readback.debugC[i]);
              if(doPrint)
                LOGI("%2d; %f; %f; %f;\n", i, *(float*)&readback.debugA[i], *(float*)&readback.debugB[i],
                     *(float*)&readback.debugC[i]);
            }
            else if(debugHex)
            {
              ImGui::TableNextColumn();
              textRight("%8X", readback.debugA[i]);
              ImGui::TableNextColumn();
              textRight("%8X", readback.debugB[i]);
              ImGui::TableNextColumn();
              textRight("%8X", readback.debugC[i]);
              if(doPrint)
                LOGI("%2d; %8X; %8X; %8X;\n", i, readback.debugA[i], readback.debugB[i], readback.debugC[i]);
            }
            else
            {
              ImGui::TableNextColumn();
              textRight("%10u", readback.debugA[i]);
              ImGui::TableNextColumn();
              textRight("%10u", readback.debugB[i]);
              ImGui::TableNextColumn();
              textRight("%10u", readback.debugC[i]);
              if(doPrint)
                LOGI("%2d; %10u; %10u; %10u;\n", i, readback.debugA[i], readback.debugB[i], readback.debugC[i]);
            }
          }
          ImGui::EndTable();
        }
      }
    }
    ImGui::End();
  }

  handleChanges();

  // Rendered image displayed fully in 'Viewport' window
  ImGui::Begin("Viewport");
  ImVec2                        corner    = ImGui::GetCursorScreenPos();  // Corner of the viewport
  ImVec2                        avail     = ImGui::GetContentRegionAvail();
  ImVec2                        imageSize = avail;
  const Resources::FrameBuffer& fb        = m_resources.m_frameBuffer;
  if(fb.fixedSize && fb.targetSize.width && fb.targetSize.height)
  {
    // fixed resolution: fit its aspect into the viewport, centered
    float scale = std::min(avail.x / float(fb.targetSize.width), avail.y / float(fb.targetSize.height));
    imageSize   = {float(fb.targetSize.width) * scale, float(fb.targetSize.height) * scale};
    corner.x += std::max(0.0f, (avail.x - imageSize.x) * 0.5f);
    corner.y += std::max(0.0f, (avail.y - imageSize.y) * 0.5f);
    ImGui::SetCursorScreenPos(corner);
  }
  ImGui::Image((ImTextureID)m_imguiTexture, imageSize);
  viewportUI(corner, imageSize);
  ImGui::End();
}

void TessellatedClusters::onUIMenu()
{
  if(ImGui::BeginMenu("File"))
  {
    if(ImGui::MenuItem("Open", "Ctrl+O"))
    {
      std::filesystem::path filename =
          nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load glTF", "glTF(.gltf, .glb)|*.gltf;*.glb");
      if(!filename.empty())
      {
        onFileDrop(filename);
      }
    }
    ImGui::EndMenu();
  }
}
}  // namespace tessellatedclusters
