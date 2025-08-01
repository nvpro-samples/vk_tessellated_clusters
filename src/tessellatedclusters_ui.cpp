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

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <implot/implot.h>
#include <nvgui/camera.hpp>
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
void uiPlot(std::string plotName, std::string tooltipFormat, const std::vector<T>& data, const T& maxValue)
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

    ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
    ImPlot::SetNextFillStyle(plotColor);
    ImPlot::PlotShaded("", data.data(), (int)data.size(), -INFINITY, 1.0, 0.0, 0, 0);
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

    pctInst            = getUsagePct(readback.numBlasClusters,
                                     (1ull << rendererConfig.numVisibleClusterBits) + (1ull << rendererConfig.numPartTriangleBits));
    pctClusterReserved = getUsagePct(readback.numGenDatas, rendererConfig.numGeneratedClusterMegs * 1024 * 1024);
    pctClusterActual   = getUsagePct(readback.numGenActualDatas, rendererConfig.numGeneratedClusterMegs * 1024 * 1024);

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

void TessellatedClusters::viewportUI(ImVec2 corner)
{
  ImVec2     mouseAbsPos = ImGui::GetMousePos();
  glm::uvec2 mousePos    = {mouseAbsPos.x - corner.x, mouseAbsPos.y - corner.y};

  m_frameConfig.frameConstants.mousePosition = mousePos * glm::uvec2(m_tweak.supersample, m_tweak.supersample);

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

    glm::uvec2 mousePos = {m_frameConfig.frameConstants.mousePosition.x / m_tweak.supersample,
                           m_frameConfig.frameConstants.mousePosition.y / m_tweak.supersample};

    const glm::mat4 view = m_info.cameraManipulator->getViewMatrix();
    const glm::mat4 proj = m_frameConfig.frameConstants.projMatrix;

    float d = decodePickingDepth(readback);

    if(d < 1.0F)  // Ignore infinite
    {
      glm::vec4       win_norm = {0, 0, m_frameConfig.frameConstants.viewport.x / m_tweak.supersample,
                                  m_frameConfig.frameConstants.viewport.y / m_tweak.supersample};
      const glm::vec3 hitPos   = glm::unProjectZO({mousePos.x, mousePos.y, d}, view, proj, win_norm);

      // Set the interest position
      glm::vec3 eye, center, up;
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
    PE::entry("Super sampling", [&]() { return m_ui.enumCombobox(GUI_SUPERSAMPLE, "##HiddenID", &m_tweak.supersample); });
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
        PE::Checkbox("Full resolution", &m_tweak.hbaoFullRes);
        PE::InputFloat("Radius", &m_tweak.hbaoRadius, 0.01f);
        PE::InputFloat("Blur sharpness", &m_frameConfig.hbaoSettings.blurSharpness, 1.0f);
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
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Rendered triangles");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(readback.numTotalTriangles).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%d", readback.numTotalTriangles);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Rendered clusters");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(readback.numBlasClusters).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%d", readback.numBlasClusters);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Rendered tri/cluster");
      ImGui::TableNextColumn();
      ImGui::Text("%.1f", float(readback.numTotalTriangles) / float(readback.numBlasClusters));
      ImGui::TableNextColumn();
      ImGui::Text("%s", "");
      ImGui::TableNextRow();
      ImGui::EndTable();
    }
  }

  if(m_renderer && m_tweak.renderer == RENDERER_RASTER_CLUSTERS_TESS
     && ImGui::CollapsingHeader("Rasterization", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Rasterization");
    PE::Checkbox("Batch meshlets", &m_rendererConfig.rasterBatchMeshlets);
    PE::end();
  }

  if(m_renderer && m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_TESS
     && ImGui::CollapsingHeader("Ray Tracing", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Raytracing");
    PE::Checkbox("Use transient == 1x1x1", &m_rendererConfig.transientClusters1X);
    PE::Checkbox("Use transient <= 2x2x2", &m_rendererConfig.transientClusters2X);
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
    PE::Checkbox("PN-Triangles displacement", &m_rendererConfig.pnDisplacement);
    PE::InputIntClamped("Max part triangles in bits", (int*)&m_rendererConfig.numPartTriangleBits, 4, 24, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue);
    PE::InputIntClamped("Max split triangles in bits", (int*)&m_rendererConfig.numSplitTriangleBits, 16, 24, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue);
    PE::end();

    if(ImGui::BeginTable("Tessellation stats", 3, ImGuiTableFlags_BordersOuter))
    {
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 150.0f);
      ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Percentage", ImGuiTableColumnFlags_WidthStretch);
      //ImGui::TableHeadersRow(); // we do not show the header, it is not visually usefull
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Full clusters");
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctFull > 100 ? warnColor : textColor, "%d", readback.numFullClusters);
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctFull > 100 ? warnColor : textColor, "%d%%", pct.pctFull);
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
              [&]() { return m_ui.enumCombobox(GUI_MESHLET, "##HiddenID", &m_tweak.clusterConfig); });
    PE::Checkbox("Use NV cluster library", &m_sceneConfig.clusterNvLibrary,
                 "uses the nv_cluster_builder library, otherwise meshoptimizer");
    PE::Checkbox("Optimize for triangle strips", &m_sceneConfig.clusterStripify,
                 "Re-order triangles within cluster optimizing for triangle strips");
    PE::InputIntClamped("CLAS Mantissa drop bits", (int*)&m_rendererConfig.positionTruncateBits, 0, 22, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue);
    PE::entry(
        "Transient CLAS build mode",
        [&]() { return m_ui.enumCombobox(GUI_BUILDMODE, "##HiddenID", &m_tweak.clusterBuildMode); }, "Transient CLAS build mode");
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

  if(m_scene && ImGui::CollapsingHeader("Model Clusters"))
  {
    ImGui::Text("Cluster max triangles: %d", m_scene->m_maxClusterTriangles);
    ImGui::Text("Cluster max vertices: %d", m_scene->m_maxClusterVertices);
    ImGui::Text("Cluster count: %d", m_scene->m_numClusters);
    ImGui::Text("Clusters with config (%d) triangles: %d (%.1f%%)", m_scene->m_config.clusterTriangles,
                m_scene->m_clusterTriangleHistogram.back(),
                float(m_scene->m_clusterTriangleHistogram.back()) * 100.f / float(m_scene->m_numClusters));

    uiPlot(std::string("Cluster Triangle Histogram"), std::string("Cluster count with %d triangles: %d"),
           m_scene->m_clusterTriangleHistogram, m_scene->m_clusterTriangleHistogramMax);
    uiPlot(std::string("Cluster Vertex Histogram"), std::string("Cluster count with %d vertices: %d"),
           m_scene->m_clusterVertexHistogram, m_scene->m_clusterVertexHistogramMax);
  }
  ImGui::End();

  ImGui::Begin("Misc Settings");

  if(ImGui::CollapsingHeader("Camera", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    nvgui::CameraWidget(m_info.cameraManipulator);
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

#ifdef _DEBUG
  ImGui::Begin("Debug");
  if(ImGui::CollapsingHeader("Misc settings", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##HiddenID");
    PE::InputInt("Colorize xor", (int*)&m_frameConfig.frameConstants.colorXor);
    PE::Checkbox("Auto reset timer", &m_tweak.autoResetTimers);
    PE::InputInt("Persistent threads", (int*)&m_rendererConfig.persistentThreads, 1, 128, ImGuiInputTextFlags_EnterReturnsTrue);
    PE::end();
  }

  if(ImGui::CollapsingHeader("Debug Shader Values", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##HiddenID");
    PE::InputInt("dbgInt", (int*)&m_frameConfig.frameConstants.dbgUint, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue);
    PE::InputFloat("dbgFloat", &m_frameConfig.frameConstants.dbgFloat, 0.1f, 1.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
    PE::end();

    ImGui::Text(" debugI :  %10d", readback.debugI);
    ImGui::Text(" debugUI:  %10u", readback.debugUI);
    ImGui::Text(" debugU64:  %llX", readback.debugU64);
    static bool debugFloat = false;
    static bool debugHex   = false;
    static bool debugAll   = false;
    ImGui::Checkbox(" as float", &debugFloat);
    ImGui::SameLine();
    ImGui::Checkbox("hex", &debugHex);
    ImGui::SameLine();
    ImGui::Checkbox("all", &debugAll);
    ImGui::SameLine();
    bool     doPrint = ImGui::Button("print");
    uint32_t count   = debugAll ? 64 : 32;
    if(debugFloat)
    {
      for(uint32_t i = 0; i < count; i++)
      {
        ImGui::Text("%2d: %f %f %f", i, *(float*)&readback.debugA[i], *(float*)&readback.debugB[i], *(float*)&readback.debugC[i]);
        if(doPrint)
          LOGI("%2d; %f; %f; %f;\n", i, *(float*)&readback.debugA[i], *(float*)&readback.debugB[i], *(float*)&readback.debugC[i]);
      }
    }
    else if(debugHex)
    {
      for(uint32_t i = 0; i < count; i++)
      {
        ImGui::Text("%2d: %8X %8X %8X", i, readback.debugA[i], readback.debugB[i], readback.debugC[i]);
        if(doPrint)
          LOGI("%2d; %8X; %8X; %8X;\n", i, readback.debugA[i], readback.debugB[i], readback.debugC[i]);
      }
    }
    else
    {
      for(uint32_t i = 0; i < count; i++)
      {
        ImGui::Text("%2d: %10u %10u %10u", i, readback.debugA[i], readback.debugB[i], readback.debugC[i]);
        if(doPrint)
          LOGI("%2d; %10u; %10u; %10u;\n", i, readback.debugA[i], readback.debugB[i], readback.debugC[i]);
      }
    }
  }
  ImGui::End();
#endif

  handleChanges();

  // Rendered image displayed fully in 'Viewport' window
  ImGui::Begin("Viewport");
  ImVec2 corner = ImGui::GetCursorScreenPos();  // Corner of the viewport
  ImGui::Image((ImTextureID)m_imguiTexture, ImGui::GetContentRegionAvail());
  viewportUI(corner);
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
