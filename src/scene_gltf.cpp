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

#include <float.h>

#include <glm/gtc/type_ptr.hpp>
#include <dlib/dlib_url.h>
#include <cgltf.h>
#include <nvutils/file_mapping.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/parallel_work.hpp>

#include "scene.hpp"

namespace {
struct FileMappingList
{
  struct Entry
  {
    nvutils::FileReadMapping mapping;
    int64_t                  refCount = 1;
  };
  std::unordered_map<std::string, Entry>       m_nameToMapping;
  std::unordered_map<const void*, std::string> m_dataToName;
#ifdef _DEBUG
  int64_t m_openBias = 0;
#endif

  bool open(const char* path, size_t* size, void** data)
  {
#ifdef _DEBUG
    m_openBias++;
#endif

    std::string pathStr(path);

    auto it = m_nameToMapping.find(pathStr);
    if(it != m_nameToMapping.end())
    {
      *data = const_cast<void*>(it->second.mapping.data());
      *size = it->second.mapping.size();
      it->second.refCount++;
      return true;
    }

    Entry entry;
    if(entry.mapping.open(path))
    {
      const void* mappingData = entry.mapping.data();
      *data                   = const_cast<void*>(mappingData);
      *size                   = entry.mapping.size();
      m_dataToName.insert({mappingData, pathStr});
      m_nameToMapping.insert({pathStr, std::move(entry)});
      return true;
    }

    return false;
  }

  void close(void* data)
  {
#ifdef _DEBUG
    m_openBias--;
#endif
    auto itName = m_dataToName.find(data);
    if(itName != m_dataToName.end())
    {
      auto itMapping = m_nameToMapping.find(itName->second);
      if(itMapping != m_nameToMapping.end())
      {
        itMapping->second.refCount--;

        if(!itMapping->second.refCount)
        {
          m_nameToMapping.erase(itMapping);
          m_dataToName.erase(itName);
        }
      }
    }
  }

  ~FileMappingList()
  {
#ifdef _DEBUG
    assert(m_openBias == 0 && "open/close bias wrong");
#endif
    assert(m_nameToMapping.empty() && m_dataToName.empty() && "not all opened files were closed");
  }
};

cgltf_result cgltf_read(const struct cgltf_memory_options* memory_options,
                        const struct cgltf_file_options*   file_options,
                        const char*                        path,
                        cgltf_size*                        size,
                        void**                             data)
{
  FileMappingList* mappings = (FileMappingList*)file_options->user_data;
  if(mappings->open(path, size, data))
  {
    return cgltf_result_success;
  }

  return cgltf_result_io_error;
}

void cgltf_release(const struct cgltf_memory_options* memory_options, const struct cgltf_file_options* file_options, void* data)
{
  FileMappingList* mappings = (FileMappingList*)file_options->user_data;
  mappings->close(data);
}

// Defines a unique_ptr that can be used for cgltf_data objects.
// Freeing a unique_cgltf_ptr calls cgltf_free, instead of delete.
// This can be constructed using unique_cgltf_ptr foo(..., &cgltf_free).
using unique_cgltf_ptr = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;


// Traverses the glTF node and any of its children, adding a MeshInstance to
// the meshSet for each referenced glTF primitive.
void addInstancesFromNode(std::vector<tessellatedclusters::Scene::Instance>& instances,
                          const cgltf_data*                                  data,
                          const cgltf_node*                                  node,
                          const glm::mat4                                    parentObjToWorldTransform = glm::mat4(1))
{
  if(node == nullptr)
    return;

  // Compute this node's object-to-world transform.
  // See https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_004_ScenesNodes.md .
  // Note that this depends on glm::mat4 being column-major.
  // The documentation above also means that vectors are multiplied on the right.
  glm::mat4 localNodeTransform(1);
  cgltf_node_transform_local(node, glm::value_ptr(localNodeTransform));
  const glm::mat4 nodeObjToWorldTransform = parentObjToWorldTransform * localNodeTransform;

  // If this node has a mesh, add instances for its primitives.
  if(node->mesh != nullptr)
  {
    const ptrdiff_t meshIndex = (node->mesh) - data->meshes;

    tessellatedclusters::Scene::Instance instance{};
    instance.geometryID = uint32_t(meshIndex);
    instance.matrix     = nodeObjToWorldTransform;

    instances.push_back(instance);
  }

  // Recurse over any children of this node.
  const size_t numChildren = node->children_count;
  for(size_t childIdx = 0; childIdx < numChildren; childIdx++)
  {
    addInstancesFromNode(instances, data, node->children[childIdx], nodeObjToWorldTransform);
  }
}

}  // namespace


namespace tessellatedclusters {
bool Scene::loadGLTF(ProcessingInfo& processingInfo, const std::filesystem::path& filePath)
{
  std::string fileName = nvutils::utf8FromPath(filePath);

  // Parse the glTF file using cgltf
  cgltf_options options = {};

  FileMappingList mappings;
  options.file.read      = cgltf_read;
  options.file.release   = cgltf_release;
  options.file.user_data = &mappings;

  // We are adding only a partial set of textures
  std::unordered_map<int, int> gltfTextureToSceneMap;
  auto                         addTexture = [&gltfTextureToSceneMap](int gltfIndex) {
    auto it = gltfTextureToSceneMap.find(gltfIndex);
    if(it != gltfTextureToSceneMap.end())
    {
      return it->second;
    }
    int nextIndex                    = int(gltfTextureToSceneMap.size());
    gltfTextureToSceneMap[gltfIndex] = nextIndex;
    return nextIndex;
  };


  cgltf_result     cgltfResult;
  unique_cgltf_ptr data = unique_cgltf_ptr(nullptr, &cgltf_free);
  {
    // We have this local pointer followed by an ownership transfer here
    // because cgltf_parse_file takes a pointer to a pointer to cgltf_data.
    cgltf_data* rawData = nullptr;
    cgltfResult         = cgltf_parse_file(&options, fileName.c_str(), &rawData);
    data                = unique_cgltf_ptr(rawData, &cgltf_free);
  }
  // Check for errors; special message for legacy files
  if(cgltfResult == cgltf_result_legacy_gltf)
  {
    LOGE(
        "loadGLTF: This glTF file is an unsupported legacy file - probably glTF 1.0, while cgltf only supports glTF "
        "2.0 files. Please load a glTF 2.0 file instead.\n");
    return false;
  }
  else if((cgltfResult != cgltf_result_success) || (data == nullptr))
  {
    LOGE("loadGLTF: cgltf_parse_file failed. Is this a valid glTF file? (cgltf result: %d)\n", cgltfResult);
    return false;
  }

  // Perform additional validation.
  cgltfResult = cgltf_validate(data.get());
  if(cgltfResult != cgltf_result_success)
  {
    LOGE(
        "loadGLTF: The glTF file could be parsed, but cgltf_validate failed. Consider using the glTF Validator at "
        "https://github.khronos.org/glTF-Validator/ to see if the non-displacement parts of the glTF file are correct. "
        "(cgltf result: %d)\n",
        cgltfResult);
    return false;
  }

  // For now, also tell cgltf to go ahead and load all buffers.
  cgltfResult = cgltf_load_buffers(&options, data.get(), fileName.c_str());
  if(cgltfResult != cgltf_result_success)
  {
    LOGE(
        "loadGLTF: The glTF file was valid, but cgltf_load_buffers failed. Are the glTF file's referenced file paths "
        "valid? (cgltf result: %d)\n",
        cgltfResult);
    return false;
  }

  m_geometries.resize(data->meshes_count);

  auto fnLoadAndProcessGeometry = [&](uint64_t meshIdx, uint32_t threadOuterIdx) {
    const cgltf_mesh gltfMesh = data->meshes[meshIdx];
    Geometry&        geometry = m_geometries[meshIdx];
    geometry.bbox             = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};


    // count pass
    geometry.numTriangles = 0;
    geometry.numVertices  = 0;
    for(size_t primIdx = 0; primIdx < gltfMesh.primitives_count; primIdx++)
    {
      cgltf_primitive* gltfPrim = &gltfMesh.primitives[primIdx];

      if(gltfPrim->type != cgltf_primitive_type_triangles)
      {
        continue;
      }

      // If the mesh has no attributes, there's nothing we can do
      if(gltfPrim->attributes_count == 0)
      {
        continue;
      }

      for(size_t attribIdx = 0; attribIdx < gltfPrim->attributes_count; attribIdx++)
      {
        const cgltf_attribute& gltfAttrib = gltfPrim->attributes[attribIdx];
        const cgltf_accessor*  accessor   = gltfAttrib.data;

        // TODO: Can we assume alignment in order to make these a single read_float call?
        if(strcmp(gltfAttrib.name, "POSITION") == 0)
        {
          geometry.numVertices += (uint32_t)gltfAttrib.data->count;
          break;
        }
      }

      geometry.numTriangles += (uint32_t)gltfPrim->indices->count / 3;
    }

    geometry.normals.resize(geometry.numVertices);
    geometry.texCoords.resize(geometry.numVertices);
    geometry.positions.resize(geometry.numVertices);
    geometry.triangles.resize(geometry.numTriangles);

    // fill pass

    uint32_t offsetVertices  = 0;
    uint32_t offsetTriangles = 0;

    for(size_t primIdx = 0; primIdx < gltfMesh.primitives_count; primIdx++)
    {
      cgltf_primitive* gltfPrim = &gltfMesh.primitives[primIdx];

      if(gltfPrim->type != cgltf_primitive_type_triangles)
      {
        continue;
      }

      // If the mesh has no attributes, there's nothing we can do
      if(gltfPrim->attributes_count == 0)
      {
        continue;
      }

      if(gltfPrim->material && gltfPrim->material->has_displacement)
      {
        int textureID = int(gltfPrim->material->displacement.displacementGeometryTexture.texture - data->textures);
        geometry.displacement.textureIndex = addTexture(textureID);
        geometry.displacement.factor       = gltfPrim->material->displacement.displacementGeometryFactor;
        geometry.displacement.offset       = gltfPrim->material->displacement.displacementGeometryOffset;
      }

      uint32_t numVertices = 0;

      for(size_t attribIdx = 0; attribIdx < gltfPrim->attributes_count; attribIdx++)
      {
        const cgltf_attribute& gltfAttrib = gltfPrim->attributes[attribIdx];
        const cgltf_accessor*  accessor   = gltfAttrib.data;

        // TODO: Can we assume alignment in order to make these a single read_float call?
        if(strcmp(gltfAttrib.name, "POSITION") == 0)
        {
          glm::vec3* writePositions = geometry.positions.data() + offsetVertices;

          if(accessor->component_type == cgltf_component_type_r_32f && accessor->type == cgltf_type_vec3
             && accessor->stride == sizeof(glm::vec3))
          {
            const glm::vec3* readPositions = (const glm::vec3*)(cgltf_buffer_view_data(accessor->buffer_view) + accessor->offset);
            for(size_t i = 0; i < accessor->count; i++)
            {
              glm::vec3 tmp     = readPositions[i];
              writePositions[i] = tmp;
              geometry.bbox.lo  = glm::min(geometry.bbox.lo, tmp);
              geometry.bbox.hi  = glm::max(geometry.bbox.hi, tmp);
            }
          }
          else
          {
            for(size_t i = 0; i < accessor->count; i++)
            {
              glm::vec3 tmp;
              cgltf_accessor_read_float(accessor, i, &tmp.x, 3);
              writePositions[i] = tmp;
              geometry.bbox.lo  = glm::min(geometry.bbox.lo, tmp);
              geometry.bbox.hi  = glm::max(geometry.bbox.hi, tmp);
            }
          }
          numVertices = (uint32_t)accessor->count;
        }
        else if(strcmp(gltfAttrib.name, "NORMAL") == 0)
        {
          glm::vec3* writeNormals = geometry.normals.data() + offsetVertices;

          if(accessor->component_type == cgltf_component_type_r_32f && accessor->type == cgltf_type_vec3
             && accessor->stride == sizeof(glm::vec3))
          {
            const glm::vec3* readPositions = (const glm::vec3*)(cgltf_buffer_view_data(accessor->buffer_view) + accessor->offset);
            for(size_t i = 0; i < accessor->count; i++)
            {
              glm::vec3 tmp   = readPositions[i];
              writeNormals[i] = tmp;
            }
          }
          else
          {
            for(size_t i = 0; i < accessor->count; i++)
            {
              glm::vec3 tmp;
              cgltf_accessor_read_float(accessor, i, &tmp.x, 3);
              writeNormals[i] = tmp;
            }
          }
          numVertices = (uint32_t)accessor->count;
        }
        else if(strcmp(gltfAttrib.name, "TEXCOORD_0") == 0)
        {
          glm::vec2* writeTexCoords = geometry.texCoords.data() + offsetVertices;

          if(accessor->component_type == cgltf_component_type_r_32f && accessor->type == cgltf_type_vec2
             && accessor->stride == sizeof(glm::vec2))
          {
            const glm::vec2* readTexCoords = (const glm::vec2*)(cgltf_buffer_view_data(accessor->buffer_view) + accessor->offset);
            for(size_t i = 0; i < accessor->count; i++)
            {
              glm::vec2 tmp     = readTexCoords[i];
              writeTexCoords[i] = tmp;
            }
          }
          else
          {
            for(size_t i = 0; i < accessor->count; i++)
            {
              glm::vec2 tmp;
              cgltf_accessor_read_float(accessor, i, &tmp.x, 2);
              writeTexCoords[i] = tmp;
            }
          }
          numVertices = (uint32_t)accessor->count;
        }
      }

      // indices
      {
        const cgltf_accessor* accessor = gltfPrim->indices;

        uint32_t* writeIndices = (uint32_t*)(geometry.triangles.data() + offsetTriangles);

        if(offsetVertices == 0 && accessor->component_type == cgltf_component_type_r_32u
           && accessor->type == cgltf_type_scalar && accessor->stride == sizeof(uint32_t))
        {
          memcpy(writeIndices, cgltf_buffer_view_data(accessor->buffer_view) + accessor->offset,
                 sizeof(uint32_t) * accessor->count);
        }
        else
        {
          for(size_t i = 0; i < accessor->count; i++)
          {
            writeIndices[i] = (uint32_t)cgltf_accessor_read_index(gltfPrim->indices, i) + offsetVertices;
          }
        }

        offsetTriangles += (uint32_t)accessor->count / 3;
      }


      offsetVertices += numVertices;
    }

    processGeometry(processingInfo, geometry);

    processingInfo.logCompletedGeometry();
  };

  processingInfo.setupParallelism(data->meshes_count);
  processingInfo.logBegin();
  nvutils::parallel_batches_pooled<1>(data->meshes_count, fnLoadAndProcessGeometry, processingInfo.numOuterThreads);
  processingInfo.logEnd();

  if(data->scenes_count > 0)
  {
    const cgltf_scene scene = (data->scene != nullptr) ? (*(data->scene)) : (data->scenes[0]);
    for(size_t nodeIdx = 0; nodeIdx < scene.nodes_count; nodeIdx++)
    {
      addInstancesFromNode(m_instances, data.get(), scene.nodes[nodeIdx]);
    }
  }
  else
  {
    for(size_t nodeIdx = 0; nodeIdx < data->nodes_count; nodeIdx++)
    {
      if(data->nodes[nodeIdx].parent == nullptr)
      {
        addInstancesFromNode(m_instances, data.get(), &(data->nodes[nodeIdx]));
      }
    }
  }

  if(data->cameras_count > 0)
  {
    for(size_t nodeIdx = 0; nodeIdx < data->nodes_count; nodeIdx++)
    {
      if(data->nodes[nodeIdx].camera != nullptr && data->nodes[nodeIdx].camera->type == cgltf_camera_type_perspective)
      {
        Camera cam{};
        cam.fovy = data->nodes[nodeIdx].camera->data.perspective.yfov;
        glm::mat4 worldNodeTransform(1);
        cgltf_node_transform_world(&data->nodes[nodeIdx], glm::value_ptr(cam.worldMatrix));
        cam.eye    = glm::vec3(cam.worldMatrix[3]);
        cam.center = (m_bbox.hi + m_bbox.lo) * 0.5f;
        cam.up     = {0, 1, 0};
        m_cameras.push_back(cam);
      }
    }
  }

  m_uriTextures.resize(gltfTextureToSceneMap.size());

  std::filesystem::path basedir = filePath.parent_path();

  for(auto& t : gltfTextureToSceneMap)
  {
    std::string           uri_decoded = dlib::urldecode(data->textures[t.first].image->uri);
    std::filesystem::path uri         = std::filesystem::path(uri_decoded);
    if(uri.is_relative())
    {
      uri = basedir / uri;
    }
    m_uriTextures[t.second] = uri.string();
  }

  return true;
}
}  // namespace tessellatedclusters
