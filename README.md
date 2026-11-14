# vk_tessellated_clusters

This sample showcases dynamic tessellation with displacement that requires per-frame generation of clusters for ray tracing using `VK_NV_cluster_acceleration_structure`.
It can also render the content using `VK_NV_mesh_shader` or `VK_EXT_mesh_shader`.

Please have a look at the [vk_animated_clusters](https://github.com/nvpro-samples/vk_animated_clusters) sample prior this, to familiarize yourself with the new cluster extension. The DirectX 12 sample that demonstrates tessellation of OpenSubDiv surfaces can be found at [RTX Mega Geometry](https://github.com/NVIDIA-RTX/RTXMG).

![image showing adaptive tessellation](docs/adaptive_triangle_tessellation.png)

## Dynamic tessellation

The principle steps for both rasterization and ray tracing are similar. The key difference is that rasterization 
directly generates the rendered vertices and triangle data within the meshlet, whilst ray tracing has to write
out vertices and build cluster acceleration structures (CLAS) for use in bottom level acceleration structures (BLAS).

Note that to reduce workload we skip tessellating instances that are not "primary visible" (outside frustum, or occluded).
This can mean that shadows of such instances change as they become visible.

### Model processing

This sample is using the same cluster generation as `vk_animated_clusters` for the loaded models. The key difference is that due to the dynamic nature of this sample,
we always generate **cluster templates** for the original geometry and they always use `VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR`.
The ray tracing template data structures are built in [`raytracing_cluster_data.cpp`](src/raytracing_cluster_data.cpp).

### Tessellation Table

To allow experimenting with different tessellation patterns, the table is embedded into the source code as a header file, feel free to generate your own.
We recommend not exceeding 128 triangles / vertices for the highest tessellation factor. This is why the current table is maxed to 11 segments per triangle edge.

The raw table header file is defined as follows

``` cpp
namespace tessellation_table {
uint32_t max_edge_segments = 11; // number of segments per edge (we recommend <= 11)
uint32_t max_vertices = ...; // total number of vertices across all subdivision patterns
uint32_t vertices[] = {
    // each vertex is stored as uint32_t, a UV coordinate is packed in pairs of 16-bit values
    // 0 = 0.0f
    // 1 << 15 == 1.0f

    // the first non-subdivided triangle's UVs are:
    // (0,0),     (1,0),     (0,1) and stored as:
    // 0x00000000,0x00008000,0x80000000
    ...
};
uint32_t max_triangles = ...; // total number of triangles across all subdivision patterns
uint32_t triangles[] = {
    // each triangle is stored packed in a single uint32_t, where the
    // first 24 bits store 3 x 8 bit counter-clockwise vertex indices and the top 8 bits are unused.
    // The vertex indices are relative to the config's `firstVertex`, see later.

    // the first triangle indices are [0,1,2] stored as 0x00020100
    ...
};
uint32_t max_configs = ...; // sum of all tessellation configs
uint16_t configs[] = {
    // each tessellation config is stored as 4 uint16_t values
    // and defines the triangulation of the config

    // u16 firstTriangle (offset into triangles[])
    // u16 firstVertex (offset into vertices[])
    // u16 numTriangles
    // u16 numVertices
    ...
};

// The configs are stored with increasing edge segments.
// We make use of symmetry and therefore only the permutations
// with edge N segments >= edge N+1 segments need to be stored.
// 
// `edge N` is from `vertex N` to `(vertex N+1) % N` of the triangle

// config: index - edge 0, edge 1, edge 2 segments
// 
// config: 0 - 1, 1, 1  (1 segment for all triangle edges)
// config: 1 - 2, 1, 1  (2 segments for edge 0 and 1 segment for the others)
// config: 2 - 2, 2, 1  (2 segments for edge 0 and edge 1, 1 segment for edge 2)
// config: 3 - 2, 2, 2  (2 segments for all edges)
// config: 4 - 3, 1, 1  ...
// config: 5 - 3, 2, 1
// config: 6 - 3, 2, 2
// config: 7 - 3, 3, 1
// config: 8 - 3, 3, 2
// config: 9 - 3, 3, 3
// ...

// Caveat: for the TRANSIENT2X optimization to work (see later)
// all configs <= 2,2,2 must have <= TESS_2X_TRIANGLES (4)
//                            and <= TESS_2X_VERTICES (6)

} // namespace tessellation_table
```

The runtime table `TessellationTable` contains the raw data for use in mesh-shaders, or when interpreting the data directly. But it also contains **cluster templates** information.

``` cpp
// BUFFER_REF expands to uint64_t in C++ and to the provided `buffer_reference` type in GLSL

struct TessellationTable
{
  BUFFER_REF(uints_in)           vertices;  // u,v barycentrics packed as 2 x 16 bit, 1 << 15 represents 1.0
  BUFFER_REF(uints_in)           triangles; // 3 x 8 bit indices packed in 32 bit

  // these arrays are indexed by a power of 2 lookup table logic, so that we can quickly
  // find patterns for a triangle that is subdivided e.g. 11 x 7 x 10

  BUFFER_REF(TessTableEntrys_in) entries;           // lookup table for each pattern's vertices and triangles
  BUFFER_REF(uint64s_in) templateAddresses;         // cluster template addresses for that pattern
  BUFFER_REF(uints_in) templateInstantiationSizes;  // max cluster instantiation size for the pattern

  // templates that store a batch of independent lowly tessellated triangles,
  // only used when the "combined templates" optimization is active (see later)
  BUFFER_REF(uint64s_in) combinedTemplateAddresses;
  BUFFER_REF(uints_in) combinedTemplateInstantiationSizes;
};
```

Just like for the regular geometry we pre-generate the **cluster templates** for each pattern.

> NOTE: Special thanks to [**Epic Games**](https://www.epicgames.com/) for originally allowing the usage of a tessellation table that was derived from `Unreal Engine 5.4`.
> However, we have since replaced it with our own open-source version, using hints provided on [Brian Karis' blog](https://graphicrants.blogspot.com/2026/02/how-to-tessellate.html).

### Runtime Operations

![image showing the data flow and runtime operations](docs/adaptive_triangle_tessellation_flow.png)

To build the content we actually render, a series of dispatches and/or indirect build operations are performed.

Next to the kernel files that are mentioned for the steps, these are the main files to look at:
- [`shaders/common.h`](shaders/common.h): All data structures that are used by host and device can be found in `namespace shaderio` within this file.
- [`renderer_raster_clusters_tess.cpp`](src/renderer_raster_clusters_tess.cpp): The renderer that implements rasterization using `VK_NV_mesh_shader` or `VK_EXT_mesh_shader`.
- [`renderer_raytrace_clusters_tess.cpp`](src/renderer_raytrace_clusters_tess.cpp): The renderer that implements ray tracing using `VK_NV_cluster_acceleration_structure`.

All relevant indirect dispatches, and lists of pointers or structs are stored in the `shaderio::SceneBuilding` struct.

``` cpp

// re-ordered for readability! doesn't match code

struct SceneBuilding
{
  uint numRenderInstances;
  
  // result of classifying instances
  BUFFER_REF(uints_inout) instanceStates;

  // result of traversal / culling
  uint                           visibleClusterCounter;
  BUFFER_REF(ClusterInfos_inout) visibleClusters;

  // indirect launch for classifying clusters
  DispatchIndirectCommand dispatchClassify;

  // clusters to be rendered as is without subdivision
  uint                                fullClusterCounter;
  BUFFER_REF(ClusterInfos_inout)      fullClusters;

  // triangle to be rendered (subdivision fits in table)
  uint                                partTriangleCounter;
  BUFFER_REF(TessTriangleInfos_inout) partTriangles;

  // triangle to be split (subdivision too big)
  // this list is updated during a recursive splitting as well.
  int                                 splitTriangleCounter;
  uint                                splitReadCounter;
  uint                                splitWriteCounter;
  BUFFER_REF(TessTriangleInfos_coh)   splitTriangles;

  // rasterization
  // -------------

  DrawMeshTasksIndirectCommandNV drawFullClusters;
  DrawMeshTasksIndirectCommandNV drawPartTriangles;
  
  // ray tracing
  // -----------

  DispatchIndirectCommand dispatchClusterInstantiate;
  DispatchIndirectCommand dispatchTriangleInstantiate;

  // Storage for all generated clusters:

  // Generated vertices are needed for template instantiations or transient cluster builds.
  // For transient cluster builds, we also store the 3 x 8-bit triangle indices in this buffer.
  uint                                          genVertexCounter;
  BUFFER_REF(vec3s_inout)                       genVertices;

  // The number of clusters we build or instantiate in total, as well
  // as the reserved size they will use. We use explicit cluster building and
  // template instantiations, both
  uint     genClusterCounter;
  uint64_t genClusterDataCounter;

  uint64_t genClusterData; // base address for all generated clusters

  // We generate clusters either through template instantiations or 
  // transient builds.

  // the actual number of templates we will instantiate
  uint     tempInstantiateCounter;
  // the actual number of transient clusters we will build
  uint     transBuildCounter;

  // for each template instantiation:
  BUFFER_REF(uints_inout)                       tempInstanceIDs;      // need to know which render instance it came from
  BUFFER_REF(TemplateInstantiateInfos_inout)    tempInstantiations;   // for each template the instantiation arguments
  BUFFER_REF(uint64s_inout)                     tempClusterAddresses; // explicit target address for this cluster
  BUFFER_REF(uints_inout)                       tempClusterSizes;     // after instantiation, output size (only used for statistics)

  // for each transient cluster build (same as above)
  BUFFER_REF(uints_inout)                       transInstanceIDs;
  BUFFER_REF(ClusterBuildInfos_inout)           transBuilds;
  BUFFER_REF(uint64s_inout)                     transClusterAddresses;
  BUFFER_REF(uints_inout)                       transClusterSizes;

  // dispatch setup to fill the instantiated template clusters into the blas lists
  DispatchIndirectCommand dispatchBlasTempInsert;
  // dispatch setup to fill the transient built clusters into the blas lists
  DispatchIndirectCommand dispatchBlasTransInsert;

  // a large array of cluster addresses, but now sectioned by blas
  // the values will come from `tempClusterAdresses` and
  // `blasBuildInfos[].clusterReferences` will point into this
  BUFFER_REF(uint64s_inout)              blasClusterAddresses; 


  // for each render instance, we generate a blas
  BUFFER_REF(ClusterBlasBuildInfo_inout) blasBuildInfos;  // blas build arguments
  BUFFER_REF(uints_inout)                blasBuildSizes;  // after build, size (only used for statistics)

  uint64_t blasBuildData;     // storage of cluster data (not really used, we use implicit builds)

  uint numBlasReservedSizes;  // just for statistics
};
```

1. **Classify Instances**
    For each instance we determine if the entire instance is within frustum and/or visible (not occluded). See [instances_classify.comp.glsl](shaders/instances_classify.comp.glsl). The occlusion determination is kept simple and we use the last frame's HiZ and projection information.

    Instances that are fully outside frustum, or fully occluded from the camera will not receive tessellation.

    |Input | Output |
    |------|--------|
    |Instance | `instanceStates[]`  |

2. **Culling of Scene Clusters**
    A series of compute shader dispatches collect all visible clusters. Optionally frustum and occlusion culling is performed. See [clusters_cull.comp.glsl](shaders/clusters_cull.comp.glsl). The occlusion culling is kept simple and we use the last frame's HiZ and projection information.

    |Input | Output |
    |------|--------|
    |Geometry | `visibleClusters[]` & counter |
    | `instanceStates[]` | |

3. **Build Setup `BUILD_SETUP_CLASSIFY`**
   
    This compute shader simply sets up the indirect dispatch for the next operation. It clamps the actual number of visible clusters to what can be stored in buffers. See
    [build_setup.comp.glsl](shaders/build_setup.comp.glsl).
    
    |Input | Output |
    |------|--------|
    |`visibleClusterCounter` | `dispatchClassify.gridX` |

4. **Classify Cluster**
   
    The visible clusters are classified regarding their tessellation status, see [cluster_classify.comp.glsl](shaders/cluster_classify.comp.glsl)
    and its two ray tracing variants that are described further below.
    We transform the cluster vertices into shared memory and then determine the tessellation factors for every triangle within the cluster.
    If all triangles are "simple" (i.e 1 x 1 x 1 segments) then the entire cluster is treated as "full cluster".

    For ray tracing this means we directly setup the instantiation arguments of this cluster, and copy the vertices from shared memory to the allocated
    global memory. This is only done if we can ensure enough space is available in the temporary arrays. In rasterization we append into `fullClusters[]`.

    Triangles that are subdivided small enough to fit in the tessellation table, are sent to the `partTriangles[]` list.
    Those that are too big and require recursive splitting are sent to `splitTriangles[]`.

    For ray tracing the sample can apply some optional optimizations in this step, the goal is to create less
    clusters per frame, and increase the number of triangles within the generated clusters. This is an important optimization for ray tracing.
    Without these optimizations, all clusters are built through template instantiations.

    The ui options `optimize == 1x1x1` and `optimize <= 2x2x2` pick which triangles get this treatment, the first covers the clusters
    where no triangle requires tessellation, the second the triangles whose edge factors are all `<= 2`. The ui option `low tessellation mode`
    picks the method, either through **transient clusters** or through **combined templates** (default). To ease readability and diffing, each of these optimizations has its own file:
    - [cluster_classify.comp.glsl](shaders/cluster_classify.comp.glsl) (also used by rasterization)
    - [cluster_classify_transient_lowtess.comp.glsl](shaders/cluster_classify_transient_lowtess.comp.glsl)
    - [cluster_classify_combined_lowtess.comp.glsl](shaders/cluster_classify_combined_lowtess.comp.glsl)

    |Input | Output |
    |------|--------|
    |`visibleClusters[]`                             | `fullClusters[]` & counter (raster only) |
    |`instanceStates[]`                              | `partTriangles[]`                        |
    |Geometry vertices, triangles, clusters..        | `splitTriangles[]` & counter             |
    |                                                | `genVertices[]` & counter                |
    |                                                | `genClusterCounter`                      |
    |                                                | `genClusterDataCounter`                  |
    |`geometry[].clusterTemplateAdresses[]`          | `tempInstantiations[]` & counter         |
    |`geometry[].clusterTemplateInstantiatonSizes[]` | `tempInstanceIDs[]`                      |
    |                                                | `tempClusterAddresses[]`                 |
    |                                                | `transBuilds[]` & counter                |
    |                                                | `transInstanceIDs[]`                     |
    |                                                | `transClusterAddresses[]`                |
    |                                                | `blasBuildInfos[instanceID].clusterReferencesCount` |

    **Transient Clusters**

    The ui option `optimize == 1x1x1` triggers the building of a transient cluster with all triangles that are simple, in case
    the entire cluster wasn't classified simple. This avoids generating lots of template instantiations with just 1 triangle.
    The triangle indices used for such transient cluster builds are stored within `genVertices[]` simply after the vertices.

    When this codepath is triggered, we also store slightly different information in the `partTriangles[]` array to be able
    to map the transient cluster's triangles (now less than the original cluster) back to the original triangle IDs. Multiple consecutive elements within `partTriangles[]` may be used
    to store this uint8 array. Effectively we store `uint instanceID; uint clusterID; uint8_t triMappings[];`

    The second ui option `optimize <= 2x2x2` can be enabled independently of the first, and batches every few lowly-subdivided triangles into a
    transient cluster build. This time `partTriangles[]` stores `uint instanceID; uint clusterID; uint16_t triInfos[];` where the `triInfos[]` array allows
    us to map every generated subdivided triangle back to the original cluster triangle, as well as which subdivision factors were used and the relative triangle index
    within the subdivision. This information is later needed to generate appropriate UV coordinates within the hit shader.

    **Combined Templates**

    The transient cluster builds require a second cluster operation next to the template instantiation, and we have to generate
    the triangle indices for them at runtime. The combined templates avoid both, they are **cluster templates** that we pre-generate at
    init time and that already store a batch of independent lowly tessellated triangles. That way we never leave the template
    instantiation codepath, and just like for any other template we only have to provide the vertices.
    See [tessellation_combined.glsl](shaders/tessellation_combined.glsl) and `TessellationTable::initCombinedTemplates`.

    > NOTE: Thanks to Jerran Schmidt <jschmidt@nvidia.com> for developing and testing the idea in a separate application.

    We generate two families of these templates, and one template for every batch size within a family, so that a batch that
    wasn't filled completely doesn't instantiate more triangles than it actually uses.

    - The "1x" family stores `n` independent triangles, which means `3 x n` vertices and `n` triangles. `n` goes up to the ui option
      `combined 1x1x1 batch size` (default 64). We use it for the triangles of a cluster that need no tessellation, just like the first
      transient option. The triangle at index `i` always uses the vertices `i * 3 + 0/1/2`, so this time we do have to duplicate the
      vertices per triangle and lose the connectivity that the transient build could benefit from.
    - The "2x" family stores `n` independent 2x2x2 patches, which means `6 x n` vertices and `4 x n` triangles. `n` goes up to
      `combined 2x2x2 batch size` (default 16). A patch is made out of the three corners of the base triangle and the mid point of each of
      its three edges. All the lower configurations (1x1x1, 2x1x1, 2x2x1) come out of that very same topology by simply collapsing the mid point
      of an edge with factor 1 onto one of the edge's corners. The triangles that referenced the collapsed vertex become degenerate and will
      never be hit by a ray, while the remaining ones still cover the base triangle exactly.

    Both families store their information at the back of `partTriangles[]` just like the transient clusters do. The "1x" family
    stores `uint instanceID; uint clusterID; uint8_t triMappings[];` and the "2x" family `uint instanceID; uint clusterID; uint16_t patchInfos[];`,
    where every `patchInfos[]` element gives us the original cluster triangle as well as the tessellation factors that the patch was
    generated with. Given a patch always occupies the same number of triangles, the hit shader can compute which patch was hit from
    `gl_PrimitiveID` directly, we don't need a mapping for every generated triangle anymore.

    Compared to the transient builds we end up with bigger CLAS, we neither benefit from connectivity nor do we get rid of the
    degenerate triangles, but the instantiation itself is cheaper than the build.

5. **Build Setup `BUILD_SETUP_SPLIT`**
   
    Clamp the counters to ensure we don't exceed splitting work. Setup index for output writes.

    |Input | Output |
    |------|--------|
    |`splitTriangleCounter` | `splitTriangleCounter` |
    |                       | `splitWriteCounter` (matches above) |

6. **Triangle Split**
   
    Recursively split every triangle whose subdivision is too high to expressed by the tessellation table,
    see [triangle_split.comp.glsl](shaders/triangle_split.comp.glsl).

    We generate partial triangles and encode their barycentric positioning with the original triangle.
    The `splitTriangles[]` list as well as read/write counters are modified in this process. Every child
    of a split triangle is classified again whether it needs splitting or can be sent to `partTriangles[]`.

    |Input | Output |
    |------|--------|
    |`splitTriangles[]` & counter              | `partTriangles[]` & counter |
    |                                          | `splitTriangles[]` & counter (recursive) |
    | `splitWriteCounter` & `splitReadCounter` | `splitWriteCounter` & `splitReadCounter` (recursive) |

7. **RASTER ONLY - Indirect Setup `BUILD_SETUP_DRAW_TESS`**
   
    Setup the indirect draw calls for both partial triangles and full clusters.

    |Input | Output |
    |------|--------|
    |`partTriangleCounter` | `drawFullClusters` |
    |`partTriangleCounter` | `drawPartTriangles` |

8. **RASTER ONLY - Draw Scene - RASTER MOSTLY ENDS HERE**
   
    Draw the full clusters via [render_raster_clusters.mesh.glsl](shaders/render_raster_clusters.mesh.glsl) and the partial either via [render_raster_clusters_tess.mesh.glsl](shaders/render_raster_clusters_tess.mesh.glsl) or a task/mesh combination of [render_raster_clusters_batched.task.glsl](shaders/render_raster_clusters_batched.task.glsl) and [render_raster_clusters_batched.mesh.glsl](shaders/render_raster_clusters_batched.mesh.glsl).

    These "batched" shaders are used by default and their usage is controlled by the ui option `batch meshlets` in the rasterization settings. The batching is conceptually similar to the "transient" clusters in the ray tracing case, however it's broader in use and will combine any tessellation levels. The [task shader](shaders/render_raster_clusters_batched.task.glsl) looks at 32 partial triangles and then tries to batch multiple of them as long as the sum of their triangles and vertices fit into the meshlet (`TESS_RASTER_BATCH_VERTICES = 96` and `TESS_RASTER_BATCH_TRIANGLES = 128`).
    The subsequent [mesh shader](shaders/render_raster_clusters_batched.mesh.glsl) then computes a total sum of work items for vertices and triangles that its subgroup iterates over. It does a dense packing of the work, so we don't waste SIMT utilization.

    |Input | Output |
    |------|--------|
    |`drawFullClusters` | the most exciting pixels ever |
    |`drawPartTriangles`|  |
    |`fullClusters[]`   |  |
    |`partTriangles[]`  |  |

9. **Build Setup `BUILD_SETUP_INSTANTIATE_TESS`**
   
    Setup the next kernel for the partial triangle template preparation.

    |Input | Output |
    |------|--------|
    |`partTriangleCounter` | `dispatchTriangleInstantiate` |

10. **Prepare Instantiation**
   
    Fills the template instantiation arguments and temporary vertices for partial triangles, see [triangle_tess_template_instantiate.comp.glsl](shaders/triangle_tess_template_instantiate.comp.glsl).
    This time the generated vertices are within a triangle, and based on the barycentric vertices of the tessellation table as well as the region the partial triangle covers.
    Each partial triangle provides the region via `TessTriangleInfo::subTriangle.vtxEncoded[]` and the subdivision configuration via `TessTriangleInfo::subTriangle.triangleID_config`.

    Just like in classify we ensure we don't exceed reserved data.

    To make better use of the threads in the workgroup, we batch the processing of multiple partial triangles into one workgroup.
    This batching is controlled via `TESS_INSTANTIATE_BATCHSIZE`. Without such batching we risk under utilizing the hardware a lot, given for example a low resolution sub triangle might only generate 4 vertices, wasting a lot of the default workgroup's 64 threads.

    We suggest to read the code with `TESS_INSTANTIATE_BATCHSIZE == 1` first, to easier understand what it does. With higher batch sizes
    the kernel first computes the sum of threads needed over all partial triangles, and then loops the workgroup's threads until this sum is reached. During the loop it figures out which of the batches' partial triangle each thread does contribute to and which subdivided vertex is computed.
    
    Note: The instantiated template clusters will have a clusterID that is tagged (highest two bits) and then the clusterID actually encodes to the index into `partTriangles[]`.
    That way during ray tracing we can find back the relevant information for this partial triangle.

    |Input | Output |
    |------|--------|
    |`partTriangles[]`                             |                                          |
    |Geometry vertices, triangles, clusters..      |                                          |
    |`tessTable.templateAdresses[]`                | `tempInstantiations[]` & counter         |
    |`tessTable.templateInstantiationSizes[]`      | `tempInstantiateDataCounter`             |
    |                                              | `tempInstanceIDs[]`                      |
    |                                              | `genVertices[]` & counter                |
    |                                              | `blasBuildInfos[instanceID].clusterReferencesCount` |

11. **Build Setup `BUILD_SETUP_BUILD_BLAS`**
   
    Setup counter of actual instantiation as well as transient cluster builds, and also the 
    dispatch kernels that insert the cluster addresses into the per-blas input lists.

    |Input | Output |
    |------|--------|
    |`tempInstantiateCounter`     | `dispatchBlasTempInsert` |
    |`transBuildCounter`          | `dispatchBlasTransInsert`  |

12. **Template Instantiation && Transient Cluster Builds**
    
    Via `vkCmdBuildClusterAccelerationStructureIndirectNV` we instantiate or build all transient clusters to be used as input for the blas builds.

    Note: after instantiation we can toss away the `genVertices` and other inputs to the cluster build. That is why you
    will see that the sample overlaps the `blasBuildData` memory with this temporary transient data.

    When the combined templates are used, there is nothing to build and we only run the instantiation below.

    **Template Instantiation**

    |Input | Output |
    |------|--------|
    |`tempInstantiations[]` & counter | `tempClusterSizes[]` |
    |`tempClusterAddresses[]`         | `genClusterData` explicit build |
    |`genVertices[]`                  |  |
    |`tessTable.templateData`         |  |
    |`geometry[].clusterTemplateData` |  |

    **Transient Cluster Builds**

    |Input | Output |
    |------|--------|
    |`transBuilds[]` & counter        | `transClusterSizes[]` |
    |`transClusterAddresses[]`        | `genClusterData` explicit build |
    |`genVertices[]` (used for vertices & index data) |  |


11. **BlAS Setup - (Overlaps previous)** 
   
    This dispatch prepares the per-blas cluster reference lists, see [blases_build_setup.comp.glsl](shaders/blases_build_setup.comp.glsl)

    |Input | Output |
    |------|--------|
    |`blasBuildInfos[].clusterReferencesCount` | `blasClusterCounter` atomically add `.clusterReferencesCount` |
    |                                          | `blasBuildInfos[].clusterReferences` acquire new start address for this list based on above |
    |                                          | `blasBuildInfos[].clusterReferencesCount` reset to 0 |

12. **Cluster BlAS Insertion**
    
    After the clusters were built, distribute their resulting addresses across the per-blas lists, see [blas_clusters_insert.comp.glsl](shaders/blas_clusters_insert.comp.glsl). 
    Originally the addresses come in random ordering from different instances, but for BLAS build we need them in dedicated sections.

    The dispatch is run twice either with template or transient inputs.

    |Input | Output |
    |------|--------|
    | `tempInstanceIDs[]` or `transInstanceIDs[]`           | `blasBuildInfos[instanceID].clusterReferencesCount` atomically add `1` for each cluster |
    | `tempClusterAddresses[]` or `transClusterAddresses[]` | `blasBuildInfos[instanceID].clusterReferences[]` append based on above |

13. **BLAS Build**
    
    Through `vkCmdBuildClusterAccelerationStructureIndirectNV` we now build the BLAS over all render instances.

    |Input | Output |
    |------|--------|
    | `blasBuildInfos[]`           | `blasBuildData` implicit build |
    |                              | `rayinstances[].accelerationStructureReference` stores result blas address directly |
    |                              | `blasBuildSizes[]` only for stats |
    | `genClusterData` indirectly  | |

14. **TLAS Build**
    
    Now that all BLAS were built and assigned to each render instances we can build the TLAS. This is done by traditional `vkCmdBuildAccelerationStructuresKHR` see `Renderer::updateRayTracingTlas` 

    |Input | Output |
    |------|--------|
    | `rayinstances[]`             | `tlasData`|
    | `blasBuildData` indirectly   |           |
    | `genClusterData` indirectly  |           |

15. **Ray Tracing**
    
    Finally. We do a classic ray tracing pipeline launch via `vkCmdTraceRaysKHR`.

    Our hit shaders have to react whether they are hitting clusters from triangulated triangles, or regular geometry, see [render_raytrace_clusters.rchit.glsl](shaders/render_raytrace_clusters.rchit.glsl).
    We make use of `VK_KHR_ray_tracing_position_fetch` to avoid having to keep the `tempVertices` around. However, for ray tracing clusters that originated from tessellation, we have to keep `partTriangles[]` so we know which region of an original geometry triangle they cover. This is needed for computing the right barycentric weights, for example for texture coordinates or other interpolated attributes.

    |Input | Output |
    |------|--------|
    | Geometry vertices, triangles, clusters... | the most beautiful pixels ever |
    | `partTriangles[]`            |           |
    | `tlasData` indirectly        |           |
    | `blasBuildData` indirectly   |           |
    | `genClusterData` indirectly  |           |

16. **Depth Write**
    
    Converts the ray tracing depth to depth-buffer via fragment shader writing `gl_FragDepth` so internal hw acceleration structures are updated along.
    While this sample doesn't really rasterize geometry on top of ray tracing, common use-cases would. 

17. **HiZ - ALSO RASTER AGAIN**
    
    Create the HiZ texture based on current depth buffer.

## Low Tessellation Performance

The low tessellation optimizations for ray tracing compared against each other, with both `optimize == 1x1x1` and `optimize <= 2x2x2` active,
and against having none of them. Measured on an NVIDIA RTX PRO 5000 Blackwell with locked clocks, default bunny scene, 1 ambient occlusion
and 1 shadow ray at 2560 x 1440. Timings are GPU time.

``` bash
vk_tessellated_clusters --headless --winsize 2560 1440 --supersample 1 --renderer 1 --dlss 0 --aosamples 1 --clusterconfig 0 --lowtess1x 1 --lowtess2x 1 --lowtessmode 1
```

64 triangles, 64 vertices per cluster (`--clusterconfig 0`):

| | none | transient builds | combined templates |
|------|-----:|-----:|-----:|
| [us]                      |           |           |           |
| Cluster Classify          |       455 |       579 |       631 |
| PrepInstantiate           |       510 |        82 |        83 |
| Clas Instantiate / Build  |      3054 |      1428 |      1157 |
| Insert                    |       503 |       150 |       161 |
| Blas Build                |      1214 |       362 |       364 |
| Render (1 AO ray)         |       914 |       868 |       994 |
| Render (2 AO rays)        |      1103 |      1047 |      1216 |
| **Frame (1 AO ray)**      |  **6766** |  **3583** |  **3503** |
| [units]                   |           |           |           |
| Generated CLAS            |    747892 |    220599 |    227955 |
| CLAS used (MiB)           |     408.6 |     230.9 |     258.0 |
| Scratch (MiB)             |     154.8 |     154.8 |     154.8 |

128 triangles, 128 vertices per cluster (`--clusterconfig 4`):

| | none | transient builds | combined templates |
|------|-----:|-----:|-----:|
| [us]                      |           |           |           |
| Cluster Classify          |       273 |       386 |       422 |
| PrepInstantiate           |       631 |        82 |        83 |
| Clas Instantiate / Build  |      3491 |      1323 |      1011 |
| Insert                    |       564 |       108 |       117 |
| Blas Build                |      1381 |       353 |       352 |
| Render (1 AO ray)         |       923 |       883 |      1044 |
| Render (2 AO rays)        |      1116 |      1067 |      1284 |
| **Frame (1 AO ray)**      |  **7376** |  **3250** |  **3144** |
| [units]                   |           |           |           |
| Generated CLAS            |    843189 |    154382 |    165673 |
| CLAS used (MiB)           |     447.6 |     212.9 |     245.7 |
| Scratch (MiB)             |     154.8 |     154.8 |     154.8 |

Both optimizations cut the number of generated CLAS to about a quarter, which more than halves the frame, as everything after the
classification scales with the number of CLAS. Between the two:

- **Transient builds** create the smallest CLAS that trace fastest, as they use the cluster's actual topology and vertex positions.
  But building a CLAS from scratch is more expensive than an instantiation.
- **Combined templates** only instantiate, which makes them the cheapest to build. The independent triangles and degenerate patch
  triangles make the CLAS bigger (+12-15 %) and slower to trace (+15-18 %).

The scratch memory is the same for all, it is dominated by the BLAS build.

With few rays per pixel the cheaper build wins, hence combined templates are the default. The more rays are traced, the more
transient builds become the better choice.

## Displacement Texturing

The sample uses the unofficial glTF 2.0  [`KHR_materials_displacement`](https://github.com/KhronosGroup/glTF/issues/948) extension.
However, the samples is kept basic and doesn't support multiple texcoords or texture coordinate transforms.

Example of the extension:

``` cpp
...
"materials": [
{
    "extensions": {
    "KHR_materials_displacement": {
        "displacementGeometryFactor": 0.4,
        "displacementGeometryOffset": -0.2,
        "displacementGeometryTexture": {
        "index": 3
        }
    }
    },
...
```
## Problem-Solving

Flickering is expected to occur when the various intermediate data buffers do not provide enough space. Within the UI the appropriate statistics will change in color and a warning message will appear on the main viewport. This is typically addressed by raising the limits and providing more memory space.

The technology being quite new, we might not have ironed out all issues. If you experience instabilities, please let us know through GitHub Issues.
You can use the commandline to change some defaults:

* `--renderer 0` starts with rasterization.
* `--supersample 0` disables the super sampling that otherwise doubles rendering resolution in each dimension. 
* `--gridcopies N` set the number of model copies in the scene.
* `--tessrate F` set the float value of the tessellation rate in pixels per segment. `0` to disable.
* `--vsync 0` disable vsync. If changing vsync via UI does not work, try to use the driver's *NVIDIA Control Panel* and set `Vulkan/OpenGL present method: native`.

## Future Improvements

- Provide an alternative open-source tessellation table beyond the version derived from Unreal Engine.
- Adaptive tessellation based on culling results for ray tracing: It might be better to gradually reduce tessellation of invisible objects. Currently the sample does a per-instance decision that is binary and therefore shadows of an invisible object might pop if an object becomes visible. One could fade tessellation rate based on distance to frustum planes for instances outside, and for occluded objects a time-based fading could be used. Alternatively, the [DirectX 12 sample](https://github.com/NVIDIA-RTX/RTXMG) on subdivision surface tessellation does use occlusion culling information on a per-patch level.
- The tessellation factors are calculated based on the un-displaced vertices to reduce complexity of the sample.
- Reduce memory consumption for rasterizer's low-tessellated triangles.

## Limitations

* Facet shading is enforced when tessellation is active. Smooth normals are not available, as the sample doesn't do procedural perturbation of normals matching the displacement effect, nor does it have support for deformed normal maps when displacement maps are used. 
* The `ClusterID` can only be accessed in shaders using  `gl_ClusterIDNV` after enabling `VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV::allowClusterAccelerationStructure` for that pipeline. We use `GL_EXT_spirv_intrinsics` rather than dedicated GLSL extension support that may come at later time.
* Few error checks are performed on out of memory situations, which can happen on higher _"render copies"_ values, or the complexity of the loaded scene
* If the cluster library generates clusters > 256 vertices, scene loading will fail. This should be addressed in future versions.
* The number of threads used in the persistent kernel is basd on a crude heuristic for now and was not evaluated to be the optimal amount.
* DLSS motion vectors don't reflect changes of the tessellation, which shift displaced surfaces slightly.

## Building and Running

Requires at least Vulkan SDK 1.4.309.0

The new `VK_NV_cluster_acceleration_structure` extension requires new drivers, earliest release version is `572.16` from 1/30/2025.
The sample should run on older drivers with just rasterization available.

Point cmake to the `vk_tessellated_clusters` directory and for example set the output directory to `/build`.
We recommend starting with a `Release` build, as the `Debug` build has a lot more UI elements.

The cmake setup will download the `Stanford Bunny` glTF 2.0 model that serves as default scene.

If `USE_DLSS` is activated in the cmake options, then the DLSS/NGX runtime is downloaded by cmake setup as well:

* **Ray tracing:** DLSS Ray Reconstruction (DLSS-RR) denoises the ambient occlusion (`DLSS - RR` in the UI, or `--dlss 1` with the ray tracer).
* **Rasterization:** DLSS Super Resolution (DLSS-SR) upscales a jittered lower-resolution render (`DLSS - SR` in the UI, or `--dlss 1` with `--renderer 0`).

Use `--dlssquality` or the UI quality preset to control the performance/quality trade-off.

It will also look for [`nvpro_core2`](https://github.com/nvpro-samples/nvpro_core2) either as subdirectory of the current project directory, or up to two levels above. If it is not found, it will automatically download the git repo into `/build/_deps`.
Note, that the repository of `nvpro_core2` needs to be updated manually in case the sample is updated manually, as version mismatches could occur over time.

### Headless and Testing

The sample can run without a window via `--headless` and it can walk a list of configurations on
its own via `--sequencefile` or `--sequencestring`, which is also used for benchmarking, as the
profiler results are printed after every sequence.

``` bash
vk_tessellated_clusters --headless --headlessframes 400 --sequencestring "SEQUENCE \"transient\" --sequenceframes 100 --lowtessmode 0 SEQUENCE \"combined\" --lowtessmode 1"
```

We use this for a set of tests that cover the various cluster building configurations as well as
running out of memory, see [tests/README.md](tests/README.md).

``` bash
python tests/run_tests.py
```

### Camera Paths

A camera path is a keyframed fly-through, authored in the UI (_Misc Settings -> Camera Paths_, with copy/paste of the
path string, save/load to `<scene>.camerapaths.txt` next to the model) or given on the command line. For deterministic
benchmarking `--runcamerapath <index> <framecount>` spreads a path across exactly that many frames, which is meant to be
used within sequences, see [tests/camerapath.txt](tests/camerapath.txt). The index is the order of `--addcamerapath`.

``` bash
vk_tessellated_clusters --headless --addcamerapath "smooth 1 ; {-0.3,2.5,8}, {-0.3,1.2,0}, {0,1,0}, {45} ; {6,3,5}, {-0.3,1.2,0}, {0,1,0}, {45}" --sequencestring "SEQUENCE \"flythrough\" --sequenceframes 16 --runcamerapath 0 16"
```

## Further Samples about NVIDIA RTX Mega Geometry

Other Vulkan samples using the new extensions:
- https://github.com/nvpro-samples/vk_animated_clusters - showcases basic usage of new ray tracing cluster extension.
- https://github.com/nvpro-samples/vk_lod_clusters - provides a sample implementation of a basic cluster-lod based rendering and streaming system.
- https://github.com/nvpro-samples/vk_partitioned_tlas - New extension to manage incremental TLAS updates.

We also recommend having a look at [RTX Mega Geometry](https://github.com/NVIDIA-RTX/RTXMG), which demonstrates tessellation of subdivision surfaces in DirectX 12.

## Acknowledgements

We would like to thank [**Epic Games**](https://www.epicgames.com/) for originally allowing the use of their nanite tessellation table, although the sample no longer ships with it and now uses an open-source version. Furthermore, the idea of using a persistent kernel,
that splits triangles recursively, was inspired from how `Unreal Engine 5.4` implements nanite tessellation.

## Third Party

[meshoptimizer](https://github.com/zeux/meshoptimizer) is used to build and optimize the clusters.

It uses a modified version of [cgltf](https://github.com/jkuhlmann/cgltf) by Johannes Kuhlmann. The unofficial [`KHR_materials_displacement`](https://github.com/KhronosGroup/glTF/issues/948) extension was added.

It also uses [dlib](http://dlib.net/dlib/server/server_http.cpp.html) by Davis E. King.
