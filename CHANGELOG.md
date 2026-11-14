# Changelog for vk_tessellated_clusters
* 2026-11-02:

  * Bugfix: hiz culling clamped lookups two texels short of the screen edge and rounded odd render sizes down, so occluded tests at the right/top edge could wrongly pass or fail. Odd sizes are handled now. The MSAA first pass also skipped the last row and column.

* 2026-10-26:
  * HBAO runs at quarter resolution by default (`--hbaoquarter 0` for full), upsampled depth-aware.
  * HBAO normals are rebuilt per pixel from the plane-consistent neighbours, which removes the dark dashes along creases.
* 2026-10-22:
  * added camera paths, ported from vk_lod_clusters: keyframed fly-through (`--addcamerapath`, `--loadcamerapaths`) with a
    "Camera Paths" ui section. `--runcamerapath <index> <framecount>` plays one deterministically across a fixed number of frames.
  * added `tests/camerapath.txt`.
  * new parameters `--pndisplacement`, `--flipwinding`, `--persistentkernel`, `--persistentthreads`, `--visibleclusterbits`, `--truncatebits`,
    `--clusterbuildmode`, `--facetshading`, `--hbao`, `--hbaoradius` and `tests/build_options.txt`.
  * fixed `--supersample` resolutions keep their aspect ratio in the viewport (letterboxed) and are saved as is, without a resolve to the window size.
* 2026-10-9:
  * DLSS SDK 310.9.1, its default ray reconstruction model is RR2 (preset F).
* 2026-10-8:
  * DLSS loads on a worker thread. With a window the sample renders without DLSS until it is ready, the ui option is
    greyed out meanwhile (`--dlssdelayload 0` waits instead, as headless runs and sequence files always do).
  * DLSS falls back to native rendering on failed evaluation or unsupported quality modes, ultra quality is gone.
  * `tests/run_tests.py` disables DLSS and ambient occlusion rays by default, `tests/dlss.txt` enables DLSS itself.
  * bugfix the HBAO blur missed rows on targets taller than wide.
* 2026-10-7:
  * added DLSS support (cmake option `USE_DLSS`): super resolution in rasterization and ray reconstruction in ray
    tracing (`--dlss`, `--dlssquality`). The tessellation rate refers to the output resolution, and the motion vectors
    account for the animation.
  * DLSS and a single ambient occlusion ray are the new defaults.
  * `--supersample` also takes fixed resolutions: 540, 720, 1080, 1440, 2160 and square 1024, 2048, 4096.
  * newer HBAO implementation, always at render resolution, hence the "full resolution" option is gone. `--hbaoblur`
    toggles its blur.
  * reversed-Z depth.
  * added `tests/dlss.txt`.
* 2026-10-6:
  * added `VK_EXT_mesh_shader` support, "Use EXT mesh shader" ui option (`--extmeshshader`). Forced on without
    `VK_NV_mesh_shader`, which is now optional. Launch grids are kept within the 16 bit per dimension limit.
  * added `--batchmeshlets` and `tests/mesh_shaders.txt`.
* 2026-9-17:
  * new "combined templates" optimization for ray tracing as alternative to the transient cluster builds. Rather than building
    transient CLAS for the lowly tessellated triangles, we instantiate pre-built templates that store a batch of independent
    triangles, or a batch of independent 2x2x2 patches (lower configurations collapse vertices within a patch). As a result the
    entire runtime stays on a single codepath: everything is a template instantiation and no index data has to be generated.
    Enabled via the new "Low tessellation mode" ui option (`--lowtessmode 1`), the ui options for the transient builds were
    renamed to "Optimize == 1x1x1" / "Optimize <= 2x2x2" (`--lowtess1x`, `--lowtess2x`) as they now drive both mechanisms.
    Thanks to Jerran Schmidt <jschmidt@nvidia.com> for developing and testing the idea in a separate application.
  * the cluster classification now has one kernel per low tessellation optimization, so that the base
    `cluster_classify.comp.glsl` stays light and the files can be diffed against each other:
    `cluster_classify_transient_lowtess.comp.glsl` and `cluster_classify_combined_lowtess.comp.glsl`.
  * bugfix `render_raytrace_clusters.rchit.glsl` decoded the sub-triangle index of the `<= 2x2x2` transient clusters with a
    wrong bit mask, it always resolved to zero. Not visible by default, as facet shading is enforced while tessellation is
    active and the barycentric coordinates are only used for the interpolated normals.
  * bugfix the number of part triangles wasn't clamped when the low tessellation optimizations were active, exceeding
    `MAX_PART_TRIANGLES` could cause a device loss.
  * the `<= 2x2x2` transient builds reserved storage for a completely filled batch, even though only the last batch of
    a subgroup is partially filled. They now reserve what the batch actually contains, which lowers the generated
    vertices noticeably and allows bigger batch sizes to be used at all (before, they could exceed the vertex budget
    and silently lose geometry).
  * the batch sizes of both low tessellation optimizations are exposed (`--transient2xbatch`, `--combined1xbatch`,
    `--combined2xbatch`) and may exceed the scene's cluster configuration, up to what the device supports. The
    `VkClusterAccelerationStructureTriangleClusterInputNV` limits and the pre-computed cluster sizes account for that.
    Defaults were picked from a sweep: 32 patches for the transient builds, 64 triangles and 16 patches for the
    combined templates.
  * added `tests/` with sequence file based tests that cover the cluster building configurations
    as well as running out of memory, see `tests/README.md`.
  * added headless & benchmarking support: `--headless`, `--headlessframes`, `--sequencefile`/`--sequencestring`,
    `--screenshotframegap`, `--screenshotmode`, `--sequencescreenshot`, as well as more command-line parameters for the
    renderer configuration (`--visualize`, `--lowtess1x`, `--lowtess2x`, `--lowtessmode`, `--combined1xbatch`, `--combined2xbatch`).
* 2028-8-6:
  * added "Statistics" - "Generated Clusters" UI.
* 2026-7-2:
  * new open-source tessellation table `tessellation_table_nv_raw.hpp` which no longer is directly derived from Unreal Engine's implementation. Thanks to Andrea Maggiordomo <amaggiordomo@nvidia.com> for the work on it.
* 2026-4-22:
  * expose "Tessellation: Max split factor" (TESS_MAX_SPLIT_FACTOR) to control the triangle split recursion logic.
* 2026-4-18:
  * bugfix crash when exceeding max splits (`BUILD_SETUP_SPLIT` wasn't clamping properly)
* 2026-4-17:
  * Add non-persistent triangle split kernel and make it default
  * Ignore false positive for another mesh-shader related validation error.
* 2026-3-29:
  * Avoid evaluating tessellation factors twice in `triangle_split.comp.glsl`
  * Always round via `uint(round(floatTessellationFactors))` for stable results.
* 2025-12-9:
  * Removed meshoptimizer as submodule, it is now part of nvpro_core2
  * Bugfix interpolating the barycentric positions within the tessellated triangles by adding the `precise` math keyword. Otherwise cracks may occur.
* 2025-11-11:
  * Removed deprecated `nv_cluster_lod_library`
  * Replace triangle strip optimization wtih `meshopt_optimizeMeshlet` and always run it.
* 2025-8-24:
  * Updated `meshoptimizer` submodule to `v 0.25`
* 2025-7-7:
  * Fix regression with sky rendering & missing sky UI
* 2025-6-27
  * Updated `nv_cluster_lod_library` submodule, which has new API & proper vertex limit support.
* 2025-6-23
  * Updated to use `nvpro_core2`, as result command-line arguments are now prefixed with `--` rather than just `-`. It is recommended to delete existing /_build or the CMake cache prior building or generating new solutions.
  * Updated `meshoptimizer` submodule to `v 0.24` using `meshopt_buildMeshletsSpatial` for improved ray tracing clusterization.
* 2025-1-30
  * Initial Release