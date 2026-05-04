# Changelog for vk_tessellated_clusters
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