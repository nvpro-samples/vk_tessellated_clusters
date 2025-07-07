# Changelog for vk_tessellated_clusters
* 2025-7-7:
  * Fix regression with sky rendering & missing sky UI
* 2025-6-27
  * Updated `nv_cluster_lod_library` submodule, which has new API & proper vertex limit support.
* 2025-6-23
  * Updated to use `nvpro_core2`, as result command-line arguments are now prefixed with `--` rather than just `-`. It is recommended to delete existing /_build or the CMake cache prior building or generating new solutions.
  * Updated `meshoptimizer` submodule to `v 0.24` using `meshopt_buildMeshletsSpatial` for improved ray tracing clusterization.
* 2025-1-30
  * Initial Release