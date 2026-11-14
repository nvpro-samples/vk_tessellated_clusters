# Tests

The sample has a lot of configurations and most of them only differ in how the clusters are
built, which makes them cheap to cover: every test is a **sequence file** that walks a set of
configurations, and the sample re-initializes the renderer whenever one of them changes.
The sequences are run headless, so no window is required.

``` bash
# runs every sequence file in this directory
python tests/run_tests.py

# only the ones whose filename contains "oom", and show the sample's output
python tests/run_tests.py --filter oom --verbose

# against a different build or scene
python tests/run_tests.py --exe _bin/Debug/vk_tessellated_clusters.exe --scene path/to/scene.gltf
```

Without `--exe` the newer binary of `_bin/Release` and `_bin/RelWithDebInfo` is tested, `_bin/Debug` only
when neither exists.

A test fails when the sample reports a validation error, a shader compilation error, a failed
renderer init, a lost device, or when it doesn't exit cleanly. The Vulkan validation layers are
enabled by default, use `--no-validation` to skip them. Everything after the known options is
passed on to the sample, so `python tests/run_tests.py -- --device 1` picks another GPU.

## The sequence files

| file | what it covers |
|------|----------------|
| [lowtess_modes.txt](lowtess_modes.txt)               | all permutations of the low tessellation optimizations, which selects between the three classification kernels |
| [combined_batchsizes.txt](combined_batchsizes.txt)   | the combined template batch sizes, including the values that have to be clamped |
| [tessellation_rates.txt](tessellation_rates.txt)     | tessellation rates from coarse to extreme, in every mode |
| [budgets_oom.txt](budgets_oom.txt)                   | deliberately undersized budgets, see below |
| [renderers.txt](renderers.txt)                       | rasterization and ray tracing, including switching between them |
| [dlss.txt](dlss.txt)                                 | DLSS super resolution and ray reconstruction, which change the render resolution |
| [camerapath.txt](camerapath.txt)                     | a camera path flown deterministically (`--runcamerapath`) in rasterization and ray tracing |
| [build_options.txt](build_options.txt)               | shader permutations and code paths: PN displacement, persistent split kernel, winding, HBAO, facet shading, cluster build flags, position truncation, visible cluster budget |
| [mesh_shaders.txt](mesh_shaders.txt)                 | `EXT_mesh_shader` and `NV_mesh_shader`, with and without the task shader batching |

Adding a test means adding a `.txt` file here, the runner picks it up automatically. The syntax
is the one of the parameter sequencer, `#` starts a comment:

```
SEQUENCE "name of the sequence"  --sequenceframes 8 --sequenceresetframes 2 --renderer 1
SEQUENCE "the next one"          --tessrate 2
```

Every sequence inherits the state of the previous one, only name the parameters that change.
Use `--help` on the sample for the full list of parameters.

## Out of memory coverage

All the arrays that the runtime builds into are pre-allocated, and the amount of work depends on
the camera and the tessellation rate, so running out of space is a normal condition that the
sample has to survive: it clamps and renders whatever still fits.

[budgets_oom.txt](budgets_oom.txt) provokes exactly that by combining a high tessellation rate
(`--tessrate 1`, "pixels per segment") with undersized budgets. Each budget is starved on its own
first, so that a failure points at a single counter:

| parameter | limits | counter it starves |
|-----------|--------|--------------------|
| `--partbits`    | part triangles         | `partTriangleCounter` / `dualPartTriangleCounter` |
| `--vertexbits`  | generated CLAS vertices| `genVertexCounter` |
| `--clustermegs` | generated CLAS data    | `genClusterDataCounter` |
| `--splitbits`   | recursively split triangles | `splitTriangleCounter` |

They are then starved together and repeated for the other low tessellation modes, because the
transient builds and the combined templates store their per-CLAS data at the *back* of the part
triangle array and therefore share its budget.

Run with `--verbose` to see the reported counters per sequence, a starved run shows a requested
amount well above the configured budget, for example `Part triangles; 162828;` with
`--partbits 12` (4096).
