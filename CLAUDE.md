# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BLIR (Blender InfraRed Material Tools) — a Blender addon/script that generates infrared thermal imaging materials for 3D models. It calculates per-vertex radiation values from multiple heat sources (ambient, solar, aerodynamic heating, engine jet, CFD approximation) and applies them as Blender shader materials. Designed for aircraft thermal visualization but works with any mesh.

All core logic runs inside Blender's Python environment (`bpy` API). There is no standalone CLI, no tests, no linting.

Note: `README.md` is outdated — it only covers the old `aircraft/` pipeline, not the `new_pipeline/` which is the active development target.

## Repository structure

```
aircraft/
  main.py              — Single-model entry point: computes radiation per vertex, assigns IR material
  batch_apply_ir.py    — Batch processor: imports .obj files from ShapeNet.v2, applies IR material, saves .blend + optional render
  bl_IR/
    __init__.py         — Empty (package marker)
    config.py           — All tunable parameters (temperatures, emissivity, physics constants, flags)
    radiation.py        — Physics: Stefan-Boltzmann, Planck's law, atmospheric transmittance, CFD jet model, sun/aero radiation
    material.py         — Blender shader creation: Vertex Attribute → Map Range → Color Ramp → Emission → Output
    location.py         — Helper to get engine object world positions
    camera.py           — Render a Cycles PNG from a configured camera
  早期脚本/              — Early experimental scripts (Chinese-named, historical)
background/
  extracting_background.py — OpenCV-based background removal using masks + inpainting (not Blender-dependent)
```

## How to run

### Single model (inside Blender GUI)
Open `aircraft/main.py` in Blender's Scripting workspace, modify `config.py` parameters as needed, then run (Alt+P).

### Batch processing
```bash
blender -b --python ./aircraft/batch_apply_ir.py
```
Requires Blender 4.2+ and `tqdm` installed in Blender's Python. Configure paths at the top of `batch_apply_ir.py`:
- `SHAPENET_ROOT` — directory containing ShapeNet .obj files
- `OUTPUT_DIR` — where .blend and rendered PNGs go
- `BL_IR_PATH` — path to the `aircraft/` directory (for sys.path)
- `RENDER_IMAGE` — toggle render output

### Background extraction
Requires `opencv-python` and `numpy` (present in the repo's `.venv`). Run standalone (not in Blender):
```bash
cd background && python extracting_background.py
```

## Dependencies

- **Blender 4.2+** with its bundled Python (all `bl_IR` and `new_pipeline` modules)
- **numpy** — used everywhere for vector math (Blender's Python bundles numpy)
- **tqdm** — progress bar in `batch_apply_ir.py` (must be installed into Blender's Python)
- **opencv-python** — background extraction only (standalone, uses `.venv`)

For `new_pipeline` external compute (`.venv`):
- **numba** — JIT compilation of Gauss-Seidel sweep and Planck integral (significant speedup)
- **scipy** — `cKDTree` for nearest-neighbor queries (calibration, decimation, cross-boundary bridges)
- **numpy** — .venv Python also needs numpy

Create and install: `python -m venv .venv && .venv\Scripts\pip install numpy numba scipy` (or `source .venv/bin/pip` on Linux/Mac)

## New pipeline (`new_pipeline/`)

A second-generation temperature calculation pipeline based on `pipeline.md` steady-state heat transfer. Separate from the original `aircraft/` code — both coexist.

```
new_pipeline/
  __init__.py            — Empty
  config.py              — All parameters (T_EXHAUST=900K, EMISSIVITY=0.85, Q_O=9.564, DECIMATE_RATIO=1.0, DIFFUSION_DECAY=0, etc.)
  mesh_graph.py          — Face adjacency graph (CSR), edge lengths, exhaust position finder, find_aircraft/engine objects
  heat_source.py         — Bisection solver for face temperature T_s (pipeline.md eq.13, in-Blender variant)
  diffusion.py           — Gauss-Seidel relaxation on mesh graph (conductance-weighted Laplacian), CSR builder
  calibrate_compute.py   — Pure-numpy/numba numerics: bisection, radiance (Planck integral), atmospheric attenuation, environment radiation, Gauss-Seidel sweep, cross-boundary bridge finding, connectivity repair. No Blender dependency.
  compute_standalone.py  — External compute entry point: reads .npz, runs full pipeline (heat source → diffusion → aero → radiance → detector), writes results. Called by main.py as subprocess.
  calibrate_qo.py        — Calibration entry point: exports mesh → spawns calibrate_compute.py → bisection search for Q_O. Must be run before main.py.
  io_mesh.py             — Blender ↔ .npz serialization for external compute (CSR, engine mask, normals, all config)
  visualize.py           — Render per-face values (T or L) via Blender Eevee: per-vertex averaging, Color Ramp shader, camera setup, multi-view renders, wireframe/solid renders
  stats.py               — Per-part min/max/mean temperature statistics output
  main.py                — Entry point: finds objects, merges+scales mesh, runs compute (external or in-Blender), upsamples results, saves process images + renders
```

### Calibration (must be done first)

Before running `main.py`, `Q_O` (engine heat power per connecting face) must be calibrated:

```bash
blender -b <model.blend> --python ./new_pipeline/calibrate_qo.py
```

This exports the mesh to `.npz`, spawns `calibrate_compute.py` via `.venv` Python, and bisection-searches `q_o` to make the engine surface mean temperature ≈ 350 K. Copy the printed `q_o` value into `config.py` as `Q_O`.

**Important**: `Q_O` is model-specific — it depends on the mesh geometry and must be recalibrated for each new model or if the engine/aircraft mesh topology changes significantly.

**Requires**: `.venv` with `numpy`, `numba`, and `scipy` (for cKDTree).

### Main run

```bash
blender -b <model.blend> --python ./new_pipeline/main.py
```

### Batch processing

Process many ShapeNetCore.v2 models at once:

```bash
blender -b --python ./new_pipeline/batch_process.py
```

Parameters at the top of `batch_process.py`:
- `INPUT_ROOT` — root directory with `Airliner/` and `Cargo/` subfolders
- `OUTPUT_ROOT` — where processed `.blend` files are saved (default `.\output\{id}\aircraft.blend`)
- `CATEGORIES` — which subfolders to scan
- `MAX_COUNT` — max models to process (`None` = unlimited)
- `SKIP_EXISTING` — skip models that already have output

The batch script handles ShapeNet object naming automatically: mesh objects are sorted by vertex count (largest → Aircraft, 2nd → Engin_L, 3rd → Engin_R). Each model is saved as `OUTPUT_ROOT/{id}/models/aircraft.blend` with IR material applied.

### External vs. in-Blender compute

`config.USE_EXTERNAL_COMPUTE` (default `True`) controls the compute path:

- **External** (`True`): `main.py` exports the unified mesh to `.npz`, spawns `compute_standalone.py` in `.venv` Python (with numba JIT for Gauss-Seidel), then imports results. Much faster for large meshes.
- **In-Blender** (`False`): All computation runs inside Blender's Python via `heat_source.py` + `diffusion.py`. No `.venv` or numba needed. Fallback if external compute fails.

Both paths produce identical results — they share the same numerics (`calibrate_compute.py` functions for radiance, environment radiation, etc., or their `heat_source.py` equivalents).

### Mesh decimation

`config.DECIMATE_RATIO` (default 1.0 = no decimation; 0.15 = keep 15% of faces) controls optional mesh decimation before compute. The temperature field is computed on the decimated mesh then upsampled back to the original mesh via nearest-neighbor. This dramatically reduces compute time for high-poly models.

### Process images

When `config.PROCESS_IMAGES_ENABLED = True`, the pipeline saves 5 intermediate PNGs to `config.PROCESS_IMAGES_DIR`:
1. Solid shaded aircraft (scaled)
2. Temperature after diffusion
3. Temperature after aero heating
4. Radiance after Planck conversion
5. Final radiance after sensor energy degradation

The wireframe render function (`_render_wireframe`) exists in `main.py` but is not called in the current process image flow.

### Algorithm

(1) Identify aircraft faces near engine exhaust → solve T_s via heat balance equation `(T_o-T)/R_N + q_o - εσ(T⁴-T_amb⁴)A_j = 0`. (2) Fix those faces as Dirichlet BCs, initialize others to `T_AIRCRAFT_INIT` (280 K). (3) Add cross-boundary structural bridges between engine and skin faces. (4) Optionally merge close vertices (`MERGE_VERTEX_DIST`) to eliminate disconnected seams. (5) `ensure_connectivity`: bridge any remaining disconnected components so heat can reach all faces. (6) Gauss-Seidel diffusion using arithmetic-mean weights (structural edges weighted by `typical_len / distance`), with optional decay toward T_amb via `DIFFUSION_DECAY`. (7) Aerodynamic heating `ΔT = T_init · 0.16 · M²`. (8) Temperature → radiance via Planck integral approximation (pipeline.md §9 eq.7). (9) Optional environment reflection radiance (pipeline.md §10). (10) Optical system energy degradation (pipeline.md §13 eq.4). (11) Atmosphere + detector directional factor (computed in standalone, applied in visualizer).

## Architecture notes

The data flow is: **config parameters → radiation calculation → per-vertex attribute → shader material**

- `config.py` is the single source of truth for all tunable parameters. Temperature deltas, physics constants, flags like `CONSIDER_SUN`/`CONSIDER_AERO`/`CONSIDER_CFD`, and output mode all live here. `obj_names` maps object names to engine heat delta values (K).
- `radiation.py` `calculate()` dispatches to either `stefan_boltzmann()` (εσT⁴) or `plank_law()` (spectral radiance at 10μm) based on `config.METHOD`. All radiation values are in W/m².
- `material.py` `assign()` creates a shader node tree that reads a `"Radiation"` vertex attribute, maps it from a computed min-max range to 0–1, feeds a Color Ramp (blue-yellow-red for `OUTPUT_MODE=0`, black-white for `OUTPUT_MODE=1`), and emits. `GLOBAL_MIN`/`GLOBAL_MAX` constants in this file are unused — min/max are computed dynamically from ambient temp.
- `main.py` iterates over mesh vertices in world space, accumulating E_self + E_sun + E_aero + E_jet, then applies atmospheric transmission correction (`tau * E * geom + (1-tau) * E_bg`). The CFD term overrides the total if higher. `apply_ir_material()` accepts an `engine_heat_delta` parameter but the function body does not use it — engine heating comes from `config.obj_names` values and the in-loop jet calculation.
- `batch_apply_ir.py` imports `main.apply_ir_material()` directly. It merges multi-part .obj imports into a single mesh named "Aircraft", disables aero/engine for ShapeNet models, and saves one .blend per input model.

### Known quirks

- `config.TAU` (0.85) is defined but unused. Atmospheric transmission uses `config.KAPPA` with Beer-Lambert: `tau = exp(-kappa * R)`, computed per-vertex based on camera distance.
- `E_jet` in `main.py` is gated by `CONSIDER_AERO` (line 71), not a separate engine flag. This means disabling aero also disables jet heating, which may not be intentional.
- `radiation.py` imports `bpy`, `sys`, and `os` solely to compute `script_dir = os.path.dirname(bpy.data.filepath)` and add it to `sys.path`, then imports `config` as a relative fallback. When run from `main.py` or `batch_apply_ir.py` (which already set up `sys.path`), this is redundant but harmless.
- `aircraft/main.py` line 9 hardcodes an absolute Windows path (`D:\codes\MTIR-Blender-InfraRed-Material-Tools\aircraft`) for `sys.path`. This must be changed when running on a different machine or checkout location.
- `new_pipeline/config.py` hardcodes `PROJECT_ROOT = r"D:\codes\MTIR-Blender-InfraRed-Material-Tools"`. However, `new_pipeline/main.py` and `calibrate_qo.py` auto-detect the project root from `__file__` or the `.blend` file path — this constant is only a fallback used when running from Blender's Text Editor where `__file__` is unavailable.
- `main.py` lines 72-73 hardcode `heat=200, decay=0.7` for engine influence instead of reading from `config.obj_names` values. The `engine_heat_delta` parameter accepted by `apply_ir_material()` is never used — engine heating is entirely determined by these hardcoded values and the in-loop jet calculation.
- `batch_apply_ir.py` does `import main` (bare module name), relying on `BL_IR_PATH` (`aircraft/`) being on `sys.path`. This is not a standard package import — it works because `setup_paths()` runs first.

The `aircraft/` directory is treated as the working directory root — `main.py`, `batch_apply_ir.py`, and `bl_IR/` all expect `aircraft/` to be on `sys.path`.

### Test data

`aircraft/` contains several `.blend` files used as test/demo models:
- `aircraft.blend` / `aircraft_1.blend` — primary test models with named parts (Aircraft, Engin_L, Engin_R)
- `obj.blend` / `obj2.blend` — single-object imports
- `波音777客机飞机.blend` — Boeing 777 reference model

`debug_wireframe.py` (untracked) is a development utility for wireframe rendering debug.

### Pipeline theory documents

The root directory has per-stage pipeline documents that feed into `pipeline.md` (the combined spec):
- `temperature_pipeline.md` — steady-state heat transfer model
- `self_radiation_pipeline.md` — temperature-to-radiance conversion (Planck's law)
- `atmosphere_transfer_pipeline.md` — Bouguer-Lambert atmospheric attenuation
- `environment_pipeline.md` — solar/sky/ground reflection radiation
- `detector_pipeline.md` — detector directional factor and optical degradation
- `clutter_noise_pipeline.md` — detector noise (Poisson + Gaussian)
- `grey-level_pipeline.md` — radiance-to-grayscale linear mapping

`pipeline.md` unifies all stages into a single comprehensive spec for the `new_pipeline/` implementation.
