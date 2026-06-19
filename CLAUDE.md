# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BLIR (Blender InfraRed Material Tools) — a Blender addon/script that generates infrared thermal imaging materials for 3D models. It calculates per-vertex radiation values from multiple heat sources (ambient, solar, aerodynamic heating, engine jet, CFD approximation) and applies them as Blender shader materials. Designed for aircraft thermal visualization but works with any mesh.

All core logic runs inside Blender's Python environment (`bpy` API). There is no standalone CLI, no tests, no linting.

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

- **Blender 4.2+** with its bundled Python (all `bl_IR` modules)
- **numpy** — used everywhere for vector math (Blender's Python bundles numpy)
- **tqdm** — progress bar in `batch_apply_ir.py` (must be installed into Blender's Python)
- **opencv-python** — background extraction only (standalone, uses `.venv`)

## New pipeline (`new_pipeline/`)

A second-generation temperature calculation pipeline based on `pipeline.md` steady-state heat transfer. Separate from the original `aircraft/` code — both coexist.

```
new_pipeline/
  __init__.py           — Empty
  config.py             — All parameters (T_EXHAUST=800K, T_AIRCRAFT_INIT=280K, EMISSIVITY=0.85, etc.)
  mesh_graph.py         — Face adjacency graph, edge lengths, thermal conductances, exhaust position finder
  heat_source.py        — Bisection solver for face temperature T_s (pipeline.md eq.*)
  diffusion.py          — Gauss-Seidel relaxation on mesh graph (conductance-weighted Laplacian)
  stats.py              — Per-part min/max/mean temperature output
  main.py               — Entry point: orchestrates the pipeline, prints all statistics
```

**Algorithm**: (1) Identify aircraft faces near engine exhaust → solve T_s via heat balance equation `(T_o-T)/R_N - εσ(T⁴-T_amb⁴)A_j = 0`. (2) Fix those faces as Dirichlet BCs, initialize others to 280 K. (3) Gauss-Seidel diffusion using conductance-weighted Laplacian `T_i = Σ(G_ij·T_j) / ΣG_ij` where `G_ij = k·t·L_edge/d_ij`. (4) Print per-part temperature statistics.

Run: `blender -b <model.blend> --python ./new_pipeline/main.py`

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
- `main.py` line 9 hardcodes an absolute Windows path (`D:\codes\MTIR-Blender-InfraRed-Material-Tools\aircraft`) for `sys.path`. This must be changed when running on a different machine or checkout location.
- `main.py` lines 72-73 hardcode `heat=200, decay=0.7` for engine influence instead of reading from `config.obj_names` values. The `engine_heat_delta` parameter accepted by `apply_ir_material()` is never used — engine heating is entirely determined by these hardcoded values and the in-loop jet calculation.
- `batch_apply_ir.py` does `import main` (bare module name), relying on `BL_IR_PATH` (`aircraft/`) being on `sys.path`. This is not a standard package import — it works because `setup_paths()` runs first.

The `aircraft/` directory is treated as the working directory root — `main.py`, `batch_apply_ir.py`, and `bl_IR/` all expect `aircraft/` to be on `sys.path`.
