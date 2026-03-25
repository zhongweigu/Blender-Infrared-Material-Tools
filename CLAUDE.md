# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BLIR (Blender InfraRed Material Tools) is a Blender addon/script for generating infrared thermal imaging materials on 3D models. It simulates thermal distribution on aircraft or engineering models considering solar radiation, aerodynamic heating, engine heat sources, and atmospheric effects.

## Running the Project

This is a Blender Python addon. To run:
1. Open Blender
2. Go to Scripting workspace
3. Open `aircraft/main.py` in the text editor
4. Run the script (Alt+P or clicking "Run Script")

The script applies IR materials to objects named "Aircraft", "Engin_L", and "Engin_R" defined in `config.obj_names`.

## Architecture

```
aircraft/
├── main.py                  # Entry point: applies IR material to mesh objects
└── bl_IR/                   # Core infrared module
    ├── config.py            # Physical parameters (ambient temp, emissivity, Mach number, etc.)
    ├── material.py           # Creates Blender shader materials with emission nodes driven by vertex radiation
    ├── radiation.py          # Physics: Stefan-Boltzmann, Planck law, engine jet influence, recovery temperature
    ├── location.py           # Gets engine positions from Blender scene objects
    └── camera.py             # Renders IR images from Blender

background/
└── extracting_background.py  # OpenCV-based background extraction using inpainting

早期脚本/                       # Legacy experimental scripts
```

## Key Physics Model

Radiation calculation in `main.py` (lines 38-93):
- `E_self` - Stefan-Boltzmann self-emission: εσT⁴
- `E_sun` - Solar absorption with atmospheric transmittance ~0.7
- `E_aero` - Aerodynamic heating via recovery temperature
- `E_jet` - Engine jet influence with exponential decay
- `E_total` - Sum of all sources
- Atmospheric correction using Beer-Lambert law with distance

Config flags control which effects are considered: `CONSIDER_SUN`, `CONSIDER_AERO`, `CONSIDER_CFD`, `CONSIDER_NOISE`.

## Important Configuration

All physical parameters are in `config.py`:
- `ambient_temp_C` - Ambient temperature (-50°C typical for high altitude)
- `emissivity` - Material emissivity (0.85 default)
- `MACH` - Flight Mach number (0.8 default)
- `obj_names` - Maps object names to engine heat deltas
- `METHOD` - "stefan_boltzmann" or "plank_law" for radiation calculation

## Background Extraction

`background/extracting_background.py` is a standalone OpenCV script (not Blender-dependent). It:
1. Groups masks by base filename pattern
2. Uses `cv2.inpaint` with `INPAINT_TELEA` method
3. Outputs to `backgrounds/` folder

Run separately from Blender: `python extracting_background.py` from the background directory.
