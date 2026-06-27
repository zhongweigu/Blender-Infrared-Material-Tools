"""
Batch launcher — spawns a fresh Blender process for each model.

Usage:
    python batch_launcher.py

Each model runs in its own blender -b process with main.py.
Complete process isolation — no module caching, no state pollution.
"""

import os
import sys
import subprocess
import time
import glob

# ═══════════════════════════════════════════════════════════
# Configurable
# ═══════════════════════════════════════════════════════════

# Blender executable
BLENDER = r"D:\SteamLibrary\steamapps\common\Blender\blender.exe"  # <-- UPDATE THIS PATH

# main.py path (relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(SCRIPT_DIR, "new_pipeline", "main.py")

# Input: ShapeNetCore.v2 root
INPUT_ROOT = r"D:\BaiduNetdiskDownload\ShapeNetCore.v2\total"

# Output root
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output")

# Categories to process
CATEGORIES = ["Airliner", "Cargo"]

# Max models to process (None = all)
MAX_COUNT = 10

# Skip existing output
SKIP_EXISTING = False

# Blender timeout per model (seconds)
TIMEOUT = 600

# ═══════════════════════════════════════════════════════════


def find_models():
    """Scan INPUT_ROOT/CATEGORY/*/aircraft.blend, return [(model_id, path), ...]."""
    models = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(INPUT_ROOT, cat)
        if not os.path.isdir(cat_dir):
            print(f"[launcher] WARNING: not a directory — {cat_dir}")
            continue
        for folder in sorted(os.listdir(cat_dir)):
            p = os.path.join(cat_dir, folder, "aircraft.blend")
            if os.path.isfile(p):
                models.append((folder, p))
    return models


def process_one(model_id, blend_path, output_path):
    """Launch blender -b for one model."""
    env = os.environ.copy()
    env['BLIR_OUTPUT_PATH'] = output_path
    env['PYTHONIOENCODING'] = 'utf-8'

    cmd = [BLENDER, '-b', blend_path, '--python', MAIN_SCRIPT]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            encoding='utf-8',
            errors='replace',
            env=env,
        )
        return result.returncode == 0 and os.path.isfile(output_path), result
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {TIMEOUT}s")
        return False, None
    except FileNotFoundError:
        print(f"  Blender not found: {BLENDER}")
        return False, None


def main():
    if not os.path.isfile(BLENDER):
        print(f"[launcher] ERROR: Blender not found at {BLENDER}")
        print(f"  Please update the BLENDER path in this script.")
        sys.exit(1)
    if not os.path.isfile(MAIN_SCRIPT):
        print(f"[launcher] ERROR: main.py not found at {MAIN_SCRIPT}")
        sys.exit(1)

    models = find_models()
    print(f"[launcher] Found {len(models)} models in {CATEGORIES}")

    if MAX_COUNT is not None:
        models = models[:MAX_COUNT]
        print(f"[launcher] Capped at {MAX_COUNT}, processing {len(models)}")

    print(f"[launcher] Blender: {BLENDER}")
    print(f"[launcher] Output: {OUTPUT_ROOT}")
    print()

    succeeded = failed = skipped = 0
    t_start = time.time()

    for i, (model_id, blend_path) in enumerate(models):
        output_path = os.path.join(OUTPUT_ROOT, model_id, "models",
                                    "aircraft.blend")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if SKIP_EXISTING and os.path.isfile(output_path):
            skipped += 1
            print(f"[{i+1}/{len(models)}] {model_id}  SKIP (exists)")
            continue

        t0 = time.time()
        print(f"[{i+1}/{len(models)}] {model_id}  processing...", end=' ',
              flush=True)

        ok, result = process_one(model_id, blend_path, output_path)
        elapsed = time.time() - t0

        if ok:
            succeeded += 1
            print(f"OK  ({elapsed:.0f}s)")
        else:
            failed += 1
            print(f"FAIL  ({elapsed:.0f}s)")
            if result and result.stderr:
                # Show last few lines of stderr
                lines = result.stderr.strip().splitlines()
                for line in lines[-5:]:
                    print(f"    {line}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"[launcher] DONE  ok={succeeded}  fail={failed}  skip={skipped}"
          f"  total={len(models)}  {elapsed:.0f}s")
    print(f"[launcher] Output → {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
