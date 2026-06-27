"""
Batch processing: apply new_pipeline IR material to ShapeNet aircraft models.

Walks INPUT_ROOT/<Airliner|Cargo>/<id>/aircraft.blend, adapts object names,
runs the full pipeline inline (no dependency on new_pipeline.main),
and saves results to OUTPUT_ROOT/<id>/models/aircraft.blend.

Usage:
    blender -b --python ./new_pipeline/batch_process.py
"""

import os
import sys
import time
import subprocess
import tempfile
import bpy
import numpy as np

# ── Ensure project root stays in sys.path across open_mainfile() calls ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── tqdm (optional) ──
try:
    from tqdm import tqdm
    HAS_TQDM = True
    def _log(*args, **kwargs):
        tqdm.write(" ".join(str(a) for a in args), **kwargs)
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable
    def _log(*args, **kwargs):
        print(*args, **kwargs)


# ═══════════════════════════════════════════════════════════
# Configurable
# ═══════════════════════════════════════════════════════════

INPUT_ROOT = r"D:\BaiduNetdiskDownload\ShapeNetCore.v2\total"
OUTPUT_ROOT = r".\output"
CATEGORIES = ["Airliner", "Cargo"]
MAX_COUNT = 10          # None = unlimited
SKIP_EXISTING = False

# ═══════════════════════════════════════════════════════════


def _resolve(root):
    """Resolve a path that may be relative to the project directory."""
    if os.path.isabs(root):
        return os.path.normpath(root)
    return os.path.normpath(os.path.join(_PROJECT_ROOT, root))


def find_models():
    """Scan INPUT_ROOT/CATEGORY/*/aircraft.blend, return [(model_id, path), ...]."""
    models = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(INPUT_ROOT, cat)
        if not os.path.isdir(cat_dir):
            print(f"[batch] WARNING: not a directory — {cat_dir}")
            continue
        for folder in sorted(os.listdir(cat_dir)):
            p = os.path.join(cat_dir, folder, "aircraft.blend")
            if os.path.isfile(p):
                models.append((folder, p))
    return models


def adapt_object_names():
    """Rename mesh objects so the pipeline can find them."""
    from new_pipeline.mesh_graph import find_object

    renamed = {}

    body_candidates = ["Aircraft", "aircraft", "Airliner", "airliner",
                       "AirCraft", "body", "Body", "fuselage", "Fuselage",
                       "plane", "Plane", "AIRFRAME",
                       "model_normalized", "Model_Normalized", "Model",
                       "air"]
    body = find_object(body_candidates)

    engine_patterns = ["engin", "engine", "eng_", "eng ", "motor", "jet",
                       "eng",
                       "model_normalized.001", "model_normalized.002",
                       "model_normalized.003", "model_normalized.004",
                       "Model_Normalized.001", "Model_Normalized.002"]

    eng_candidates = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if body and obj == body:
            continue
        name_lower = obj.name.lower()
        if any(pat.lower() in name_lower for pat in engine_patterns):
            eng_candidates.append(obj)

    if not eng_candidates:
        meshes = sorted(
            [(obj, len(obj.data.vertices)) for obj in bpy.data.objects if obj.type == 'MESH'],
            key=lambda x: x[1], reverse=True,
        )
        start = 1 if body is not None else 0
        eng_candidates = [obj for obj, _ in meshes[start:]]

    def _centroid_x(obj):
        verts = obj.data.vertices
        mw = obj.matrix_world
        return sum((mw @ v.co).x for v in verts) / len(verts)

    left, right = [], []
    for obj in eng_candidates:
        x = _centroid_x(obj)
        if x < 0:
            left.append((x, obj))
        else:
            right.append((x, obj))
    left.sort(key=lambda p: p[0])
    right.sort(key=lambda p: p[0])

    renames = []
    if body is not None and body.name != "Aircraft":
        renames.append((body, "Aircraft"))
    for side, eng_list in [('L', left), ('R', right)]:
        for idx, (_, obj) in enumerate(eng_list):
            if idx == 0:
                target = f"Engin_{side}"
            else:
                target = f"Engin_{side}_{idx + 1}"
            if obj.name != target:
                renames.append((obj, target))

    for i, (obj, _target) in enumerate(renames):
        obj.name = f"_tmp_rn_{i}"
    for obj, target in renames:
        if target in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[target], do_unlink=True)
        old_name = obj.name
        obj.name = target
        renamed[old_name] = target

    if renamed:
        _log(f"[batch]   renamed: {renamed}")

    n_mesh = len([obj for obj in bpy.data.objects if obj.type == 'MESH'])
    return n_mesh


def clear_scene():
    """Remove all data blocks so the next open_mainfile starts clean."""
    for coll in list(bpy.data.collections):
        for obj in list(coll.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    for tp in (bpy.data.meshes, bpy.data.materials, bpy.data.textures,
               bpy.data.images, bpy.data.lights, bpy.data.cameras,
               bpy.data.curves, bpy.data.node_groups):
        for item in list(tp):
            tp.remove(item)


# ═══════════════════════════════════════════════════════════
# Pipeline helpers (inlined from main.py to avoid module cache issues)
# ═══════════════════════════════════════════════════════════

def _select_only(obj):
    """Select only *obj* (deselect all others)."""
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _copy_obj(obj):
    """Duplicate a mesh object via API (no operators)."""
    mesh = obj.data.copy()
    dup = bpy.data.objects.new(obj.name + "_dup", mesh)
    dup.matrix_world = obj.matrix_world.copy()
    bpy.context.collection.objects.link(dup)
    return dup


def _write_vertex_attr(mesh, attr_name, vert_values):
    """Write a per-vertex float attribute to a mesh data block."""
    if attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name])
    attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
    attr.data.foreach_set('value', vert_values.tolist())


def _engine_name(side, idx):
    """Canonical engine name."""
    if idx == 0:
        return f"Engin_{side}"
    return f"Engin_{side}_{idx + 1}"


def _find_venv_python():
    """Locate the .venv Python interpreter."""
    venv_dir = os.path.join(_PROJECT_ROOT, ".venv")
    if not os.path.isdir(venv_dir):
        return None
    for sub in ("Scripts", "bin"):
        for name in ("python.exe", "python", "python3"):
            p = os.path.join(venv_dir, sub, name)
            if os.path.isfile(p):
                return p
    return None


def _upsample_fallback(src_c, src_t, dst_c):
    """Nearest-neighbor upsample from src to dst centers."""
    n_dst = len(dst_c)
    out = np.empty(n_dst, dtype=np.float64)
    for b0 in range(0, n_dst, 1000):
        b1 = min(b0 + 1000, n_dst)
        d2 = np.sum((dst_c[b0:b1, None, :] - src_c[None, :, :]) ** 2, axis=2)
        out[b0:b1] = np.asarray(src_t, dtype=np.float64)[np.argmin(d2, axis=1)]
    return out


# ═══════════════════════════════════════════════════════════
# External compute
# ═══════════════════════════════════════════════════════════

def _run_external_compute(merged, exhaust_positions, engine_mask,
                          project_root, io_mesh):
    """Export → subprocess compute_standalone.py → import results."""
    python_exe = _find_venv_python()
    if python_exe is None:
        print("[外部计算] .venv 未找到，回退到 Blender 内计算")
        return (None, None, 0, 0, None, None, None, None)

    standalone = os.path.join(project_root, "new_pipeline",
                              "compute_standalone.py")
    if not os.path.isfile(standalone):
        print(f"[外部计算] 脚本未找到: {standalone}")
        return (None, None, 0, 0, None, None, None, None)

    tmpdir = tempfile.gettempdir()
    input_npz = os.path.join(tmpdir, "_blir_batch_input.npz")
    output_npz = os.path.join(tmpdir, "_blir_batch_output.npz")

    try:
        print("\n[外部计算] 导出统一网格...")
        t_export = time.time()
        io_mesh.export_unified_mesh(input_npz, merged,
                                     exhaust_positions, engine_mask)
        print(f"  导出耗时: {time.time() - t_export:.1f} s")

        print(f"\n[外部计算] 启动外部进程: {python_exe}")
        t_run = time.time()
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(
            [python_exe, standalone, input_npz, output_npz],
            capture_output=True, text=True, timeout=600,
            encoding='utf-8', errors='replace', env=env,
        )
        elapsed = time.time() - t_run
        print(f"  外部进程耗时: {elapsed:.1f} s")

        if result.stdout:
            for line in result.stdout.strip().splitlines():
                print(f"  {line}")

        if result.returncode != 0:
            print(f"[外部计算] 进程失败 (code={result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            return (None, None, 0, 0, None, None, None, None)

        res = io_mesh.import_results(output_npz)
        if res is None:
            return (None, None, 0, 0, None, None, None, None)
        return (res['T'], res['L'], res['iterations'], res['max_change'],
                engine_mask,
                res.get('T_diffusion'), res.get('T_aero'),
                res.get('L_radiance'))

    except FileNotFoundError:
        print(f"[外部计算] Python 未找到: {python_exe}")
        return (None, None, 0, 0, None, None, None, None)
    except subprocess.TimeoutExpired:
        print("[外部计算] 进程超时 (600s)")
        return (None, None, 0, 0, None, None, None, None)
    except Exception as e:
        print(f"[外部计算] 异常: {e}")
        return (None, None, 0, 0, None, None, None, None)
    finally:
        for f in (input_npz, output_npz):
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════
# In-Blender compute
# ═══════════════════════════════════════════════════════════

def _run_in_blender(merged, exhaust_positions, engine_mask,
                    config, mesh_graph, heat_source, diffusion, calibrate_compute):
    """Compute temperature field entirely inside Blender (pure Python)."""
    print("\n[Blender] 构建面片邻接图...")
    centers, areas, _ = mesh_graph.get_mesh_data(merged)
    neighbors, edge_lengths = mesh_graph.build_face_adjacency(merged)

    n_faces = len(centers)
    print(f"  面片总数: {n_faces}")
    print(f"  发动机面片: {engine_mask.sum()}, 蒙皮面片: {(~engine_mask).sum()}")

    # ── Heat source faces ──
    print("\n[Blender] 识别热源面片并求解 T_s...")
    n_engine = int(engine_mask.sum())
    n_source_per_exhaust = max(5, min(200, int(n_engine * 0.05)))
    engine_indices = np.where(engine_mask)[0]

    source_faces = set()
    for ep in exhaust_positions:
        dists = np.linalg.norm(centers[engine_indices] - ep, axis=1)
        nearest = engine_indices[np.argsort(dists)[:n_source_per_exhaust]]
        source_faces.update(nearest.tolist())

    print(f"  热源面片: {len(source_faces)} "
          f"(每尾焰 {n_source_per_exhaust}, 发动机共 {n_engine} 面)")

    T_source_dict = heat_source.solve_all_source_faces(
        list(source_faces), centers, exhaust_positions, areas,
        T_o=config.T_EXHAUST, q_o=config.Q_O
    )

    T_s_vals = np.array(list(T_source_dict.values()))
    print(f"  T_s: [{T_s_vals.min():.0f}, {T_s_vals.max():.0f}] K "
          f"mean={T_s_vals.mean():.0f} K")

    # ── Gauss-Seidel ──
    print(f"\n[Blender] Gauss-Seidel 扩散 "
          f"(tol={config.DIFFUSION_TOL} K, max_iter={config.MAX_ITERATIONS})...")

    cross_pairs = mesh_graph.find_cross_boundary_pairs(
        centers, engine_mask, max_pairs=3, max_distance=2.0)
    for ei, si, dist in cross_pairs:
        neighbors[ei].append(si)
        neighbors[si].append(ei)
        edge_lengths[(ei, si)] = -dist
        edge_lengths[(si, ei)] = -dist

    mesh_graph.ensure_connectivity(neighbors, edge_lengths, centers, source_faces)

    T = np.full(n_faces, config.T_AIRCRAFT_INIT, dtype=np.float64)
    for fi, T_s in T_source_dict.items():
        T[fi] = T_s

    conductances = {}
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if (i, j) in conductances:
                continue
            conductances[(i, j)] = 1.0
            conductances[(j, i)] = 1.0

    T, iterations, max_change = diffusion.gauss_seidel(
        T, neighbors, conductances,
        fixed_faces=source_faces,
        tol=config.DIFFUSION_TOL,
        max_iter=config.MAX_ITERATIONS,
        decay=config.DIFFUSION_DECAY,
        T_amb=config.T_AMB,
    )

    print(f"  完成: {iterations} 次迭代, 最终 max ΔT = {max_change:.6f} K")
    print(f"  扩散后 T range: [{T.min():.1f}, {T.max():.1f}] K")

    T_diffusion = T.copy()

    # ── Aero heating ──
    delta_T_aero = config.T_AIRCRAFT_INIT * 0.16 * config.MACH_NUMBER ** 2
    T += delta_T_aero
    print(f"\n[Blender] 气动加热: M={config.MACH_NUMBER}, ΔT = +{delta_T_aero:.2f} K")
    print(f"  T range: [{T.min():.1f}, {T.max():.1f}] K  mean={T.mean():.1f} K")

    T_aero = T.copy()

    # ── Temperature → Self Radiance ──
    print(f"\n[Blender] 温度→自身辐亮度 (波段 "
          f"{config.LAMBDA_1*1e6:.0f}-{config.LAMBDA_2*1e6:.0f} μm)...")
    L_self = calibrate_compute.compute_radiance(
        T, config.EMISSIVITY, config.LAMBDA_1, config.LAMBDA_2)
    print(f"  L_self range: [{L_self.min():.2f}, {L_self.max():.2f}] W/(m²·sr) "
          f"mean={L_self.mean():.2f}")

    # ── Normals ──
    _, _, face_verts = mesh_graph.get_mesh_data(merged)
    normals = np.empty((n_faces, 3), dtype=np.float64)
    for i, fv in enumerate(face_verts):
        v0, v1, v2 = np.asarray(fv[0]), np.asarray(fv[1]), np.asarray(fv[2])
        nrm = np.cross(v1 - v0, v2 - v0)
        nlen = np.linalg.norm(nrm)
        if nlen > 1e-9:
            nrm /= nlen
        normals[i] = nrm

    # ── Environment reflection ──
    if config.ENV_RADIATION_ENABLED:
        env_config = {
            'I0': config.SUN_CONSTANT,
            'P': config.ATM_TRANSPARENCY,
            'h': config.SUN_ELEVATION,
            'azimuth': config.SUN_AZIMUTH,
            'n_day': config.DAY_NUMBER,
            'e': config.WATER_VAPOR_PRESSURE,
            'T_air': config.AIR_TEMPERATURE,
            'f_fi': config.EARTH_ANGLE_COEFF,
            'alpha_1': config.ALPHA_1,
            'sigma': config.SIGMA,
        }
        print(f"\n[Blender] 环境反射辐亮度...")
        L_refl = calibrate_compute.compute_environment_radiance(
            centers, normals, config.EMISSIVITY, env_config)
        print(f"  L_refl range: [{L_refl.min():.2f}, {L_refl.max():.2f}] W/(m²·sr) "
              f"mean={L_refl.mean():.2f}")
        L = L_self + L_refl
    else:
        L = L_self
    print(f"  L_total range: [{L.min():.2f}, {L.max():.2f}] W/(m²·sr) "
          f"mean={L.mean():.2f}")

    L_radiance = L.copy()

    # ── Energy degradation ──
    tau0 = config.TAU0
    Ke = config.K_E
    beta_ratio = config.BETA_RATIO
    denom = 4.0 * Ke * Ke * (1.0 - beta_ratio) ** 2
    if denom < 1e-9:
        denom = 1e-9
    eta = tau0 * np.pi / denom
    L = L * eta
    print(f"\n[Blender] 光学能量衰减: τ₀={tau0}, K_e={Ke}, "
          f"β'/β_p={beta_ratio}, η={eta:.4f}")
    print(f"  L range: [{L.min():.2f}, {L.max():.2f}] W/(m²·sr)")

    return (T, L, iterations, max_change, engine_mask, source_faces,
            T_diffusion, T_aero, L_radiance)


# ═══════════════════════════════════════════════════════════
# Per-model processing (self-contained, no import of main)
# ═══════════════════════════════════════════════════════════

def process_one(model_id, blend_path, output_path):
    """Full IR pipeline for one model — no dependency on new_pipeline.main."""
    t0 = time.time()

    bpy.ops.wm.open_mainfile(filepath=blend_path)

    # open_mainfile() may reset sys.path
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    n = adapt_object_names()
    if n == 0:
        _log(f"[batch]   ERROR: no mesh objects")
        return False
    _log(f"[batch]   {n} mesh object(s)")

    # ── Import pipeline modules ──
    from new_pipeline import config
    from new_pipeline import mesh_graph
    from new_pipeline import heat_source
    from new_pipeline import diffusion
    from new_pipeline import visualize
    from new_pipeline import io_mesh
    from new_pipeline import calibrate_compute

    # ── 0. Find objects ──
    aircraft = mesh_graph.find_aircraft()
    engines_left, engines_right = mesh_graph.find_all_engines()
    all_engines = engines_left + engines_right

    if aircraft is None:
        _log(f"[batch]   ERROR: no aircraft mesh")
        return False

    ac_n_orig = len(aircraft.data.polygons)
    ac_n = ac_n_orig
    eng_face_counts = [len(eng.data.polygons) for eng in all_engines]

    exhaust_positions_model = []
    for eng in all_engines:
        pos = mesh_graph.find_exhaust_position(eng)
        if pos is not None:
            exhaust_positions_model.append(np.array(pos))

    if not exhaust_positions_model:
        _log(f"[batch]   ERROR: no exhaust positions")
        return False
    if config.Q_O is None:
        _log(f"[batch]   ERROR: Q_O not calibrated")
        return False

    _log(f"[batch]   Aircraft='{aircraft.name}' ({ac_n} faces)")

    # ── 1. Prepare mesh ──
    ac_dup = _copy_obj(aircraft)
    eng_copies = [_copy_obj(eng) for eng in all_engines]

    if config.MERGE_VERTEX_DIST > 0:
        _select_only(ac_dup)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=config.MERGE_VERTEX_DIST)
        bpy.ops.object.mode_set(mode='OBJECT')
        ac_n = len(ac_dup.data.polygons)

    aircraft.hide_viewport = True
    aircraft.hide_render = True
    for eng in all_engines:
        eng.hide_viewport = True
        eng.hide_render = True

    # ── 2. Join ──
    _select_only(ac_dup)
    for ec in eng_copies:
        ec.select_set(True)
    bpy.ops.object.join()

    merged = ac_dup
    merged.name = "IR_Unified_Mesh"

    # ── 3. Scale ──
    if config.MODEL_SCALE != 1.0:
        _select_only(merged)
        merged.scale *= config.MODEL_SCALE
        bpy.ops.object.transform_apply(scale=True)

    exhaust_positions = [ep * config.MODEL_SCALE
                         for ep in exhaust_positions_model]

    # ── Symmetrize ──
    if config.SYMMETRIZE_MESH:
        mesh_graph.symmetrize_mesh(merged)

    # ── Build engine_mask ──
    n_faces = len(merged.data.polygons)
    engine_mask = np.zeros(n_faces, dtype=bool)
    offset = ac_n
    for fc in eng_face_counts:
        if fc > 0:
            engine_mask[offset:offset + fc] = True
            offset += fc

    # ── Decimate (optional) ──
    compute_mesh = merged
    compute_engine_mask = engine_mask
    decimated = None
    is_decimated = False
    dec_centers = None

    if config.DECIMATE_RATIO < 1.0:
        _log(f"[batch]   Decimating ratio={config.DECIMATE_RATIO}...")
        orig_centers, _, _ = mesh_graph.get_mesh_data(merged)
        orig_eng_idx = np.where(engine_mask)[0]

        decimated = _copy_obj(merged)
        decimated.name = merged.name + "_decimated"
        is_decimated = True

        _select_only(decimated)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.decimate(ratio=config.DECIMATE_RATIO)
        bpy.ops.object.mode_set(mode='OBJECT')

        dec_centers, _, _ = mesh_graph.get_mesh_data(decimated)
        _log(f"[batch]   Decimated: {len(dec_centers)} faces (was {n_faces})")

        compute_engine_mask = np.zeros(len(dec_centers), dtype=bool)
        if len(orig_eng_idx) > 0:
            if mesh_graph.cKDTree is not None:
                tree = mesh_graph.cKDTree(dec_centers)
                _, nearest = tree.query(orig_centers[orig_eng_idx])
                compute_engine_mask[np.unique(nearest)] = True
            else:
                for ei in orig_eng_idx:
                    d = np.sum((dec_centers - orig_centers[ei]) ** 2, axis=1)
                    compute_engine_mask[np.argmin(d)] = True
        _log(f"[batch]   Engine faces mapped: {compute_engine_mask.sum()}")

        compute_mesh = decimated

    # ── 4. Compute ──
    source_faces = set()
    L = None
    T_diffusion = T_aero = L_radiance = None

    if config.USE_EXTERNAL_COMPUTE:
        (T, L, iterations, max_change, _,
         T_diffusion, T_aero, L_radiance) = \
            _run_external_compute(compute_mesh, exhaust_positions,
                                   compute_engine_mask, _PROJECT_ROOT, io_mesh)
        if T is not None:
            _log(f"[batch]   External compute OK")
            n_eng = int(compute_engine_mask.sum())
            n_src = max(5, min(200, int(n_eng * 0.05)))
            eng_idx = np.where(compute_engine_mask)[0]
            cc, _, _ = mesh_graph.get_mesh_data(compute_mesh)
            for ep in exhaust_positions:
                dists = np.linalg.norm(cc[eng_idx] - ep, axis=1)
                nearest = eng_idx[np.argsort(dists)[:n_src]]
                source_faces.update(nearest.tolist())
        else:
            _log(f"[batch]   External compute failed, falling back to Blender")
            (T, L, iterations, max_change, _, source_faces,
             T_diffusion, T_aero, L_radiance) = \
                _run_in_blender(compute_mesh, exhaust_positions,
                                 compute_engine_mask,
                                 config, mesh_graph, heat_source,
                                 diffusion, calibrate_compute)
    else:
        (T, L, iterations, max_change, _, source_faces,
         T_diffusion, T_aero, L_radiance) = \
            _run_in_blender(compute_mesh, exhaust_positions,
                             compute_engine_mask,
                             config, mesh_graph, heat_source,
                             diffusion, calibrate_compute)

    # ── Upsample ──
    if is_decimated:
        _log(f"[batch]   Upsampling {len(T)} → {n_faces} faces...")
        merged_centers, _, _ = mesh_graph.get_mesh_data(merged)

        T = _upsample_fallback(dec_centers, T, merged_centers)
        L = _upsample_fallback(dec_centers, L, merged_centers)
        if T_diffusion is not None:
            T_diffusion = _upsample_fallback(dec_centers, T_diffusion, merged_centers)
        if T_aero is not None:
            T_aero = _upsample_fallback(dec_centers, T_aero, merged_centers)
        if L_radiance is not None:
            L_radiance = _upsample_fallback(dec_centers, L_radiance, merged_centers)

        mesh_graph.cleanup_decimated(decimated, is_decimated)

    # ── 5. Stats ──
    delta_T_aero = config.T_AIRCRAFT_INIT * 0.16 * config.MACH_NUMBER ** 2
    _log(f"[batch]   T: [{T.min():.1f}, {T.max():.1f}] K  "
          f"iter={iterations}  ΔT_aero={delta_T_aero:.1f}K")

    skin_mask = ~engine_mask
    if skin_mask.sum() > 0:
        skin_T = T[skin_mask]
        _log(f"[batch]   Skin: [{skin_T.min():.1f}, {skin_T.max():.1f}] "
              f"mean={skin_T.mean():.1f} K")
    if engine_mask.sum() > 0:
        eng_T = T[engine_mask]
        _log(f"[batch]   Engine: [{eng_T.min():.1f}, {eng_T.max():.1f}] "
              f"mean={eng_T.mean():.1f} K")
    if L is not None:
        _log(f"[batch]   L: [{L.min():.1f}, {L.max():.1f}] "
              f"mean={L.mean():.1f} W/(m²·sr)")

    # ── 6. Assign material to original objects ──
    vmin = float(L.min())
    vmax = float(np.percentile(L, config.RENDER_VMAX_PERCENTILE))
    if vmax <= vmin:
        vmax = vmin + 1.0

    visualize.clear_scene_materials()
    visualize.assign_value_material(
        merged, merged.data, L,
        attr_name="Radiance",
        color_mode=config.RENDER_COLOR_MODE,
        vmin=vmin, vmax=vmax,
        mat_name="IR_Radiance",
    )

    merged_centers, _, _ = mesh_graph.get_mesh_data(merged)

    # Delete merged mesh
    merged_mesh_data = merged.data
    bpy.data.objects.remove(merged, do_unlink=True)
    if merged_mesh_data.users == 0:
        bpy.data.meshes.remove(merged_mesh_data)

    # Clean orphan materials
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)

    # Assign to original aircraft
    aircraft_L = L[:ac_n]
    if ac_n != ac_n_orig:
        ac_orig_centers, _, _ = mesh_graph.get_mesh_data(aircraft)
        aircraft_L = _upsample_fallback(
            merged_centers[:ac_n], aircraft_L, ac_orig_centers)

    visualize.assign_value_material(
        aircraft, aircraft.data, aircraft_L,
        attr_name="Radiance",
        color_mode=config.RENDER_COLOR_MODE,
        vmin=vmin, vmax=vmax,
        mat_name="IR_Radiance",
    )
    ir_mat = bpy.data.materials.get("IR_Radiance")

    # Assign to original engines
    slice_start = ac_n
    for eng, fc in zip(all_engines, eng_face_counts):
        if fc == 0:
            continue
        eng_L = L[slice_start:slice_start + fc]
        vert = visualize.face_to_vertex(eng.data, eng_L)
        _write_vertex_attr(eng.data, "Radiance", vert)
        if not eng.data.materials:
            eng.data.materials.append(ir_mat)
        slice_start += fc

    # ── Rename to canonical names ──
    renames = []
    if aircraft.name != "Aircraft":
        renames.append((aircraft, "Aircraft"))
    for side, eng_list in [('L', engines_left), ('R', engines_right)]:
        for idx, eng in enumerate(eng_list):
            target_name = _engine_name(side, idx)
            if eng.name != target_name:
                renames.append((eng, target_name))

    for i, (obj, _target) in enumerate(renames):
        obj.name = f"_tmp_rn_{i}"
    for obj, target in renames:
        if target in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[target], do_unlink=True)
        obj.name = target

    # ── Restore visibility ──
    aircraft.hide_viewport = False
    aircraft.hide_render = False
    for eng in all_engines:
        eng.hide_viewport = False
        eng.hide_render = False

    # ── Save ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)

    _log(f"[batch]   {time.time() - t0:.1f}s → {output_path}")
    return True


def main():
    models = find_models()
    print(f"[batch] scanned {len(models)} models under {CATEGORIES}")

    if MAX_COUNT is not None:
        models = models[:MAX_COUNT]
        print(f"[batch] capped at {MAX_COUNT}, processing {len(models)}")

    output_root = _resolve(OUTPUT_ROOT)
    print(f"[batch] output → {output_root}")

    succeeded = failed = skipped = 0
    t_start = time.time()

    pbar = tqdm(models, desc="[batch]", unit="model",
                ncols=100, disable=not HAS_TQDM)

    for model_id, blend_path in pbar:
        output_path = os.path.join(output_root, model_id, "models", "aircraft.blend")

        if SKIP_EXISTING and os.path.isfile(output_path):
            skipped += 1
            pbar.set_postfix({"ok": succeeded, "fail": failed, "skip": skipped})
            continue

        pbar.set_description(f"[batch] {model_id[:12]}")

        try:
            ok = process_one(model_id, blend_path, output_path)
        except Exception:
            import traceback
            _log(traceback.format_exc())
            ok = False

        if ok:
            succeeded += 1
        else:
            failed += 1

        pbar.set_postfix({"ok": succeeded, "fail": failed, "skip": skipped})
        clear_scene()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"[batch] DONE  ok={succeeded}  fail={failed}  skip={skipped}"
          f"  total={len(models)}  {elapsed:.0f}s")
    print(f"[batch] output → {output_root}")


if __name__ == "__main__":
    main()
