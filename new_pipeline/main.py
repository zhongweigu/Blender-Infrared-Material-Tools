"""
新管线 —— 统一网格稳态温度计算
基于 pipeline.md 的热平衡 + Gauss-Seidel 扩散

用法:
    blender -b <模型文件.blend> --python ./new_pipeline/main.py
    blender -b --python ./new_pipeline/main.py  (场景中已加载模型)

前置: 需先运行 calibrate_qo.py 校准得到 Q_O，填入 config.py

外部加速: config.USE_EXTERNAL_COMPUTE = True 时，
合并+缩放后导出 .npz → .venv Python (numba JIT) 计算 → 读回渲染。
"""

import os
import sys
import time
import subprocess
import tempfile

import bpy
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Path setup
# ══════════════════════════════════════════════════════════════════════════════

_project_root = None

try:
    _sdir = os.path.dirname(os.path.abspath(__file__))
    _candidate = os.path.dirname(_sdir)
    if os.path.isdir(os.path.join(_candidate, "new_pipeline")):
        _project_root = _candidate
except NameError:
    pass

if _project_root is None:
    for _text in bpy.data.texts:
        if _text.filepath:
            _sdir = os.path.dirname(os.path.abspath(bpy.path.abspath(_text.filepath)))
            _candidate = os.path.dirname(_sdir)
            if os.path.isdir(os.path.join(_candidate, "new_pipeline")):
                _project_root = _candidate
                break

if _project_root is None and bpy.data.filepath:
    _d = os.path.dirname(bpy.path.abspath("//"))
    for _ in range(5):
        if os.path.isdir(os.path.join(_d, "new_pipeline")):
            _project_root = _d
            break
        _parent = os.path.dirname(_d)
        if _parent == _d:
            break
        _d = _parent

if _project_root is None:
    _config_candidates = []
    for _text in bpy.data.texts:
        if _text.filepath:
            _sdir = os.path.dirname(os.path.abspath(bpy.path.abspath(_text.filepath)))
            _cp = os.path.join(_sdir, "config.py")
            if os.path.isfile(_cp):
                _config_candidates.append(_cp)
    if bpy.data.filepath:
        _d = os.path.dirname(bpy.path.abspath("//"))
        for _ in range(5):
            _cp = os.path.join(_d, "new_pipeline", "config.py")
            if os.path.isfile(_cp) and _cp not in _config_candidates:
                _config_candidates.append(_cp)
            _parent = os.path.dirname(_d)
            if _parent == _d:
                break
            _d = _parent
    for _cp in _config_candidates:
        with open(_cp, "r", encoding="utf-8") as _f:
            for _line in _f:
                if _line.startswith("PROJECT_ROOT"):
                    _val = _line.split("=", 1)[1].strip().strip('"').strip("'")
                    if _val and os.path.isdir(_val):
                        _project_root = _val
                    break
        if _project_root:
            break

if _project_root is None:
    raise RuntimeError(
        "无法定位项目根目录。请:\n"
        "  1) 在 config.py 中设置 PROJECT_ROOT 为项目根目录绝对路径，或\n"
        "  2) 确保 .blend 保存在与 new_pipeline/ 同级的目录下"
    )

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import importlib
for _mod_name in ("new_pipeline.config", "new_pipeline.mesh_graph",
                   "new_pipeline.heat_source", "new_pipeline.diffusion",
                   "new_pipeline.visualize", "new_pipeline.io_mesh"):
    if _mod_name in sys.modules:
        importlib.reload(sys.modules[_mod_name])

from new_pipeline import config
from new_pipeline import mesh_graph
from new_pipeline import heat_source
from new_pipeline import diffusion
from new_pipeline import visualize
from new_pipeline import io_mesh


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _copy_obj(obj):
    """Duplicate a mesh object. Returns the duplicate."""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.duplicate()
    dup = bpy.context.active_object
    obj.select_set(False)
    return dup


def _find_venv_python():
    """Locate the .venv Python interpreter."""
    venv_dir = os.path.join(_project_root, ".venv")
    if not os.path.isdir(venv_dir):
        return None
    for sub in ("Scripts", "bin"):
        for name in ("python.exe", "python", "python3"):
            p = os.path.join(venv_dir, sub, name)
            if os.path.isfile(p):
                return p
    return None


# ══════════════════════════════════════════════════════════════════════════════
# External compute
# ══════════════════════════════════════════════════════════════════════════════

def _run_external_compute(merged, exhaust_positions, engine_mask):
    """Export unified mesh → subprocess compute_standalone.py → import T.

    Returns (T, L, iterations, max_change, engine_mask) or
    (None, None, 0, 0, None) on failure.
    """
    python_exe = _find_venv_python()
    if python_exe is None:
        print("[外部计算] .venv 未找到，回退到 Blender 内计算")
        return None, None, 0, 0, None

    standalone = os.path.join(_project_root, "new_pipeline",
                              "compute_standalone.py")
    if not os.path.isfile(standalone):
        print(f"[外部计算] 脚本未找到: {standalone}")
        return None, None, 0, 0, None

    tmpdir = tempfile.gettempdir()
    input_npz = os.path.join(tmpdir, "_blir_main_input.npz")
    output_npz = os.path.join(tmpdir, "_blir_main_output.npz")

    try:
        # Step A: export
        print("\n[外部计算] 导出统一网格...")
        t_export = time.time()
        io_mesh.export_unified_mesh(input_npz, merged,
                                     exhaust_positions, engine_mask)
        print(f"  导出耗时: {time.time() - t_export:.1f} s")

        # Step B: run standalone
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
            return None, None, 0, 0, None

        # Step C: read results
        res = io_mesh.import_results(output_npz)
        if res is None:
            return None, None, 0, 0, None
        return (res['T'], res['L'], res['iterations'], res['max_change'],
                engine_mask)

    except FileNotFoundError:
        print(f"[外部计算] Python 未找到: {python_exe}")
        return None, None, 0, 0, None
    except subprocess.TimeoutExpired:
        print("[外部计算] 进程超时 (600s)")
        return None, None, 0, 0, None
    except Exception as e:
        print(f"[外部计算] 异常: {e}")
        return None, None, 0, 0, None
    finally:
        for f in (input_npz, output_npz):
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except OSError:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# In-Blender compute (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _run_in_blender(merged, exhaust_positions, ac_n, el_n, er_n):
    """Compute temperature field entirely inside Blender (pure Python)."""
    # ── Build mesh graph ──
    print("\n[Blender] 构建面片邻接图...")
    centers, areas, _ = mesh_graph.get_mesh_data(merged)
    neighbors, edge_lengths = mesh_graph.build_face_adjacency(merged)

    n_faces = len(centers)
    print(f"  面片总数: {n_faces}")

    # Engine mask
    engine_mask = np.zeros(n_faces, dtype=bool)
    if el_n > 0:
        engine_mask[ac_n:ac_n + el_n] = True
    if er_n > 0:
        engine_mask[ac_n + el_n:ac_n + el_n + er_n] = True
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

    # 添加发动机↔蒙皮结构连接桥（merge 不会创建跨对象边）
    cross_pairs = mesh_graph.find_cross_boundary_pairs(
        centers, engine_mask, max_pairs=3, max_distance=2.0)
    structural_edges = set()
    for ei, si, dist in cross_pairs:
        neighbors[ei].append(si)
        neighbors[si].append(ei)
        edge_lengths[(ei, si)] = -dist  # 负值标记为结构连接
        edge_lengths[(si, ei)] = -dist
        structural_edges.add((ei, si))
        structural_edges.add((si, ei))
    print(f"  跨边界结构桥: {len(cross_pairs)} 对 "
          f"({len(structural_edges)//2} 条边)")

    T = np.full(n_faces, config.T_AIRCRAFT_INIT, dtype=np.float64)
    for fi, T_s in T_source_dict.items():
        T[fi] = T_s

    # 算术平均：所有权重相等
    conductances = {}
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if (i, j) not in conductances:
                conductances[(i, j)] = 1.0
                conductances[(j, i)] = 1.0

    T, iterations, max_change = diffusion.gauss_seidel(
        T, neighbors, conductances,
        fixed_faces=source_faces,
        tol=config.DIFFUSION_TOL,
        max_iter=config.MAX_ITERATIONS,
    )

    print(f"  完成: {iterations} 次迭代, 最终 max ΔT = {max_change:.6f} K")
    print(f"  扩散后 T range: [{T.min():.1f}, {T.max():.1f}] K")

    # ── Aero heating ──
    delta_T_aero = config.T_AIRCRAFT_INIT * 0.16 * config.MACH_NUMBER ** 2
    T += delta_T_aero
    print(f"\n[Blender] 气动加热: M={config.MACH_NUMBER}, "
          f"ΔT = +{delta_T_aero:.2f} K")
    print(f"  T range: [{T.min():.1f}, {T.max():.1f}] K  mean={T.mean():.1f} K")

    # ── Temperature → Self Radiance ──
    from new_pipeline.calibrate_compute import compute_radiance
    print(f"\n[Blender] 温度→自身辐亮度 (波段 "
          f"{config.LAMBDA_1*1e6:.0f}-{config.LAMBDA_2*1e6:.0f} μm)...")
    L_self = compute_radiance(T, config.EMISSIVITY, config.LAMBDA_1, config.LAMBDA_2)
    print(f"  L_self range: [{L_self.min():.2f}, {L_self.max():.2f}] W/(m²·sr) "
          f"mean={L_self.mean():.2f}")

    # ── Environment reflection radiation ──
    from new_pipeline.calibrate_compute import compute_environment_radiance
    _, _, face_verts = mesh_graph.get_mesh_data(merged)
    normals = np.empty((n_faces, 3), dtype=np.float64)
    for i, fv in enumerate(face_verts):
        v0, v1, v2 = np.asarray(fv[0]), np.asarray(fv[1]), np.asarray(fv[2])
        nrm = np.cross(v1 - v0, v2 - v0)
        nlen = np.linalg.norm(nrm)
        if nlen > 1e-9:
            nrm /= nlen
        normals[i] = nrm

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
    L_refl = compute_environment_radiance(centers, normals, config.EMISSIVITY, env_config)
    print(f"  L_refl range: [{L_refl.min():.2f}, {L_refl.max():.2f}] W/(m²·sr) "
          f"mean={L_refl.mean():.2f}")

    L = L_self + L_refl
    print(f"  L_total range: [{L.min():.2f}, {L.max():.2f}] W/(m²·sr) "
          f"mean={L.mean():.2f}")

    # ── Detector directional radiation ──
    from new_pipeline.calibrate_compute import compute_detector_directional
    det_pos = np.array(config.DETECTOR_POS, dtype=np.float64)
    det_los = (np.array(config.DETECTOR_LOS, dtype=np.float64)
               if config.DETECTOR_LOS is not None else None)
    print(f"\n[Blender] 探测器方向辐射 (17)...")
    L = compute_detector_directional(L, centers, normals, det_pos, det_los)
    print(f"  L_cam range: [{L.min():.2f}, {L.max():.2f}] W/(m²·sr) "
          f"mean={L.mean():.2f}")

    # ── Atmospheric attenuation ──
    from new_pipeline.calibrate_compute import apply_atmospheric_attenuation
    detector_pos = np.array(config.DETECTOR_POS, dtype=np.float64)
    print(f"\n[Blender] 大气衰减: μ={config.MU_ATM:.2e} m⁻¹, "
          f"探测器=({detector_pos[0]:.0f}, {detector_pos[1]:.0f}, {detector_pos[2]:.0f})")
    dists = np.linalg.norm(centers - detector_pos, axis=1)
    tau_vals = np.exp(-config.MU_ATM * dists)
    print(f"  距离范围: [{dists.min():.0f}, {dists.max():.0f}] m, "
          f"τ range: [{tau_vals.min():.3f}, {tau_vals.max():.3f}]")
    L = apply_atmospheric_attenuation(L, centers, detector_pos, config.MU_ATM)
    print(f"  L_detected range: [{L.min():.2f}, {L.max():.2f}] W/(m²·sr)")

    return T, L, iterations, max_change, engine_mask, source_faces


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # ── 0. 查找对象 ──────────────────────────────────────────────────────
    aircraft = mesh_graph.find_aircraft()
    engine_l = mesh_graph.find_engine_left()
    engine_r = mesh_graph.find_engine_right()

    if aircraft is None:
        print("[错误] 未找到蒙皮网格对象")
        return

    ac_n = len(aircraft.data.polygons)
    el_n = len(engine_l.data.polygons) if engine_l else 0
    er_n = len(engine_r.data.polygons) if engine_r else 0

    exhaust_positions_model = []
    for eng in (engine_l, engine_r):
        if eng:
            pos = mesh_graph.find_exhaust_position(eng)
            if pos is not None:
                exhaust_positions_model.append(np.array(pos))

    print(f"物体: Aircraft='{aircraft.name}' ({ac_n} 面)")
    if engine_l:
        print(f"       Engin_L='{engine_l.name}' ({el_n} 面)")
    if engine_r:
        print(f"       Engin_R='{engine_r.name}' ({er_n} 面)")
    for i, ep in enumerate(exhaust_positions_model):
        print(f"  尾焰核心 {i+1} (模型坐标): "
              f"({ep[0]:.3f}, {ep[1]:.3f}, {ep[2]:.3f})")

    if not exhaust_positions_model:
        print("[错误] 未找到发动机尾焰位置")
        return
    if config.Q_O is None:
        print("[错误] Q_O 未校准，请先运行 calibrate_qo.py 并将结果填入 config.py")
        return

    # ── 1. 合并所有部件 ──────────────────────────────────────────────────
    print("\n[1] 合并所有部件为统一网格...")
    ac_dup = _copy_obj(aircraft)
    el_dup = _copy_obj(engine_l) if engine_l else None
    er_dup = _copy_obj(engine_r) if engine_r else None

    aircraft.hide_viewport = True
    aircraft.hide_render = True
    if engine_l:
        engine_l.hide_viewport = True
        engine_l.hide_render = True
    if engine_r:
        engine_r.hide_viewport = True
        engine_r.hide_render = True

    bpy.ops.object.select_all(action='DESELECT')
    ac_dup.select_set(True)
    bpy.context.view_layer.objects.active = ac_dup
    for d in (el_dup, er_dup):
        if d:
            d.select_set(True)
    bpy.ops.object.join()

    merged = ac_dup
    merged.name = "IR_Unified_Mesh"
    print(f"  合并完成: {len(merged.data.polygons)} 面 "
          f"(ac={ac_n}, el={el_n}, er={er_n})")

    # ── 2. 缩放到真实尺寸 ───────────────────────────────────────────────
    if config.MODEL_SCALE != 1.0:
        print(f"\n[2] 缩放到真实尺寸: ×{config.MODEL_SCALE}")
        bpy.ops.object.select_all(action='DESELECT')
        merged.select_set(True)
        bpy.context.view_layer.objects.active = merged
        merged.scale *= config.MODEL_SCALE
        bpy.ops.object.transform_apply(scale=True)
        print("  缩放完成 (合并后统一缩放，相对位置正确)")

    exhaust_positions = [ep * config.MODEL_SCALE
                         for ep in exhaust_positions_model]
    for i, ep in enumerate(exhaust_positions):
        print(f"  尾焰核心 {i+1} (真实坐标): "
              f"({ep[0]:.3f}, {ep[1]:.3f}, {ep[2]:.3f})")

    # Build engine_mask (needed by both paths and for stats/render)
    n_faces = len(merged.data.polygons)
    engine_mask = np.zeros(n_faces, dtype=bool)
    if el_n > 0:
        engine_mask[ac_n:ac_n + el_n] = True
    if er_n > 0:
        engine_mask[ac_n + el_n:ac_n + el_n + er_n] = True

    # ── 3. Compute (external or in-Blender) ─────────────────────────────
    source_faces = set()
    L = None

    if config.USE_EXTERNAL_COMPUTE:
        T, L, iterations, max_change, _ = _run_external_compute(
            merged, exhaust_positions, engine_mask)
        if T is not None:
            print("\n[外部计算] 成功")
            # Recompute source_faces for stats (same logic as external)
            n_eng = int(engine_mask.sum())
            n_src = max(5, min(200, int(n_eng * 0.05)))
            eng_idx = np.where(engine_mask)[0]
            centers, _, _ = mesh_graph.get_mesh_data(merged)
            for ep in exhaust_positions:
                dists = np.linalg.norm(centers[eng_idx] - ep, axis=1)
                nearest = eng_idx[np.argsort(dists)[:n_src]]
                source_faces.update(nearest.tolist())
        else:
            print("\n[外部计算] 失败，回退到 Blender 内计算")
            T, L, iterations, max_change, engine_mask, source_faces = \
                _run_in_blender(merged, exhaust_positions, ac_n, el_n, er_n)
    else:
        T, L, iterations, max_change, engine_mask, source_faces = \
            _run_in_blender(merged, exhaust_positions, ac_n, el_n, er_n)

    # ── 4. 统计输出 ────────────────────────────────────────────────────
    delta_T_aero = config.T_AIRCRAFT_INIT * 0.16 * config.MACH_NUMBER ** 2
    print(f"\n{'='*60}")
    print(f"  稳态温度计算结果")
    print(f"{'='*60}")
    print(f"  统一网格面片总数: {n_faces}")
    print(f"  热源面片: {len(source_faces)}")
    print(f"  扩散迭代: {iterations}, 最终 max ΔT: {max_change:.6f} K")
    print(f"  气动加热: M={config.MACH_NUMBER}, ΔT={delta_T_aero:.2f} K")
    print()

    print(f"  {'区域':12s} | {'最低温':>10s}  {'最高温':>10s}  {'平均温':>10s}")
    print(f"  {'-'*12}-+-{'-'*10}--{'-'*10}--{'-'*10}")
    print(f"  {'整体':12s} | {T.min():10.2f}  {T.max():10.2f}  {T.mean():10.2f} K")

    skin_mask = ~engine_mask
    if skin_mask.sum() > 0:
        skin_T = T[skin_mask]
        print(f"  {'蒙皮':12s} | {skin_T.min():10.2f}  {skin_T.max():10.2f}  {skin_T.mean():10.2f} K")
    if engine_mask.sum() > 0:
        eng_T = T[engine_mask]
        print(f"  {'发动机':12s} | {eng_T.min():10.2f}  {eng_T.max():10.2f}  {eng_T.mean():10.2f} K")

    if source_faces:
        source_T = np.array([T[fi] for fi in source_faces])
        print(f"  {'热源面片':12s} | {source_T.min():10.2f}  {source_T.max():10.2f}  {source_T.mean():10.2f} K")

    print(f"\n{'='*60}")
    t_compute = time.time()
    print(f"计算耗时: {t_compute - t_start:.1f} s")

    # ── 5. 材质 + 渲染 (基于辐亮度 L) ────────────────────────────────────
    vmin = float(L.min())
    vmax = float(L.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    # 始终给合并网格上材质（GUI 下可看到结果）
    print(f"\n[可视化] 辐亮度范围: {vmin:.2f} ~ {vmax:.2f} W/(m²·sr)")
    visualize.clear_scene_materials()
    visualize.assign_value_material(
        merged, merged.data, L,
        attr_name="Radiance",
        color_mode=config.RENDER_COLOR_MODE,
        vmin=vmin, vmax=vmax,
        mat_name="IR_Radiance"
    )

    if config.RENDER_ENABLED:
        print("[可视化] 多视角渲染...")
        output_dir = bpy.path.abspath(config.RENDER_OUTPUT_DIR)
        visualize.render_multiview(
            merged, merged.data, L,
            output_dir=output_dir,
            base_name="radiance",
            color_mode=config.RENDER_COLOR_MODE,
            vmin=vmin, vmax=vmax,
        )

    # ── 清理 ────────────────────────────────────────────────────────────
    if bpy.app.background:
        # 后台模式：删除合并网格，恢复原始对象
        bpy.data.objects.remove(merged, do_unlink=True)
        aircraft.hide_viewport = False
        aircraft.hide_render = False
        if engine_l:
            engine_l.hide_viewport = False
            engine_l.hide_render = False
        if engine_r:
            engine_r.hide_viewport = False
            engine_r.hide_render = False
    else:
        # GUI 模式：保留合并网格供查看；原始对象保持隐藏
        print("[可视化] GUI 模式 — 合并网格已保留在场景中，原始对象已隐藏")

    print("完成")


if __name__ == "__main__":
    main()
