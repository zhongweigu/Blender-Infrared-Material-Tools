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

def _select_only(obj):
    """Select only *obj* (deselect all others via API). No operators."""
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
    """Canonical engine name: 'Engin_L', 'Engin_L_2', 'Engin_R', etc."""
    if idx == 0:
        return f"Engin_{side}"
    return f"Engin_{side}_{idx + 1}"


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

    Returns (T, L, iterations, max_change, engine_mask,
             T_diffusion, T_aero, L_radiance) or all Nones on failure.
    """
    python_exe = _find_venv_python()
    print(f"[DIAG] 项目根目录: {_project_root}")
    print(f"[DIAG] USE_EXTERNAL_COMPUTE={config.USE_EXTERNAL_COMPUTE}, "
          f".venv={'found' if python_exe else 'NOT found'}")
    if python_exe is None:
        print("[外部计算] .venv 未找到，回退到 Blender 内计算")
        return None, None, 0, 0, None, None, None, None

    standalone = os.path.join(_project_root, "new_pipeline",
                              "compute_standalone.py")
    if not os.path.isfile(standalone):
        print(f"[外部计算] 脚本未找到: {standalone}")
        return None, None, 0, 0, None, None, None, None

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
            return (None, None, 0, 0, None,
                    None, None, None)

        # Step C: read results
        res = io_mesh.import_results(output_npz)
        if res is None:
            return (None, None, 0, 0, None,
                    None, None, None)
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


# ══════════════════════════════════════════════════════════════════════════════
# In-Blender compute (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _run_in_blender(merged, exhaust_positions, engine_mask):
    """Compute temperature field entirely inside Blender (pure Python).

    Returns (T, L, iterations, max_change, engine_mask, source_faces,
             T_diffusion, T_aero, L_radiance)
    """
    # ── Build mesh graph ──
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
    print(f"  (网格已通过顶点焊接自然连通，无需结构桥)")

    # 确保所有面片与热源面片在同一连通分量
    mesh_graph.ensure_connectivity(neighbors, edge_lengths, centers, source_faces)

    T = np.full(n_faces, config.T_AIRCRAFT_INIT, dtype=np.float64)
    for fi, T_s in T_source_dict.items():
        T[fi] = T_s

    # 算术平均：所有边权重相等
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
    skin_mask = ~engine_mask
    if skin_mask.sum() > 0:
        skin_T_bl = T[skin_mask]
        print(f"  [DIAG] 蒙皮 T: [{skin_T_bl.min():.1f}, {skin_T_bl.max():.1f}] K "
              f"mean={skin_T_bl.mean():.1f} K")
    else:
        print(f"  [DIAG] 警告: engine_mask 全部为 True ({engine_mask.sum()}/{len(T)})")

    T_diffusion = T.copy()

    # ── Aero heating ──
    delta_T_aero = config.T_AIRCRAFT_INIT * 0.16 * config.MACH_NUMBER ** 2
    T += delta_T_aero
    print(f"\n[Blender] 气动加热: M={config.MACH_NUMBER}, "
          f"ΔT = +{delta_T_aero:.2f} K")
    print(f"  T range: [{T.min():.1f}, {T.max():.1f}] K  mean={T.mean():.1f} K")

    T_aero = T.copy()

    # ── Temperature → Self Radiance ──
    from new_pipeline.calibrate_compute import compute_radiance
    print(f"\n[Blender] 温度→自身辐亮度 (波段 "
          f"{config.LAMBDA_1*1e6:.0f}-{config.LAMBDA_2*1e6:.0f} μm)...")
    L_self = compute_radiance(T, config.EMISSIVITY, config.LAMBDA_1, config.LAMBDA_2)
    print(f"  L_self range: [{L_self.min():.2f}, {L_self.max():.2f}] W/(m²·sr) "
          f"mean={L_self.mean():.2f}")

    # ── Normals (needed for detector directional and optionally env) ──
    _, _, face_verts = mesh_graph.get_mesh_data(merged)
    normals = np.empty((n_faces, 3), dtype=np.float64)
    for i, fv in enumerate(face_verts):
        v0, v1, v2 = np.asarray(fv[0]), np.asarray(fv[1]), np.asarray(fv[2])
        nrm = np.cross(v1 - v0, v2 - v0)
        nlen = np.linalg.norm(nrm)
        if nlen > 1e-9:
            nrm /= nlen
        normals[i] = nrm

    # ── Environment reflection radiation ──
    if config.ENV_RADIATION_ENABLED:
        from new_pipeline.calibrate_compute import compute_environment_radiance
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
    else:
        L = L_self
    print(f"  L_total range: [{L.min():.2f}, {L.max():.2f}] W/(m²·sr) "
          f"mean={L.mean():.2f}")

    L_radiance = L.copy()

    # ── Energy degradation (光学系统能量衰减) ──
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


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 确保每次调用都拿到最新的模块状态 ──────────────────────────────────
    import importlib as _il
    for _mn in ("new_pipeline.config", "new_pipeline.mesh_graph",
                "new_pipeline.heat_source", "new_pipeline.diffusion",
                "new_pipeline.visualize", "new_pipeline.io_mesh",
                "new_pipeline.calibrate_compute"):
        if _mn in sys.modules:
            _il.reload(sys.modules[_mn])

    t_start = time.time()

    # ── 0. 查找对象 ──────────────────────────────────────────────────────
    aircraft = mesh_graph.find_aircraft()
    engines_left, engines_right = mesh_graph.find_all_engines()
    all_engines = engines_left + engines_right

    if aircraft is None:
        print("[错误] 未找到蒙皮网格对象")
        return

    ac_n_orig = len(aircraft.data.polygons)
    ac_n = ac_n_orig
    eng_face_counts = [len(eng.data.polygons) for eng in all_engines]
    total_eng_faces = sum(eng_face_counts)

    exhaust_positions_model = []
    for eng in all_engines:
        pos = mesh_graph.find_exhaust_position(eng)
        if pos is not None:
            exhaust_positions_model.append(np.array(pos))

    print(f"物体: Aircraft='{aircraft.name}' ({ac_n} 面)")
    for side, eng_list in [('L', engines_left), ('R', engines_right)]:
        for idx, eng in enumerate(eng_list):
            i = all_engines.index(eng)
            print(f"       {_engine_name(side, idx)}='{eng.name}' ({eng_face_counts[i]} 面)")
    for i, ep in enumerate(exhaust_positions_model):
        print(f"  尾焰核心 {i+1} (模型坐标): "
              f"({ep[0]:.3f}, {ep[1]:.3f}, {ep[2]:.3f})")

    if not exhaust_positions_model:
        print("[错误] 未找到发动机尾焰位置")
        return
    if config.Q_O is None:
        print("[错误] Q_O 未校准，请先运行 calibrate_qo.py 并将结果填入 config.py")
        return

    # ── 1. 焊接机身接缝顶点（合并前，只焊机身不动发动机） ──────────────
    print("\n[1] 准备网格...")
    ac_dup = _copy_obj(aircraft)
    eng_copies = [_copy_obj(eng) for eng in all_engines]

    if config.MERGE_VERTEX_DIST > 0:
        _select_only(ac_dup)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        nv_before = len(ac_dup.data.vertices)
        bpy.ops.mesh.remove_doubles(threshold=config.MERGE_VERTEX_DIST)
        bpy.ops.object.mode_set(mode='OBJECT')
        nv_after = len(ac_dup.data.vertices)
        ac_n = len(ac_dup.data.polygons)
        if nv_before != nv_after:
            print(f"  机身顶点焊接: {nv_before - nv_after} 顶点 "
                  f"({nv_before} → {nv_after}), 面片={ac_n}")

    aircraft.hide_viewport = True
    aircraft.hide_render = True
    for eng in all_engines:
        eng.hide_viewport = True
        eng.hide_render = True

    # ── 2. 合并所有部件 ──────────────────────────────────────────────────
    _select_only(ac_dup)
    for ec in eng_copies:
        ec.select_set(True)
    bpy.ops.object.join()

    merged = ac_dup
    merged.name = "IR_Unified_Mesh"
    print(f"  合并完成: {len(merged.data.polygons)} 面 "
          f"(ac={ac_n}, eng={total_eng_faces})")

    # 焊接发动机-机身交界处顶点，使 mesh graph 自然连通
    nf_before = len(merged.data.polygons)
    nv_before = len(merged.data.vertices)
    _select_only(merged)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=config.MERGE_VERTEX_DIST)
    bpy.ops.object.mode_set(mode='OBJECT')
    nv_after = len(merged.data.vertices)
    nf_after = len(merged.data.polygons)
    if nv_before != nv_after:
        print(f"  交界面焊接: {nv_before - nv_after} 顶点 "
              f"({nv_before} → {nv_after})")
    if nf_before != nf_after:
        print(f"  {nf_before - nf_after} 个退化面片被移除")

    # ── 3. 缩放到真实尺寸 ───────────────────────────────────────────────
    if config.MODEL_SCALE != 1.0:
        print(f"\n[2] 缩放到真实尺寸: ×{config.MODEL_SCALE}")
        _select_only(merged)
        merged.scale *= config.MODEL_SCALE
        bpy.ops.object.transform_apply(scale=True)
        print("  缩放完成 (合并后统一缩放，相对位置正确)")

    exhaust_positions = [ep * config.MODEL_SCALE
                         for ep in exhaust_positions_model]
    for i, ep in enumerate(exhaust_positions):
        print(f"  尾焰核心 {i+1} (真实坐标): "
              f"({ep[0]:.3f}, {ep[1]:.3f}, {ep[2]:.3f})")

    # ── 对称化网格 ──────────────────────────────────────────────────────
    if config.SYMMETRIZE_MESH:
        print(f"\n[对称化] 沿X=0强制对称化网格...")
        mesh_graph.symmetrize_mesh(merged)

    # Build engine_mask (engine faces are at the end of the merged face list)
    n_faces = len(merged.data.polygons)
    total_eng = sum(eng_face_counts)
    ac_n = n_faces - total_eng  # recalculate after possible face removal
    engine_mask = np.zeros(n_faces, dtype=bool)
    if total_eng > 0:
        engine_mask[ac_n:] = True
    print(f"  engine_mask: {engine_mask.sum()} engine / {(~engine_mask).sum()} body (总计 {n_faces})")

    # ── 3.5. Uniform减面 ──────────────────────────────────────────────
    compute_mesh = merged
    compute_engine_mask = engine_mask
    decimated = None
    is_decimated = False

    if config.DECIMATE_RATIO < 1.0:
        print(f"\n[3.5] Uniform减面 ratio={config.DECIMATE_RATIO}...")
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
        print(f"  减面后: {len(dec_centers)} 面 (原 {n_faces} 面)")

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
        print(f"  发动机面片映射: {compute_engine_mask.sum()}")

        compute_mesh = decimated

    # ── 4. Compute (external or in-Blender) ─────────────────────────────
    source_faces = set()
    L = None
    T_diffusion = T_aero = L_radiance = None

    if config.USE_EXTERNAL_COMPUTE:
        (T, L, iterations, max_change, _,
         T_diffusion, T_aero, L_radiance) = _run_external_compute(
            compute_mesh, exhaust_positions, compute_engine_mask)
        if T is not None:
            print("\n[外部计算] 成功")
            n_eng = int(compute_engine_mask.sum())
            n_src = max(5, min(200, int(n_eng * 0.05)))
            eng_idx = np.where(compute_engine_mask)[0]
            cc, _, _ = mesh_graph.get_mesh_data(compute_mesh)
            for ep in exhaust_positions:
                dists = np.linalg.norm(cc[eng_idx] - ep, axis=1)
                nearest = eng_idx[np.argsort(dists)[:n_src]]
                source_faces.update(nearest.tolist())
        else:
            print("\n[外部计算] 失败，回退到 Blender 内计算")
            (T, L, iterations, max_change, _, source_faces,
             T_diffusion, T_aero, L_radiance) = \
                _run_in_blender(compute_mesh, exhaust_positions, compute_engine_mask)
    else:
        (T, L, iterations, max_change, _, source_faces,
         T_diffusion, T_aero, L_radiance) = \
            _run_in_blender(compute_mesh, exhaust_positions, compute_engine_mask)

    # ── Upsample back to original mesh ─────────────────────────────────
    if is_decimated:
        print(f"\n[upsample] 温度映射回原始网格 ({len(T)} → {n_faces} 面)...")
        merged_centers, _, _ = mesh_graph.get_mesh_data(merged)

        def _upsample_fb(src_c, src_t, dst_c):
            n_dst = len(dst_c)
            out = np.empty(n_dst, dtype=np.float64)
            for b0 in range(0, n_dst, 1000):
                b1 = min(b0 + 1000, n_dst)
                d2 = np.sum((dst_c[b0:b1, None, :] - src_c[None, :, :]) ** 2, axis=2)
                out[b0:b1] = np.asarray(src_t, dtype=np.float64)[np.argmin(d2, axis=1)]
            return out

        T = _upsample_fb(dec_centers, T, merged_centers)
        L = _upsample_fb(dec_centers, L, merged_centers)
        if T_diffusion is not None:
            T_diffusion = _upsample_fb(dec_centers, T_diffusion, merged_centers)
        if T_aero is not None:
            T_aero = _upsample_fb(dec_centers, T_aero, merged_centers)
        if L_radiance is not None:
            L_radiance = _upsample_fb(dec_centers, L_radiance, merged_centers)
        mesh_graph.cleanup_decimated(decimated, is_decimated)

    # ── 5. 统计输出 ────────────────────────────────────────────────────
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
        if skin_T.max() - skin_T.min() < 0.01:
            print(f"  [DIAG] 警告: 蒙皮温度无扩散 (max-min={skin_T.max()-skin_T.min():.4f} K)")
    if engine_mask.sum() > 0:
        eng_T = T[engine_mask]
        print(f"  {'发动机':12s} | {eng_T.min():10.2f}  {eng_T.max():10.2f}  {eng_T.mean():10.2f} K")

    if source_faces:
        source_T = np.array([T[fi] for fi in source_faces])
        print(f"  {'热源面片':12s} | {source_T.min():10.2f}  {source_T.max():10.2f}  {source_T.mean():10.2f} K")

    # 左右发动机分别统计
    n_left_eng = len(engines_left)
    n_right_eng = len(engines_right)
    if n_left_eng > 0 and n_right_eng > 0 and L is not None:
        off = ac_n
        left_mask = np.zeros(n_faces, dtype=bool)
        for fc in eng_face_counts[:n_left_eng]:
            left_mask[off:off + fc] = True
            off += fc
        right_mask = np.zeros(n_faces, dtype=bool)
        for fc in eng_face_counts[n_left_eng:]:
            right_mask[off:off + fc] = True
            off += fc
        if left_mask.sum() > 0 and right_mask.sum() > 0:
            print(f"  {'左发 L':12s} | {L[left_mask].min():10.2f}  {L[left_mask].max():10.2f}  {L[left_mask].mean():10.2f} W/(m²·sr)")
            print(f"  {'右发 L':12s} | {L[right_mask].min():10.2f}  {L[right_mask].max():10.2f}  {L[right_mask].mean():10.2f} W/(m²·sr)")
            print(f"  {'左发 T':12s} | {T[left_mask].min():10.2f}  {T[left_mask].max():10.2f}  {T[left_mask].mean():10.2f} K")
            print(f"  {'右发 T':12s} | {T[right_mask].min():10.2f}  {T[right_mask].max():10.2f}  {T[right_mask].mean():10.2f} K")
    print(f"\n{'='*60}")
    t_compute = time.time()
    print(f"计算耗时: {t_compute - t_start:.1f} s")

    # ── 5.5 过程图片输出 ─────────────────────────────────────────────────
    if config.PROCESS_IMAGES_ENABLED:
        print(f"\n[过程图片] 输出管线各阶段结果...")
        img_dir = bpy.path.abspath(config.PROCESS_IMAGES_DIR)
        os.makedirs(img_dir, exist_ok=True)
        # 正面偏左下方向上看飞机
        cam_loc = (-35, -70, -8)
        target = (0, 0, 4)

        def _render_step(face_vals, filename, label, color_mode="thermal"):
            print(f"  {label} → {filename}")
            visualize.clear_scene_materials()
            vmi = float(face_vals.min())
            vma = float(face_vals.max())
            if vma <= vmi:
                vma = vmi + 1.0
            visualize.assign_value_material(
                merged, merged.data, face_vals,
                attr_name="Radiance",
                color_mode=color_mode,
                vmin=vmi, vmax=vma,
                mat_name="IR_Process",
            )
            visualize.setup_camera(cam_loc, target=target)
            visualize.render_to_file(os.path.join(img_dir, filename))

        # 温度阶段用热成像彩色，辐射阶段用黑白
        _render_step(T_diffusion, "01_temperature_diffusion.png", "扩散后温度", "thermal")
        _render_step(T_aero, "02_temperature_aero.png", "气动加热后温度", "thermal")
        _render_step(L_radiance, "03_radiance.png", "Planck辐亮度", "bw")
        _render_step(L, "04_radiance_final.png", "最终辐亮度(能量衰减后)", "bw")

    # ── 6. 材质 + 渲染 ────────────────────────────────────────────────
    vmin = float(L.min())
    vmax = float(np.percentile(L, config.RENDER_VMAX_PERCENTILE))
    if vmax <= vmin:
        vmax = vmin + 1.0

    print(f"\n[可视化] 辐亮度范围: {vmin:.2f} ~ {vmax:.2f} W/(m²·sr)")
    visualize.clear_scene_materials()
    visualize.assign_value_material(
        merged, merged.data, L,
        attr_name="Radiance",
        color_mode=config.RENDER_COLOR_MODE,
        vmin=vmin, vmax=vmax,
        mat_name="IR_Radiance",
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
    merged_centers, _, _ = mesh_graph.get_mesh_data(merged)

    merged_mesh = merged.data
    bpy.data.objects.remove(merged, do_unlink=True)
    if merged_mesh.users == 0:
        bpy.data.meshes.remove(merged_mesh)

    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)

    # 将 IR 材质应用到原始机身（面片布局: [aircraft | eng_0 | eng_1 | ...]）
    aircraft_L = L[:ac_n]
    if ac_n != ac_n_orig:
        ac_orig_centers, _, _ = mesh_graph.get_mesh_data(aircraft)
        aircraft_L = mesh_graph.upsample_temperatures_fallback(
            merged_centers[:ac_n], aircraft_L, ac_orig_centers)
    visualize.assign_value_material(
        aircraft, aircraft.data, aircraft_L,
        attr_name="Radiance",
        color_mode=config.RENDER_COLOR_MODE,
        vmin=vmin, vmax=vmax,
        mat_name="IR_Radiance",
    )
    ir_mat = bpy.data.materials.get("IR_Radiance")

    # 发动机
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

    # 重命名
    renames = []
    if aircraft.name != "Aircraft":
        renames.append((aircraft, "Aircraft"))
    for side, eng_list in [('L', engines_left), ('R', engines_right)]:
        for idx, eng in enumerate(eng_list):
            target_name = _engine_name(side, idx)
            if eng.name != target_name:
                renames.append((eng, target_name))

    for i, (obj, _target) in enumerate(renames):
        obj.name = f"_tmp_rename_{i}"
    for obj, target in renames:
        if target in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[target], do_unlink=True)
        old_name = obj.name
        obj.name = target
        print(f"  重命名: {old_name} → {target}")

    # 恢复可见性
    aircraft.hide_viewport = False
    aircraft.hide_render = False
    for eng in all_engines:
        eng.hide_viewport = False
        eng.hide_render = False

    # ── 保存处理后的 .blend ──
    _batch_output = os.environ.get('BLIR_OUTPUT_PATH', '')
    if _batch_output:
        os.makedirs(os.path.dirname(_batch_output), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=_batch_output)
        print(f"已保存: {_batch_output}")
    elif config.SAVE_PROCESSED_BLEND and bpy.data.filepath:
        dir_path = os.path.dirname(bpy.path.abspath("//"))
        base_name = os.path.splitext(bpy.path.basename(bpy.data.filepath))[0]
        save_path = os.path.join(dir_path, f"{base_name}{config.PROCESSED_BLEND_SUFFIX}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=save_path)
        print(f"已保存: {save_path}")

    print("完成")


if __name__ == "__main__":
    main()
