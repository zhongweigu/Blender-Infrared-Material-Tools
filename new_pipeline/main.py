"""
新管线 —— 统一网格稳态温度计算
基于 pipeline.md 的热平衡 + Gauss-Seidel 扩散

用法:
    blender -b <模型文件.blend> --python ./new_pipeline/main.py
    blender -b --python ./new_pipeline/main.py  (场景中已加载模型)

流程:
    1. 查找部件 (Aircraft, Engin_L, Engin_R)
    2. 合并网格 + 缩放 + 焊接接缝 + [可选减面]
    3. 构建邻接图 + 跨边界结构连接
    4. 热源面片识别 + 热平衡求解 T_s
    5. Gauss-Seidel 扩散
    6. 氨动加热
    7. 温度→辐亮度 (Planck)
    8. 能量衰减
    9. [可选上采样回原始网格]
    10. 材质应用 + 渲染 + 保存

外部计算: config.USE_EXTERNAL_COMPUTE = True 时，
导出 .npz → 子进程 compute_standalone.py (numba JIT) → 读回结果
"""

import os
import sys
import time
import subprocess
import tempfile

import bpy
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 项目路径设置
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
                   "new_pipeline.visualize", "new_pipeline.calibrate_compute"):
    if _mod_name in sys.modules:
        importlib.reload(sys.modules[_mod_name])

from new_pipeline import config
from new_pipeline import mesh_graph
from new_pipeline import heat_source
from new_pipeline import diffusion
from new_pipeline import visualize
from new_pipeline import calibrate_compute


# ══════════════════════════════════════════════════════════════════════════════
# Blender 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _select_only(obj):
    """Select only *obj*."""
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _copy_obj(obj):
    """Duplicate a mesh object."""
    mesh = obj.data.copy()
    dup = bpy.data.objects.new(obj.name + "_dup", mesh)
    dup.matrix_world = obj.matrix_world.copy()
    bpy.context.collection.objects.link(dup)
    return dup


def _write_vertex_attr(mesh, attr_name, vert_values):
    """Write a per-vertex float attribute."""
    if attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name])
    attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
    attr.data.foreach_set('value', vert_values.tolist())


def _engine_name(side, idx):
    """Canonical engine name."""
    return f"Engin_{side}" if idx == 0 else f"Engin_{side}_{idx + 1}"


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
# 上采样：减面网格 → 原始网格
# ══════════════════════════════════════════════════════════════════════════════

def upsample_to_original(src_centers, src_values, dst_centers):
    """
    最近邻上采样：从减面网格映射到原始网格。

    Args:
        src_centers: (M, 3) 减面网格面片中心
        src_values: (M,) 减面网格面片值
        dst_centers: (N, 3) 原始网格面片中心

    Returns:
        dst_values: (N,) 原始网格面片值
    """
    n_dst = len(dst_centers)
    dst_values = np.empty(n_dst, dtype=np.float64)

    # 分批处理避免内存溢出
    batch_size = 1000
    for b0 in range(0, n_dst, batch_size):
        b1 = min(b0 + batch_size, n_dst)
        # 计算距离矩阵
        d2 = np.sum((dst_centers[b0:b1, None, :] - src_centers[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(d2, axis=1)
        dst_values[b0:b1] = src_values[nearest]

    return dst_values


# ══════════════════════════════════════════════════════════════════════════════
# 步骤1: 网格准备（含减面）
# ══════════════════════════════════════════════════════════════════════════════

def prepare_mesh(aircraft, all_engines, exhaust_positions_model):
    """
    合并所有部件、焊接接缝、缩放、减面。

    Returns:
        merged: 合并后的 mesh object（用于计算）
        merged_full: 合并后未减面的 mesh object（用于上采样，仅当减面时）
        engine_mask: (N,) bool, True = 发动机面片（减面后）
        engine_mask_full: (N_full,) bool（仅当减面时）
        exhaust_positions: list of np.ndarray, 真实尺度尾焰坐标
        ac_n: 机身面片数（减面后）
        ac_n_full: 机身面片数（未减面，仅当减面时）
        eng_face_counts: 各发动机面片数列表（减面后）
        eng_face_counts_full: 各发动机面片数列表（未减面）
        is_decimated: bool, 是否进行了减面
    """
    # 复制原始对象
    ac_dup = _copy_obj(aircraft)
    eng_copies = [_copy_obj(eng) for eng in all_engines]

    eng_face_counts_full = [len(eng.data.polygons) for eng in all_engines]
    ac_n_full = len(ac_dup.data.polygons)

    # 焊接机身接缝顶点
    if config.MERGE_VERTEX_DIST > 0:
        _select_only(ac_dup)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        nv_before = len(ac_dup.data.vertices)
        bpy.ops.mesh.remove_doubles(threshold=config.MERGE_VERTEX_DIST)
        bpy.ops.object.mode_set(mode='OBJECT')
        ac_n_full = len(ac_dup.data.polygons)
        if nv_before != len(ac_dup.data.vertices):
            print(f"  机身焊接: {nv_before - len(ac_dup.data.vertices)} 顶点")

    # 隐藏原始对象
    aircraft.hide_viewport = True
    aircraft.hide_render = True
    for eng in all_engines:
        eng.hide_viewport = True
        eng.hide_render = True

    # 合并所有部件
    _select_only(ac_dup)
    for ec in eng_copies:
        ec.select_set(True)
    bpy.ops.object.join()

    merged_full = ac_dup
    merged_full.name = "IR_Unified_Mesh_Full"

    # 再次焊接发动机-机身交界处
    _select_only(merged_full)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=config.MERGE_VERTEX_DIST)
    bpy.ops.object.mode_set(mode='OBJECT')

    ac_n_full = len(merged_full.data.polygons) - sum(eng_face_counts_full)

    # 缩放到真实尺寸
    if config.MODEL_SCALE != 1.0:
        _select_only(merged_full)
        merged_full.scale *= config.MODEL_SCALE
        bpy.ops.object.transform_apply(scale=True)
        print(f"  缩放: ×{config.MODEL_SCALE}")

    # 尾焰位置缩放
    exhaust_positions = [ep * config.MODEL_SCALE for ep in exhaust_positions_model]

    # 对称化（可选）
    if config.SYMMETRIZE_MESH:
        mesh_graph.symmetrize_mesh(merged_full)
        print("  对称化: X=0")

    # 构建 engine_mask_full
    n_faces_full = len(merged_full.data.polygons)
    engine_mask_full = np.zeros(n_faces_full, dtype=bool)
    total_eng = sum(eng_face_counts_full)
    if total_eng > 0:
        engine_mask_full[ac_n_full:] = True

    print(f"  合并完成: {n_faces_full} 面 (机身={ac_n_full}, 发动机={total_eng})")

    # ── 减面 ──
    is_decimated = config.DECIMATE_RATIO < 1.0
    merged = merged_full
    engine_mask = engine_mask_full
    ac_n = ac_n_full
    eng_face_counts = eng_face_counts_full

    if is_decimated:
        print(f"\n[减面] ratio={config.DECIMATE_RATIO}...")

        # 提取减面前的面片中心
        centers_full, _, _ = mesh_graph.get_mesh_data(merged_full)

        # 复制并减面
        merged = _copy_obj(merged_full)
        merged.name = "IR_Unified_Mesh_Decimated"

        _select_only(merged)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.decimate(ratio=config.DECIMATE_RATIO)
        bpy.ops.object.mode_set(mode='OBJECT')

        centers_dec, _, _ = mesh_graph.get_mesh_data(merged)
        print(f"  减面后: {len(centers_dec)} 面 (原 {n_faces_full} 面)")

        # 映射 engine_mask 从原始网格到减面网格
        engine_mask = np.zeros(len(centers_dec), dtype=bool)
        if total_eng > 0:
            eng_idx_full = np.where(engine_mask_full)[0]
            # 每个发动机面片找最近的减面网格面片
            for ei in eng_idx_full:
                d2 = np.sum((centers_dec - centers_full[ei]) ** 2, axis=1)
                nearest = np.argmin(d2)
                engine_mask[nearest] = True

        # 更新 ac_n 和 eng_face_counts
        ac_n = len(centers_dec) - int(engine_mask.sum())
        # 估算各发动机面片数（简化：按比例）
        eng_face_counts = [int(fc * config.DECIMATE_RATIO) for fc in eng_face_counts_full]
        # 调整使总数匹配
        total_dec_eng = int(engine_mask.sum())
        if sum(eng_face_counts) != total_dec_eng:
            eng_face_counts[-1] = total_dec_eng - sum(eng_face_counts[:-1])

        print(f"  发动机面片映射: {engine_mask.sum()} (原 {total_eng})")

        # 隐藏 full mesh
        merged_full.hide_viewport = True
        merged_full.hide_render = True

    return (merged, merged_full if is_decimated else None,
            engine_mask, engine_mask_full if is_decimated else None,
            exhaust_positions, ac_n, ac_n_full if is_decimated else None,
            eng_face_counts, eng_face_counts_full,
            is_decimated)


# ══════════════════════════════════════════════════════════════════════════════
# 热源面片识别
# ══════════════════════════════════════════════════════════════════════════════

def identify_heat_sources(centers, engine_mask, exhaust_positions):
    """选择距尾焰最近的发动机面片作为热源。"""
    n_engine = int(engine_mask.sum())
    n_source_per_exhaust = max(5, min(200, int(n_engine * 0.05)))
    engine_indices = np.where(engine_mask)[0]

    source_faces = set()
    for ep in exhaust_positions:
        dists = np.linalg.norm(centers[engine_indices] - ep, axis=1)
        nearest = engine_indices[np.argsort(dists)[:n_source_per_exhaust]]
        source_faces.update(nearest.tolist())

    print(f"  热源面片: {len(source_faces)} (每尾焰 {n_source_per_exhaust})")
    return source_faces, n_source_per_exhaust


# ══════════════════════════════════════════════════════════════════════════════
# 热平衡求解 T_s
# ══════════════════════════════════════════════════════════════════════════════

def solve_source_temperatures(source_faces, centers, exhaust_positions, areas):
    """求解每个热源面片的 T_s。"""
    T_source_dict = calibrate_compute.solve_all_source_faces(
        list(source_faces), centers, exhaust_positions, areas,
        q_o=config.Q_O, T_o=config.T_EXHAUST, T_amb=config.T_AMB,
        emissivity=config.EMISSIVITY, k_struct=config.K_STRUCTURE,
        a_struct=config.A_STRUCTURE, sigma=config.SIGMA, q_i=config.Q_I,
        tol=config.HEAT_SOURCE_TOL,
    )
    T_s_vals = np.array(list(T_source_dict.values()))
    print(f"  T_s: [{T_s_vals.min():.0f}, {T_s_vals.max():.0f}] K, mean={T_s_vals.mean():.0f} K")
    return T_source_dict


# ══════════════════════════════════════════════════════════════════════════════
# Gauss-Seidel 扩散（含跨边界桥接）
# ══════════════════════════════════════════════════════════════════════════════

def run_diffusion(T_init, centers, neighbors, edge_lengths, source_faces, engine_mask):
    """Gauss-Seidel 扩散。"""
    # 【关键】跨边界结构连接
    n_bridges = calibrate_compute.add_cross_boundary_bridges(
        neighbors, edge_lengths, centers, engine_mask,
        max_pairs=config.CROSS_BOUNDARY_MAX_PAIRS,
        max_distance=config.CROSS_BOUNDARY_MAX_DISTANCE,
    )

    # 算术平均权重
    conductances = {}
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if (i, j) in conductances:
                continue
            conductances[(i, j)] = 1.0
            conductances[(j, i)] = 1.0

    T, iterations, max_change = diffusion.gauss_seidel(
        T_init, neighbors, conductances,
        fixed_faces=source_faces,
        tol=config.DIFFUSION_TOL,
        max_iter=config.MAX_ITERATIONS,
        decay=config.DIFFUSION_DECAY,
        T_amb=config.T_AMB,
    )

    skin_mask = ~engine_mask
    if skin_mask.sum() > 0:
        skin_T = T[skin_mask]
        print(f"  扩散后蒙皮: [{skin_T.min():.1f}, {skin_T.max():.1f}] K, mean={skin_T.mean():.1f} K")

    print(f"  迭代: {iterations}, max ΔT={max_change:.6f} K")
    return T, iterations, max_change


# ══════════════════════════════════════════════════════════════════════════════
# 氨动加热
# ══════════════════════════════════════════════════════════════════════════════

def apply_aero_heating(T):
    """气动加热叠加。"""
    delta_T = config.T_AMB * 0.16 * config.MACH_NUMBER ** 2
    T_aero = T + delta_T
    print(f"  气动加热: M={config.MACH_NUMBER}, ΔT={delta_T:.2f} K")
    print(f"  T range: [{T_aero.min():.1f}, {T_aero.max():.1f}] K")
    return T_aero, delta_T


# ══════════════════════════════════════════════════════════════════════════════
# 辐亮度计算 + 能量衰减
# ══════════════════════════════════════════════════════════════════════════════

def compute_final_radiance(T):
    """温度→辐亮度 + 能量衰减。"""
    L_self = calibrate_compute.compute_radiance(T, config.EMISSIVITY, config.LAMBDA_1, config.LAMBDA_2)
    print(f"  L_self: [{L_self.min():.2f}, {L_self.max():.2f}] W/(m²·sr)")

    denom = 4.0 * config.K_E * config.K_E * (1.0 - config.BETA_RATIO) ** 2
    if denom < 1e-9:
        denom = 1e-9
    eta = config.TAU0 * np.pi / denom

    L_out = L_self * eta
    print(f"  能量衰减: η={eta:.4f}")
    return L_self, L_out, eta


# ══════════════════════════════════════════════════════════════════════════════
# Blender 内计算流程
# ══════════════════════════════════════════════════════════════════════════════

def run_in_blender_compute(merged, exhaust_positions, engine_mask):
    """Blender 内完整计算。"""
    print("\n[计算] Blender 内计算...")

    centers, areas, _ = mesh_graph.get_mesh_data(merged)
    neighbors, edge_lengths = mesh_graph.build_face_adjacency(merged)

    n_faces = len(centers)
    print(f"  面片: {n_faces} (发动机={engine_mask.sum()}, 机身={(~engine_mask).sum()})")

    source_faces, _ = identify_heat_sources(centers, engine_mask, exhaust_positions)

    print("\n[热平衡] 求解 T_s...")
    T_source_dict = solve_source_temperatures(source_faces, centers, exhaust_positions, areas)

    T = np.full(n_faces, config.T_AIRCRAFT_INIT, dtype=np.float64)
    for fi, T_s in T_source_dict.items():
        T[fi] = T_s

    print("\n[扩散] Gauss-Seidel...")
    T, iterations, max_change = run_diffusion(T, centers, neighbors, edge_lengths, source_faces, engine_mask)
    T_diffusion = T.copy()

    print("\n[气动] 加热...")
    T, delta_T = apply_aero_heating(T)
    T_aero = T.copy()

    print("\n[辐亮度] Planck + 衰减...")
    L_self, L, eta = compute_final_radiance(T)
    L_radiance = L_self.copy()

    return T, L, iterations, max_change, T_diffusion, T_aero, L_radiance


# ══════════════════════════════════════════════════════════════════════════════
# 外部计算
# ══════════════════════════════════════════════════════════════════════════════

def _build_csr_from_neighbors(neighbors, edge_lengths):
    """从 list of lists 构建 CSR。"""
    n = len(neighbors)
    offsets = np.zeros(n + 1, dtype=np.int32)
    counts = [len(nbrs) for nbrs in neighbors]
    offsets[1:] = np.cumsum(counts)
    total_edges = offsets[-1]

    indices = np.empty(total_edges, dtype=np.int32)
    edge_lens_flat = np.empty(total_edges, dtype=np.float32)

    for i in range(n):
        start = offsets[i]
        nbrs = neighbors[i]
        for k, j in enumerate(nbrs):
            indices[start + k] = j
            edge_lens_flat[start + k] = edge_lengths.get((i, j), 1.0)

    return offsets, indices, edge_lens_flat


def export_for_compute(output_path, centers, areas, neighbors, edge_lengths,
                       exhaust_positions, engine_mask):
    """导出网格数据为 .npz。"""
    offsets, indices, edge_lens_flat = _build_csr_from_neighbors(neighbors, edge_lengths)

    np.savez_compressed(output_path,
        centers=centers.astype(np.float32),
        areas=areas.astype(np.float32),
        offsets=offsets,
        indices=indices,
        edge_lens=edge_lens_flat,
        exhaust_positions=np.array(exhaust_positions, dtype=np.float32),
        engine_mask=engine_mask,
        # 配置参数
        T_EXHAUST=config.T_EXHAUST, Q_O=config.Q_O,
        T_AIRCRAFT_INIT=config.T_AIRCRAFT_INIT, T_AMB=config.T_AMB,
        EMISSIVITY=config.EMISSIVITY, K_STRUCTURE=config.K_STRUCTURE,
        A_STRUCTURE=config.A_STRUCTURE, SIGMA=config.SIGMA,
        HEAT_SOURCE_TOL=config.HEAT_SOURCE_TOL,
        DIFFUSION_TOL=config.DIFFUSION_TOL, MAX_ITERATIONS=config.MAX_ITERATIONS,
        DIFFUSION_DECAY=config.DIFFUSION_DECAY, Q_I=config.Q_I,
        MACH_NUMBER=config.MACH_NUMBER, LAMBDA_1=config.LAMBDA_1, LAMBDA_2=config.LAMBDA_2,
        TAU0=config.TAU0, K_E=config.K_E, BETA_RATIO=config.BETA_RATIO,
        CROSS_BOUNDARY_MAX_PAIRS=config.CROSS_BOUNDARY_MAX_PAIRS,
        CROSS_BOUNDARY_MAX_DISTANCE=config.CROSS_BOUNDARY_MAX_DISTANCE,
    )


def import_compute_results(input_path):
    """读回外部计算结果。"""
    data = np.load(input_path)
    return {
        'T': data['T'],
        'L': data['L'],
        'iterations': int(data['iterations']),
        'max_change': float(data['max_change']),
        'T_diffusion': data.get('T_diffusion', None),
        'T_aero': data.get('T_aero', None),
        'L_radiance': data.get('L_radiance', None),
    }


def run_external_compute(merged, exhaust_positions, engine_mask):
    """导出 → 子进程 → 导入。"""
    python_exe = _find_venv_python()
    if python_exe is None:
        print("[外部计算] .venv 未找到，回退")
        return None, None, 0, 0, None, None, None

    standalone = os.path.join(_project_root, "new_pipeline", "compute_standalone.py")
    if not os.path.isfile(standalone):
        print(f"[外部计算] 脚本未找到: {standalone}")
        return None, None, 0, 0, None, None, None

    tmpdir = tempfile.gettempdir()
    input_npz = os.path.join(tmpdir, "_blir_main_input.npz")
    output_npz = os.path.join(tmpdir, "_blir_main_output.npz")

    try:
        print("\n[外部计算] 导出...")
        centers, areas, _ = mesh_graph.get_mesh_data(merged)
        neighbors, edge_lengths = mesh_graph.build_face_adjacency(merged)
        export_for_compute(input_npz, centers, areas, neighbors, edge_lengths,
                          exhaust_positions, engine_mask)

        print(f"[外部计算] 启动: {python_exe}")
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(
            [python_exe, standalone, input_npz, output_npz],
            capture_output=True, text=True, timeout=600,
            encoding='utf-8', errors='replace', env=env,
        )

        if result.stdout:
            for line in result.stdout.strip().splitlines():
                print(f"  {line}")

        if result.returncode != 0:
            print(f"[外部计算] 失败 (code={result.returncode})")
            return None, None, 0, 0, None, None, None

        res = import_compute_results(output_npz)
        return (res['T'], res['L'], res['iterations'], res['max_change'],
                res['T_diffusion'], res['T_aero'], res['L_radiance'])

    except subprocess.TimeoutExpired:
        print("[外部计算] 超时")
        return None, None, 0, 0, None, None, None
    except Exception as e:
        print(f"[外部计算] 异常: {e}")
        return None, None, 0, 0, None, None, None
    finally:
        for f in (input_npz, output_npz):
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except OSError:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# 统计输出
# ══════════════════════════════════════════════════════════════════════════════

def print_statistics(T, L, engine_mask, source_faces):
    """输出统计。"""
    delta_T_aero = config.T_AMB * 0.16 * config.MACH_NUMBER ** 2

    print(f"\n{'='*60}")
    print(f"  稳态温度计算结果")
    print(f"{'='*60}")
    print(f"  网格面片: {len(T)}")
    print(f"  热源面片: {len(source_faces)}")
    print(f"  气动加热: M={config.MACH_NUMBER}, ΔT={delta_T_aero:.2f} K")
    print()

    print(f"  {'区域':12s} | {'最低':>10s}  {'最高':>10s}  {'平均':>10s}")
    print(f"  {'-'*12}-+-{'-'*10}--{'-'*10}--{'-'*10}")
    print(f"  {'整体 T':12s} | {T.min():10.2f}  {T.max():10.2f}  {T.mean():10.2f} K")

    skin_mask = ~engine_mask
    if skin_mask.sum() > 0:
        skin_T = T[skin_mask]
        print(f"  {'蒙皮 T':12s} | {skin_T.min():10.2f}  {skin_T.max():10.2f}  {skin_T.mean():10.2f} K")

    if engine_mask.sum() > 0:
        eng_T = T[engine_mask]
        print(f"  {'发动机 T':12s} | {eng_T.min():10.2f}  {eng_T.max():10.2f}  {eng_T.mean():10.2f} K")

    print()
    print(f"  {'整体 L':12s} | {L.min():10.2f}  {L.max():10.2f}  {L.mean():10.2f} W/(m²·sr)")
    print(f"\n{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# 过程图片
# ══════════════════════════════════════════════════════════════════════════════

def render_process_images(merged, T_diffusion, T_aero, L_radiance, L_final):
    """输出过程图片。"""
    img_dir = bpy.path.abspath(config.PROCESS_IMAGES_DIR)
    os.makedirs(img_dir, exist_ok=True)

    cam_loc = (-35, -70, -8)
    target = (0, 0, 4)

    def _render_step(face_vals, filename, label, color_mode):
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

    print(f"\n[过程图片] 输出...")
    if T_diffusion is not None:
        _render_step(T_diffusion, "01_temperature_diffusion.png", "扩散后温度", "thermal")
    if T_aero is not None:
        _render_step(T_aero, "02_temperature_aero.png", "气动加热后温度", "thermal")
    if L_radiance is not None:
        _render_step(L_radiance, "03_radiance.png", "Planck辐亮度", "bw")
    _render_step(L_final, "04_radiance_final.png", "最终辐亮度", "bw")


# ══════════════════════════════════════════════════════════════════════════════
# 材质应用 + 保存
# ══════════════════════════════════════════════════════════════════════════════

def apply_material_and_save(merged, merged_full, L, T, engine_mask,
                             is_decimated, aircraft, all_engines,
                             engines_left, engines_right):
    """应用材质并保存。"""
    # 辐亮度归一化
    vmin = float(L.min())
    vmax = float(np.percentile(L, config.RENDER_VMAX_PERCENTILE))
    if vmax <= vmin:
        vmax = vmin + 1.0

    print(f"\n[材质] 辐亮度范围: {vmin:.2f} ~ {vmax:.2f} W/(m²·sr)")

    # 如果减面，上采样到原始网格
    if is_decimated and merged_full is not None:
        print("  [上采样] 映射到原始网格...")
        centers_dec, _, _ = mesh_graph.get_mesh_data(merged)
        centers_full, _, _ = mesh_graph.get_mesh_data(merged_full)

        T = upsample_to_original(centers_dec, T, centers_full)
        L = upsample_to_original(centers_dec, L, centers_full)

        # 删除减面网格
        dec_mesh = merged.data
        bpy.data.objects.remove(merged, do_unlink=True)
        if dec_mesh.users == 0:
            bpy.data.meshes.remove(dec_mesh)

        merged = merged_full
        merged.hide_viewport = False
        merged.hide_render = False

        print(f"  上采样完成: {len(L)} 面")

    # 应用材质到 merged（作为最终输出对象）
    visualize.clear_scene_materials()
    visualize.assign_value_material(
        merged, merged.data, L,
        attr_name="Radiance",
        color_mode=config.RENDER_COLOR_MODE,
        vmin=vmin, vmax=vmax,
        mat_name="IR_Radiance",
    )

    # 重命名合并网格为 Aircraft (统一输出命名)
    merged.name = "Aircraft"

    # 删除原始对象
    if aircraft and aircraft.name in bpy.data.objects:
        bpy.data.objects.remove(aircraft, do_unlink=True)
    for eng in all_engines:
        if eng and eng.name in bpy.data.objects:
            bpy.data.objects.remove(eng, do_unlink=True)

    # 清理孤立材质和 mesh
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)

    # 保存
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


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """主入口。"""
    t_start = time.time()

    # 刷新模块
    import importlib as _il
    for _mn in ("new_pipeline.config", "new_pipeline.mesh_graph",
                "new_pipeline.heat_source", "new_pipeline.diffusion",
                "new_pipeline.visualize", "new_pipeline.calibrate_compute"):
        if _mn in sys.modules:
            _il.reload(sys.modules[_mn])

    print("\n" + "="*60)
    print("  BLIR 新管线 - 稳态温度计算")
    print("="*60)

    # 1. 查找对象
    print("\n[准备] 查找对象...")
    aircraft = mesh_graph.find_aircraft()
    engines_left, engines_right = mesh_graph.find_all_engines()
    all_engines = engines_left + engines_right

    if aircraft is None:
        print("[错误] 未找到蒙皮网格")
        return

    exhaust_positions_model = []
    for eng in all_engines:
        pos = mesh_graph.find_exhaust_position(eng)
        if pos is not None:
            exhaust_positions_model.append(np.array(pos))

    if not exhaust_positions_model:
        print("[错误] 未找到发动机尾焰位置")
        return

    print(f"  Aircraft: {aircraft.name} ({len(aircraft.data.polygons)} 面)")
    for side, eng_list in [('L', engines_left), ('R', engines_right)]:
        for idx, eng in enumerate(eng_list):
            print(f"  {_engine_name(side, idx)}: {eng.name} ({len(eng.data.polygons)} 面)")

    # 2. 准备网格（含减面）
    print("\n[网格] 合并 + 缩放...")
    (merged, merged_full, engine_mask, engine_mask_full,
     exhaust_positions, ac_n, ac_n_full,
     eng_face_counts, eng_face_counts_full,
     is_decimated) = prepare_mesh(aircraft, all_engines, exhaust_positions_model)

    centers, _, _ = mesh_graph.get_mesh_data(merged)

    # 3. 计算
    if config.USE_EXTERNAL_COMPUTE:
        T, L, iterations, max_change, T_diffusion, T_aero, L_radiance = \
            run_external_compute(merged, exhaust_positions, engine_mask)
        if T is None:
            print("\n[外部计算] 回退到 Blender 内计算")
            T, L, iterations, max_change, T_diffusion, T_aero, L_radiance = \
                run_in_blender_compute(merged, exhaust_positions, engine_mask)
    else:
        T, L, iterations, max_change, T_diffusion, T_aero, L_radiance = \
            run_in_blender_compute(merged, exhaust_positions, engine_mask)

    # 4. 统计
    source_faces, _ = identify_heat_sources(centers, engine_mask, exhaust_positions)
    print_statistics(T, L, engine_mask, source_faces)

    # 5. 过程图片
    if config.PROCESS_IMAGES_ENABLED:
        render_process_images(merged, T_diffusion, T_aero, L_radiance, L)

    # 6. 材质 + 保存
    apply_material_and_save(
        merged, merged_full, L, T, engine_mask, is_decimated,
        aircraft, all_engines, engines_left, engines_right
    )

    print(f"\n完成，耗时: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()