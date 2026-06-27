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

# 尝试导入 PIL 用于颜色条生成
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


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

    # ── 扩面（细分） ──
    is_subdivided = config.SUBDIVIDE_LEVEL > 0
    if is_subdivided:
        print(f"\n[扩面] level={config.SUBDIVIDE_LEVEL}...")
        n_before = len(merged_full.data.polygons)

        _select_only(merged_full)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        for level in range(config.SUBDIVIDE_LEVEL):
            bpy.ops.mesh.subdivide()
        bpy.ops.object.mode_set(mode='OBJECT')

        n_after = len(merged_full.data.polygons)
        print(f"  细分后: {n_after} 面 (原 {n_before} 面, ×{n_after/n_before:.1f})")

        # 更新面片计数
        n_faces_full = n_after
        # 细分后发动机面片数也相应增加
        eng_face_counts_full = [int(fc * (n_after / n_before)) for fc in eng_face_counts_full]
        # 调整使总数匹配
        total_eng_new = sum(eng_face_counts_full)
        if total_eng_new != n_faces_full - ac_n_full * (n_after // (n_before - total_eng if n_before - total_eng > 0 else 1)):
            # 简化：发动机区域也按比例细分
            eng_face_counts_full[-1] = n_faces_full - ac_n_full - sum(eng_face_counts_full[:-1])

        ac_n_full = n_faces_full - sum(eng_face_counts_full)
        total_eng = sum(eng_face_counts_full)

        # 重建 engine_mask_full
        engine_mask_full = np.zeros(n_faces_full, dtype=bool)
        if total_eng > 0:
            engine_mask_full[ac_n_full:] = True

        print(f"  细分后面片: 机身={ac_n_full}, 发动机={total_eng}")

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
    """选择距尾焰最近的发动机面片作为热源。

    热源面片数量应该很少（每尾焰 2-3 片），模拟热量从尾焰核心
    通过少数连接点传导到发动机蒙皮。
    """
    engine_indices = np.where(engine_mask)[0]

    # 固定数量：每尾焰取最近的 2 片（不是 5%）
    n_source_per_exhaust = config.HEAT_SOURCE_COUNT  # 从 config 读取

    source_faces = set()
    for ep in exhaust_positions:
        dists = np.linalg.norm(centers[engine_indices] - ep, axis=1)
        nearest = engine_indices[np.argsort(dists)[:n_source_per_exhaust]]
        source_faces.update(nearest.tolist())

    print(f"  热源面片: {len(source_faces)} (每尾焰 {n_source_per_exhaust} 片)")
    return source_faces, n_source_per_exhaust


# ══════════════════════════════════════════════════════════════════════════════
# q_o 校准：二分搜索使发动机均温 ≈ 350K
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_qo(centers, areas, neighbors, edge_lengths, engine_mask, exhaust_positions,
                 target_temp=350.0, tol=5.0, max_iter=30):
    """二分搜索 q_o 使发动机均温 ≈ target_temp。

    校准流程:
      1. 设 q_o 未知
      2. 求解热源面片 T_s (热平衡方程)
      3. Gauss-Seidel 扩散 (仅发动机面片参与，机身初始280K但不固定)
      4. 计算发动机均温
      5. 调整 q_o 直至均温 ≈ target_temp

    Args:
        centers, areas: 网格数据
        neighbors, edge_lengths: 邻接图
        engine_mask: 发动机面片标记
        exhaust_positions: 尾焰位置
        target_temp: 目标发动机均温 (默认 350K)
        tol: 温度容差 (默认 5K)
        max_iter: 最大二分迭代次数

    Returns:
        q_o_calibrated: 校准后的 q_o 值
    """
    print(f"\n[校准 q_o] 目标: 发动机均温 ≈ {target_temp} K")

    n_faces = len(centers)
    source_faces, _ = identify_heat_sources(centers, engine_mask, exhaust_positions)
    engine_indices = np.where(engine_mask)[0]

    # 转换 neighbors 为纯 Python list（避免 numpy 问题）
    neighbors_list = [list(nbrs) for nbrs in neighbors]
    edge_lengths_dict = dict(edge_lengths)

    # 跨边界连接（仅做一次，只从热源面片连接）
    n_bridges = calibrate_compute.add_cross_boundary_bridges(
        neighbors_list, edge_lengths_dict, centers, engine_mask,
        max_pairs=config.CROSS_BOUNDARY_MAX_PAIRS,
        max_distance=config.CROSS_BOUNDARY_MAX_DISTANCE,
        source_faces=source_faces,
    )
    print(f"  跨边界桥接: {n_bridges} 条边")

    # 构建传导系数（根据配置选择权重模式）
    conductances = {}

    if config.USE_DISTANCE_WEIGHTED_DIFFUSION:
        # 边长度权重: G_ij = typical_len / d_ij
        pos_lens = [abs(v) for v in edge_lengths_dict.values() if v > 0]
        typical_len = float(np.median(pos_lens)) if pos_lens else 1.0
        print(f"  权重模式: 边长度权重 (typical_len={typical_len:.4f} m)")

        for i, nbrs in enumerate(neighbors_list):
            for j in nbrs:
                if (i, j) not in conductances:
                    d = edge_lengths_dict.get((i, j), typical_len)
                    if d < 1e-9:
                        d = 1e-9
                    G = typical_len / d
                    conductances[(i, j)] = G
                    conductances[(j, i)] = G
    else:
        # 算术平均权重: G_ij = 1.0
        print(f"  权重模式: 算术平均权重")
        for i, nbrs in enumerate(neighbors_list):
            for j in nbrs:
                if (i, j) not in conductances:
                    conductances[(i, j)] = 1.0
                    conductances[(j, i)] = 1.0

    def _run_diffusion_for_qo(q_o):
        """给定 q_o，运行扩散并返回发动机均温。"""
        # 求解热源面片 T_s
        T_source_dict = calibrate_compute.solve_all_source_faces(
            list(source_faces), centers, exhaust_positions, areas,
            q_o=q_o, T_o=config.T_EXHAUST, T_amb=config.T_AMB,
            emissivity=config.EMISSIVITY, k_struct=config.K_STRUCTURE,
            a_struct=config.A_STRUCTURE, sigma=config.SIGMA, q_i=config.Q_I,
            tol=config.HEAT_SOURCE_TOL,
        )

        # 初始化温度场
        T = np.full(n_faces, config.T_AIRCRAFT_INIT, dtype=np.float64)
        for fi, T_s in T_source_dict.items():
            T[fi] = T_s

        # Gauss-Seidel 扩散（不使用 numba，避免 JIT 问题）
        T, _, _ = diffusion.gauss_seidel(
            T, neighbors_list, conductances,
            fixed_faces=source_faces,
            tol=1.0,  # 校准时用较大容差加速
            max_iter=5000,
            decay=config.DIFFUSION_DECAY,
            T_amb=config.T_AMB,
        )

        # 计算发动机均温
        eng_T = T[engine_mask]
        return eng_T.mean(), eng_T.min(), eng_T.max()

    # 二分搜索
    q_lo, q_hi = 0.01, 1000.0  # q_o 搜索范围（放宽上限）
    best_qo = config.Q_O

    for iteration in range(max_iter):
        q_mid = (q_lo + q_hi) * 0.5
        eng_mean, eng_min, eng_max = _run_diffusion_for_qo(q_mid)

        print(f"  iter {iteration+1}: q_o={q_mid:.4f} W → 发动机均温={eng_mean:.1f} K [{eng_min:.1f}, {eng_max:.1f}]")

        if abs(eng_mean - target_temp) < tol:
            best_qo = q_mid
            print(f"  ✓ 校准完成: q_o = {best_qo:.4f} W")
            return best_qo

        if eng_mean < target_temp:
            q_lo = q_mid  # 温度太低，增加 q_o
        else:
            q_hi = q_mid  # 温度太高，减小 q_o

    best_qo = (q_lo + q_hi) * 0.5
    print(f"  校准收敛: q_o ≈ {best_qo:.4f} W (均温 ≈ {target_temp} K)")
    return best_qo

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
    source_areas = areas[list(source_faces)]
    print(f"  T_s: [{T_s_vals.min():.0f}, {T_s_vals.max():.0f}] K, mean={T_s_vals.mean():.0f} K")
    print(f"  热源面片面积: min={source_areas.min()*1e4:.2f} cm², max={source_areas.max()*1e4:.2f} cm²")
    print(f"  辐射散热估算 (T=900K): εσA(T⁴-T_amb⁴) = {0.85*5.67e-8*(900**4-280**4)*source_areas.min():.2f} ~ {0.85*5.67e-8*(900**4-280**4)*source_areas.max():.2f} W")
    print(f"  q_o = {config.Q_O:.3f} W (传入热量)")
    return T_source_dict


# ══════════════════════════════════════════════════════════════════════════════
# 温度场对称化
# ══════════════════════════════════════════════════════════════════════════════

def symmetrize_temperature(T, centers):
    """将温度场沿X=0对称化（左=右的平均值）。

    解决Gauss-Seidel迭代方向偏差导致的左右温度不对称问题。
    与SYMMETRIZE_MESH不同，只对称化温度值，不改变网格几何。

    Args:
        T: (N,) 温度数组
        centers: (N, 3) 面片中心坐标

    Returns:
        T_sym: (N,) 对称化后的温度数组
    """
    n = len(T)
    T_sym = T.copy()

    # 找到X<0（左）和X>0（右）的面片
    left_idx = np.where(centers[:, 0] < -1e-6)[0]
    right_idx = np.where(centers[:, 0] > 1e-6)[0]

    if len(left_idx) == 0 or len(right_idx) == 0:
        print(f"  [对称化] 未找到左右面片，跳过")
        return T_sym

    # 对每个左侧面片找镜像右侧面片
    paired_count = 0
    for i in left_idx:
        # 镜像位置：翻转X坐标
        mirror_pos = centers[i].copy()
        mirror_pos[0] = -mirror_pos[0]

        # 找最近的右侧面片
        dists = np.linalg.norm(centers[right_idx] - mirror_pos, axis=1)
        j = right_idx[np.argmin(dists)]

        # 如果距离足够近（认为是镜像面片），取平均
        if dists.min() < 0.1:  # 10cm容差
            avg_T = (T_sym[i] + T_sym[j]) / 2
            T_sym[i] = avg_T
            T_sym[j] = avg_T
            paired_count += 1

    # X≈0的中线面片保持不变
    print(f"  [对称化] 已配对 {paired_count} 组面片，左{len(left_idx)}右{len(right_idx)}")

    return T_sym


# ══════════════════════════════════════════════════════════════════════════════
# Gauss-Seidel 扩散（含跨边界桥接）
# ══════════════════════════════════════════════════════════════════════════════

def run_diffusion(T_init, centers, neighbors, edge_lengths, source_faces, engine_mask):
    """Gauss-Seidel 扩散。"""
    # 【关键】跨边界结构连接（只从热源面片连接）
    n_bridges = calibrate_compute.add_cross_boundary_bridges(
        neighbors, edge_lengths, centers, engine_mask,
        max_pairs=config.CROSS_BOUNDARY_MAX_PAIRS,
        max_distance=config.CROSS_BOUNDARY_MAX_DISTANCE,
        source_faces=source_faces,
    )

    # 构建传导系数（根据配置选择权重模式）
    conductances = {}

    if config.USE_DISTANCE_WEIGHTED_DIFFUSION:
        # 边长度权重: G_ij = typical_len / d_ij
        pos_lens = [abs(v) for v in edge_lengths.values() if v > 0]
        typical_len = float(np.median(pos_lens)) if pos_lens else 1.0
        print(f"  权重模式: 边长度权重 (typical_len={typical_len:.4f} m)")

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                if (i, j) in conductances:
                    continue
                d = edge_lengths.get((i, j), typical_len)
                if d < 1e-9:
                    d = 1e-9
                G = typical_len / d
                conductances[(i, j)] = G
                conductances[(j, i)] = G
    else:
        # 算术平均权重: G_ij = 1.0
        print(f"  权重模式: 算术平均权重")
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

    # ── q_o 校准 ──
    if config.CALIBRATE_Q_O:
        calibrated_qo = calibrate_qo(
            centers, areas, neighbors, edge_lengths, engine_mask, exhaust_positions,
            target_temp=config.CALIBRATE_TARGET_TEMP,
        )
        # 更新 config.Q_O（仅当前会话有效）
        config.Q_O = calibrated_qo
        print(f"\n[校准完成] 使用 q_o = {config.Q_O:.4f} W 进行后续计算")

    source_faces, _ = identify_heat_sources(centers, engine_mask, exhaust_positions)

    print("\n[热平衡] 求解 T_s...")
    T_source_dict = solve_source_temperatures(source_faces, centers, exhaust_positions, areas)

    T = np.full(n_faces, config.T_AIRCRAFT_INIT, dtype=np.float64)
    for fi, T_s in T_source_dict.items():
        T[fi] = T_s

    print("\n[扩散] Gauss-Seidel...")
    T, iterations, max_change = run_diffusion(T, centers, neighbors, edge_lengths, source_faces, engine_mask)

    # 温度场对称化（消除迭代方向偏差）
    if config.SYMMETRIZE_TEMPERATURE:
        print("\n[对称化] 温度场...")
        T = symmetrize_temperature(T, centers)

    T_diffusion = T.copy()
    print_temperature_distribution(T_diffusion, "扩散后温度")

    print("\n[气动] 加热...")
    T, delta_T = apply_aero_heating(T)
    T_aero = T.copy()
    print_temperature_distribution(T_aero, "气动加热后温度")

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
        HEAT_SOURCE_COUNT=config.HEAT_SOURCE_COUNT,  # 热源面片数量
        USE_DISTANCE_WEIGHTED_DIFFUSION=config.USE_DISTANCE_WEIGHTED_DIFFUSION,  # 权重模式
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

        # ── q_o 校准 ──
        if config.CALIBRATE_Q_O:
            calibrated_qo = calibrate_qo(
                centers, areas, neighbors, edge_lengths, engine_mask, exhaust_positions,
                target_temp=config.CALIBRATE_TARGET_TEMP,
            )
            # 更新 config.Q_O（仅当前会话有效）
            config.Q_O = calibrated_qo
            print(f"\n[校准完成] 使用 q_o = {config.Q_O:.4f} W 进行外部计算")

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

        # 温度场对称化（消除迭代方向偏差）
        if config.SYMMETRIZE_TEMPERATURE:
            print("\n[对称化] 温度场...")
            if res['T'] is not None:
                res['T'] = symmetrize_temperature(res['T'], centers)
            if res['T_diffusion'] is not None:
                res['T_diffusion'] = symmetrize_temperature(res['T_diffusion'], centers)
            if res['T_aero'] is not None:
                res['T_aero'] = symmetrize_temperature(res['T_aero'], centers)

        # 输出温度分布统计
        if res['T_diffusion'] is not None:
            print_temperature_distribution(res['T_diffusion'], "扩散后温度(外部)")
        if res['T_aero'] is not None:
            print_temperature_distribution(res['T_aero'], "气动加热后温度(外部)")

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

def print_temperature_distribution(T, label="温度分布"):
    """输出温度分布统计：百分位和区间分布。

    帮助用户决定 vmin/vmax 的设置。
    """
    n = len(T)
    print(f"\n[{label}] 面片数: {n}")
    print(f"  温度范围: {T.min():.1f} ~ {T.max():.1f} K")

    # 百分位统计
    percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    print(f"\n  百分位统计:")
    print(f"  {'百分位':>8s} | {'温度(K)':>8s} | {'累计面片':>10s}")
    print(f"  {'-'*8}-+-{'-'*8}--+-{'-'*10}")
    for p in percentiles:
        t_val = np.percentile(T, p)
        count = int(n * p / 100)
        print(f"  {p:>6d}% | {t_val:>8.1f} | {count:>10d}")

    # 区间分布（按温度范围划分）
    T_min = float(T.min())
    T_max = float(T.max())
    T_range = T_max - T_min

    if T_range > 0:
        # 分成 10 个区间
        bins = np.linspace(T_min, T_max, 11)
        counts = np.histogram(T, bins=bins)[0]

        print(f"\n  温度区间分布:")
        print(f"  {'区间(K)':>18s} | {'面片数':>8s} | {'百分比':>8s}")
        print(f"  {'-'*18}-+-{'-'*8}--+-{'-'*8}")
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            pct = counts[i] / n * 100
            print(f"  {lo:>8.1f} ~ {hi:>6.1f} | {counts[i]:>8d} | {pct:>6.1f}%")

    print()


def print_statistics(T, L, engine_mask, source_faces):
    """输出统计。"""
    delta_T_aero = config.T_AMB * 0.16 * config.MACH_NUMBER ** 2

    print(f"\n{'='*60}")
    print(f"  稳态温度计算结果")
    print(f"{'='*60}")
    print(f"  网格面片: {len(T)}")
    print(f"  热源面片: {len(source_faces)}")
    print(f"  氨动加热: M={config.MACH_NUMBER}, ΔT={delta_T_aero:.2f} K")
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

    # 添加温度分布统计
    print_temperature_distribution(T, "最终温度分布")

    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# 颜色条生成
# ══════════════════════════════════════════════════════════════════════════════

def get_color_for_value(color_mode, t):
    """根据 color_mode 和归一化值 t 返回 RGBA 颜色。

    t ∈ [0, 1]，0 = 最小值，1 = 最大值
    颜色映射与 visualize.py Color Ramp 一致。

    thermal: 深蓝(0) → 青(0.25) → 绿(0.45) → 黄(0.65) → 橙(0.85) → 红(1)
    bw: 灰色平台(gray_zone以下) → 白色
    """
    from new_pipeline import config

    if color_mode == "thermal":
        # thermal 颜色渐变 (与 visualize.py Color Ramp 一致)
        # 拉长低温区色标间距，让蓝→青→绿过渡更平滑
        # 0.0: (0.0, 0.0, 0.5) 深蓝
        # 0.25: (0.0, 0.5, 1.0) 青色
        # 0.45: (0.0, 1.0, 0.0) 绿色
        # 0.65: (1.0, 1.0, 0.0) 黄色
        # 0.85: (1.0, 0.5, 0.0) 橙色
        # 1.0: (1.0, 0.0, 0.0) 红色
        if t < 0.25:
            s = t / 0.25
            r, g, b = 0.0, 0.0 + 0.5 * s, 0.5 + 0.5 * s
        elif t < 0.45:
            s = (t - 0.25) / 0.20
            r, g, b = 0.0, 0.5 + 0.5 * s, 1.0 - s
        elif t < 0.65:
            s = (t - 0.45) / 0.20
            r, g, b = 0.0 + s, 1.0, 0.0
        elif t < 0.85:
            s = (t - 0.65) / 0.20
            r, g, b = 1.0, 1.0 - 0.5 * s, 0.0
        else:
            s = (t - 0.85) / 0.15
            r, g, b = 1.0, 0.5 - 0.5 * s, 0.0
        return (int(r * 255), int(g * 255), int(b * 255), 255)

    elif color_mode == "bw":
        # bw 模式: gray_zone 平台 → 白色 (与 visualize.py 一致)
        gray_zone = float(config.RENDER_BW_GRAY_ZONE)
        min_gray = float(config.RENDER_BW_MIN_GRAY)
        saturation = float(config.RENDER_BW_SATURATION)

        if t < gray_zone:
            # 灰色平台 (Color Ramp extrapolate)
            gray = min_gray
        else:
            # 灰 → 白 (linear interpolation)
            s = (t - gray_zone) / (1.0 - gray_zone)
            gray = min_gray + s * (saturation - min_gray)

        g_val = int(gray * 255)
        return (g_val, g_val, g_val, 255)

    return (128, 128, 128, 255)


def composite_with_colorbar(render_path, output_dir, color_mode, vmin, vmax, output_filename):
    """将渲染图与颜色条合成到一张图。

    Args:
        render_path: 渲染图路径
        output_dir: 输出目录
        color_mode: 'thermal' 或 'bw'
        vmin, vmax: 数值范围
        output_filename: 输出文件名

    合成布局: 渲染图(左) + 颜色条(右)
    """
    if not HAS_PIL:
        # 没有 PIL，直接复制渲染图
        import shutil
        shutil.copy(render_path, os.path.join(output_dir, output_filename))
        return

    # 加载渲染图
    render_img = Image.open(render_path)
    render_w, render_h = render_img.size

    # 颜色条尺寸
    bar_width = 40
    bar_height = render_h - 40  # 留出上下边距
    margin = 20
    text_width = 60
    total_bar_width = bar_width + margin + text_width

    # 创建合成图
    composite_w = render_w + total_bar_width + 20
    composite_h = render_h
    composite = Image.new('RGBA', (composite_w, composite_h), (255, 255, 255, 255))

    # 放入渲染图
    composite.paste(render_img, (0, 0))

    # 生成颜色条
    bar_x = render_w + 20
    bar_y = 20

    # 绘制颜色条
    draw = ImageDraw.Draw(composite)

    for y in range(bar_y, bar_y + bar_height):
        t = (y - bar_y) / bar_height  # 0 (top/vmax) → 1 (bottom/vmin)
        t_inv = 1 - t  # 反转，顶部是 vmax
        color = get_color_for_value(color_mode, t_inv)
        draw.rectangle([bar_x, y, bar_x + bar_width, y + 1], fill=color)

    # 尝试使用系统字体
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    # 标注数值
    unit = "K" if color_mode == "thermal" else "W/(m²·sr)"
    n_labels = 5
    for i in range(n_labels):
        frac = i / (n_labels - 1)
        y = bar_y + int(frac * bar_height)
        val = vmax - frac * (vmax - vmin)
        text = f"{val:.1f}"
        # 文字居中于颜色条右侧
        draw.text((bar_x + bar_width + 8, y - 7), text, fill=(50, 50, 50, 255), font=font)

    # 单位标注（底部）
    draw.text((bar_x + bar_width + 8, bar_y + bar_height + 5), unit, fill=(80, 80, 80, 255), font=font)

    # 标题（顶部）
    title = "温度" if color_mode == "thermal" else "辐射"
    draw.text((bar_x + 5, 5), title, fill=(80, 80, 80, 255), font=font)

    # 保存
    composite.save(os.path.join(output_dir, output_filename))


# ══════════════════════════════════════════════════════════════════════════════
# 孤立面修复
# ══════════════════════════════════════════════════════════════════════════════

def fix_isolated_faces(T, centers, T_init):
    """修复未被温度传播到的孤立面。

    孤立面特征：温度仍为初始温度 T_init（未被扩散影响）。
    修复策略：找到最近的非孤立面，用其温度填充。

    Args:
        T: (N,) 温度数组
        centers: (N, 3) 面片中心坐标
        T_init: 初始温度阈值

    Returns:
        T_fixed: 修复后的温度数组
        n_fixed: 修复的面片数
    """
    # 孤立面判定：温度接近初始温度（温差小于 0.5K）
    isolated_mask = np.abs(T - T_init) < 0.5
    n_isolated = int(isolated_mask.sum())

    if n_isolated == 0:
        return T, 0

    print(f"  [孤立面修复] 发现 {n_isolated} 个孤立面 (T≈{T_init:.1f}K)")

    non_isolated_idx = np.where(~isolated_mask)[0]
    isolated_idx = np.where(isolated_mask)[0]

    if len(non_isolated_idx) == 0:
        print(f"  [WARN] 全部面片均为孤立，无法修复")
        return T, 0

    # 为每个孤立面找最近的非孤立面
    non_isolated_centers = centers[non_isolated_idx]
    T_fixed = T.copy()

    for idx in isolated_idx:
        c = centers[idx]
        dists = np.linalg.norm(non_isolated_centers - c, axis=1)
        nearest = non_isolated_idx[np.argmin(dists)]
        T_fixed[idx] = T[nearest]

    print(f"  [孤立面修复] 已修复，温度范围调整: {T.min():.1f} → {T_fixed.min():.1f} K")
    return T_fixed, n_isolated


# ══════════════════════════════════════════════════════════════════════════════
# 过程图片
# ══════════════════════════════════════════════════════════════════════════════

def render_process_images(merged, T_diffusion, T_aero, L_final):
    """输出过程图片。

    图片列表:
      00_solid.png        - 无材质模型示意图
      01_wireframe.png    - mesh线框示意图
      02_temperature_diff.png  - 扩散后温度（彩色）+ 颜色条合成
      03_temperature_aero.png  - 气动加热后温度（彩色）+ 颜色条合成
      04_radiance.png     - 辐射分布图（灰白）

    两张温度图共用同一色域（基于T_aero的vmax），呈现升温效果。
    相机角度: 从机头前方(Y+)往机尾(-Y)看，从下往上仰视
    背景: 纯白色
    """
    img_dir = bpy.path.abspath(config.PROCESS_IMAGES_DIR)
    os.makedirs(img_dir, exist_ok=True)

    # 相机: 机头前方(Y>0)，右侧(X>0)，低位置(Z<0)，仰视看飞机上方
    # 从下往上看：相机位置很低，目标位置较高
    cam_loc = (25, 70, -100)   # 右侧，机头前方，更低位置
    target = (0, -10, 18)     # 看向机尾上方较高位置

    # ── 温度色标范围（从 config.py 读取）──
    T_vmin = config.T_VMIN
    T_vmax = config.T_VMAX
    T_range = T_vmax - T_vmin

    print(f"\n[过程图片] 输出...")
    print(f"  温度色标: vmin={T_vmin:.1f}K, vmax={T_vmax:.1f}K (范围{int(T_range)}K)")

    # 范围限制：确保颜色分布合理
    if T_range < 50:
        T_vmax = T_vmin + 50
        T_range = 50

    L_vmin = float(L_final.min())
    L_vmax = float(np.percentile(L_final, config.RENDER_VMAX_PERCENTILE))
    if L_vmax <= L_vmin:
        L_vmax = L_vmin + 1.0

    print(f"\n[过程图片] 输出...")
    print(f"  温度色标: vmin={T_vmin:.1f}K, vmax={T_vmax:.1f}K (固定范围100K)")
    print(f"  辐射范围: {L_vmin:.2f} ~ {L_vmax:.2f} W/(m²·sr)")

    # 获取要输出的步骤列表
    steps = config.PROCESS_IMAGES_STEPS if config.PROCESS_IMAGES_STEPS else ["00", "01", "02", "03", "04"]
    print(f"  输出步骤: {steps}")

    def should_output(step_num):
        """检查是否应该输出该步骤。"""
        return step_num in steps

    # ── 00: 无材质模型图 ──
    if should_output("00"):
        print(f"  无材质模型 → 00_solid.png")
        visualize.clear_scene_materials()
        # 创建简单灰色材质
        mat_gray = bpy.data.materials.new("Process_Gray")
        mat_gray.use_nodes = True
        nodes = mat_gray.node_tree.nodes
        nodes.clear()
        diff = nodes.new("ShaderNodeBsdfDiffuse")
        diff.inputs["Color"].default_value = (0.6, 0.6, 0.6, 1.0)
        out = nodes.new("ShaderNodeOutputMaterial")
        mat_gray.node_tree.links.new(diff.outputs["BSDF"], out.inputs["Surface"])
        merged.data.materials.append(mat_gray)
        visualize.setup_camera(cam_loc, target=target)
        visualize.render_to_file(os.path.join(img_dir, "00_solid.png"))

    # ── 01: mesh线框图 (Freestyle 线框渲染) ──
    if should_output("01"):
        print(f"  mesh线框 → 01_wireframe.png")
        # 使用 Freestyle 渲染清晰的线框
        scene = bpy.context.scene

        # 启用 Freestyle
        scene.render.use_freestyle = True

        # 获取 view layer 的 freestyle 设置
        view_layer = scene.view_layers[0]
        freestyle_settings = view_layer.freestyle_settings
        linesets = freestyle_settings.linesets

        # 创建或获取线条集
        if len(linesets) == 0:
            lineset = linesets.new("MeshLines")
        else:
            lineset = linesets[0]

        # 设置线条类型
        lineset.select_by_edge_types = True
        lineset.select_silhouette = True      # 轮廓线
        lineset.select_border = True          # 边界线
        lineset.select_crease = True          # 折痕线

        # 线条样式
        linestyle = lineset.linestyle
        linestyle.thickness = 2.5  # 较粗的线条
        linestyle.color = (0.15, 0.15, 0.15)  # 深灰色线条

        # 渲染带线框的灰色实体（重新创建材质）
        visualize.clear_scene_materials()
        mat_gray2 = bpy.data.materials.new("Process_Gray_Wire")
        mat_gray2.use_nodes = True
        nodes2 = mat_gray2.node_tree.nodes
        nodes2.clear()
        diff2 = nodes2.new("ShaderNodeBsdfDiffuse")
        diff2.inputs["Color"].default_value = (0.65, 0.65, 0.65, 1.0)
        out2 = nodes2.new("ShaderNodeOutputMaterial")
        mat_gray2.node_tree.links.new(diff2.outputs["BSDF"], out2.inputs["Surface"])
        merged.data.materials.append(mat_gray2)
        visualize.setup_camera(cam_loc, target=target)
        visualize.render_to_file(os.path.join(img_dir, "01_wireframe.png"))

        # 关闭 Freestyle
        scene.render.use_freestyle = False

    # ── 02: 扩散后温度图 + 颜色条 ──
    if should_output("02"):
        print(f"  扩散温度 → 02_temperature_diff.png (含颜色条)")
    visualize.clear_scene_materials()
    visualize.assign_value_material(
        merged, merged.data, T_diffusion,
        attr_name="Temperature",
        color_mode="thermal",
        vmin=T_vmin, vmax=T_vmax,
        mat_name="IR_Temperature_Diff",
    )
    visualize.setup_camera(cam_loc, target=target)
    temp_path = os.path.join(img_dir, "_temp_diff.png")
    visualize.render_to_file(temp_path)

    try:
        composite_with_colorbar(temp_path, img_dir, "thermal", T_vmin, T_vmax, "02_temperature_diff.png")
        os.remove(temp_path)
    except Exception as e:
        print(f"  [WARN] 合成失败: {e}")
        import shutil
        shutil.move(temp_path, os.path.join(img_dir, "02_temperature_diff.png"))

    # ── 03: 氨动加热后温度图 + 颜色条 ──
    if should_output("03"):
        print(f"  氨动温度 → 03_temperature_aero.png (含颜色条)")
        visualize.clear_scene_materials()
        visualize.assign_value_material(
            merged, merged.data, T_aero,
            attr_name="Temperature",
            color_mode="thermal",
            vmin=T_vmin, vmax=T_vmax,  # 共用同一色域
            mat_name="IR_Temperature_Aero",
        )
        visualize.setup_camera(cam_loc, target=target)
        temp_path = os.path.join(img_dir, "_temp_aero.png")
        visualize.render_to_file(temp_path)

        try:
            composite_with_colorbar(temp_path, img_dir, "thermal", T_vmin, T_vmax, "03_temperature_aero.png")
            os.remove(temp_path)
        except Exception as e:
            print(f"  [WARN] 合成失败: {e}")
            import shutil
            shutil.move(temp_path, os.path.join(img_dir, "03_temperature_aero.png"))

    # ── 04: 辐射分布图 (无颜色条) ──
    if should_output("04"):
        print(f"  辐射分布 → 04_radiance.png")
        visualize.clear_scene_materials()
        visualize.assign_value_material(
            merged, merged.data, L_final,
            attr_name="Radiance",
            color_mode="bw",
            vmin=L_vmin, vmax=L_vmax,
            mat_name="IR_Radiance",
        )
        visualize.setup_camera(cam_loc, target=target)
        visualize.render_to_file(os.path.join(img_dir, "04_radiance.png"))


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

    # 4. 孤立面修复（在统计之前，确保修复后数据反映在最终输出）
    T_init_diff = config.T_AIRCRAFT_INIT
    T_init_aero = config.T_AIRCRAFT_INIT + config.T_AIRCRAFT_INIT * 0.16 * config.MACH_NUMBER ** 2

    T_diffusion_fixed, n1 = fix_isolated_faces(T_diffusion, centers, T_init_diff)
    T_aero_fixed, n2 = fix_isolated_faces(T_aero, centers, T_init_aero)

    # 更新 T 为修复后的值（气动加热后）
    T = T_aero_fixed.copy()

    # 5. 统计（使用修复后的数据）
    source_faces, _ = identify_heat_sources(centers, engine_mask, exhaust_positions)
    print_statistics(T, L, engine_mask, source_faces)

    # 6. 过程图片（使用修复后的数据）
    if config.PROCESS_IMAGES_ENABLED:
        render_process_images(merged, T_diffusion_fixed, T_aero_fixed, L)

    # 7. 材质 + 保存（使用修复后的数据）
    apply_material_and_save(
        merged, merged_full, L, T, engine_mask, is_decimated,
        aircraft, all_engines, engines_left, engines_right
    )

    print(f"\n完成，耗时: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()