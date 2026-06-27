"""
校准脚本: 二分搜索 q_o 使发动机表面均温 ≈ 350 K (巡航状态)。

用法:
    blender -b <模型.blend> --python ./new_pipeline/calibrate_qo.py

原理:
    尾焰核心 T_o=900K 是唯一热源。
    距核心 ≤ 0.04m 的面片(热源面片)用热平衡方程求解 T_s:
        F(T) = (T_o-T)/R_N + q_o - εσ₀(T⁴-T_amb⁴)A_j = 0
    其余面片用 Gauss-Seidel 扩散。
    二分搜索 q_o 使原发动机区域面片均温 ≈ 350 K。
"""

import os
import sys
import time
import subprocess
import tempfile

import bpy
import numpy as np


# ── Path setup ──────────────────────────────────────────────────────────────
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
    raise RuntimeError("无法定位项目根目录")

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from new_pipeline import config
from new_pipeline import mesh_graph


# ══════════════════════════════════════════════════════════════════════════════
# Blender 端: 合并 mesh 并导出
# ══════════════════════════════════════════════════════════════════════════════

def _find_exhaust_position_merged(engines):
    """从发动机对象列表中找到尾焰核心位置 (Y 最小 10% 顶点均值)."""
    all_verts = []
    for obj in engines:
        if obj is not None:
            for v in obj.data.vertices:
                all_verts.append(obj.matrix_world @ v.co)
    if not all_verts:
        return np.array([0, 0, 0], dtype=np.float32)
    verts = np.array(all_verts)
    y_min = verts[:, 1].min()
    y_max = verts[:, 1].max()
    threshold = y_min + 0.1 * (y_max - y_min)
    rear = verts[verts[:, 1] <= threshold]
    return np.mean(rear, axis=0).astype(np.float32)


def export_merged_mesh(output_path, aircraft, all_engines):
    """合并所有对象为一个 mesh, 导出数据到 .npz 供外部校准脚本使用.

    输出 keys:
        centers, areas, offsets, indices, edge_lens  — 合并 mesh 面片数据
        exhaust_pos     — 尾焰核心世界坐标 (3,)
        engine_mask     — bool 数组, True 表示原发动机面片
        heat_source_r   — 热源面片判定半径 (m)
        <config params> — 与 io_mesh.export_mesh_data 相同
    """
    import bmesh

    # 记录各对象面片数 (合并前)
    ac_n = len(aircraft.data.polygons)
    eng_face_counts = [len(eng.data.polygons) for eng in all_engines]

    # 尾焰核心位置 (合并前记录)
    exhaust_pos = _find_exhaust_position_merged(all_engines)
    print(f"[校准导出] 尾焰核心: ({exhaust_pos[0]:.3f}, {exhaust_pos[1]:.3f}, {exhaust_pos[2]:.3f})")

    # 复制并合并
    bpy.ops.object.select_all(action='DESELECT')

    def _copy_obj(obj):
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.duplicate()
        dup = bpy.context.active_object
        obj.select_set(False)
        return dup

    ac_dup = _copy_obj(aircraft)
    eng_copies = [_copy_obj(eng) for eng in all_engines]

    # 合并: aircraft 为 active, 其他 selected
    bpy.ops.object.select_all(action='DESELECT')
    ac_dup.select_set(True)
    bpy.context.view_layer.objects.active = ac_dup
    for d in eng_copies:
        d.select_set(True)
    bpy.ops.object.join()

    merged = ac_dup
    merged.name = "Calibration_Merged"
    print(f"[校准导出] 合并完成: {len(merged.data.polygons)} 面 "
          f"(ac={ac_n}, eng={sum(eng_face_counts)})")

    # 构建 engine_mask
    engine_mask = np.zeros(len(merged.data.polygons), dtype=bool)
    offset = ac_n
    for fc in eng_face_counts:
        if fc > 0:
            engine_mask[offset:offset + fc] = True
            offset += fc
    print(f"[校准导出] 发动机面片: {engine_mask.sum()}")

    # 提取面片数据
    centers, areas, _ = mesh_graph.get_mesh_data(merged)
    neighbors, edge_lengths = mesh_graph.build_face_adjacency(merged)

    n = len(neighbors)
    counts = np.array([len(nbrs) for nbrs in neighbors], dtype=np.int32)
    total = int(counts.sum())
    offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    indices = np.zeros(total, dtype=np.int32)
    edge_lens = np.zeros(total, dtype=np.float32)
    for i, nbrs in enumerate(neighbors):
        start = int(offsets[i])
        for k, j in enumerate(nbrs):
            indices[start + k] = j
            edge_lens[start + k] = edge_lengths.get((i, j), 0.0)

    np.savez_compressed(output_path,
        centers=centers.astype(np.float32),
        areas=areas.astype(np.float32),
        offsets=offsets,
        indices=indices,
        edge_lens=edge_lens.astype(np.float32),
        exhaust_pos=exhaust_pos,
        engine_mask=engine_mask,
        T_EXHAUST=np.float32(config.T_EXHAUST),
        T_AMB=np.float32(config.T_AMB),
        EMISSIVITY=np.float32(config.EMISSIVITY),
        K_SKIN=np.float32(config.K_SKIN),
        K_STRUCTURE=np.float32(config.K_STRUCTURE),
        A_STRUCTURE=np.float32(config.A_STRUCTURE),
        SKIN_THICKNESS=np.float32(config.SKIN_THICKNESS),
        SIGMA=np.float32(config.SIGMA),
        HEAT_SOURCE_TOL=np.float32(config.HEAT_SOURCE_TOL),
        DIFFUSION_TOL=np.float32(config.DIFFUSION_TOL),
        MAX_ITERATIONS=np.int32(config.MAX_ITERATIONS),
        Q_I=np.float32(config.Q_I),
        TARGET_ENGINE_T=np.float32(350.0),
    )
    print(f"[校准导出] 已保存: {output_path}")

    # 清理合并后的临时对象
    bpy.data.objects.remove(merged, do_unlink=True)
    print("[校准导出] 已清理临时合并对象")


# ══════════════════════════════════════════════════════════════════════════════
# 外部计算: 二分搜索 q_o
# ══════════════════════════════════════════════════════════════════════════════

def _find_venv_python():
    venv_dir = os.path.join(_project_root, ".venv")
    if not os.path.isdir(venv_dir):
        return None
    for sub in ("Scripts", "bin"):
        for name in ("python.exe", "python", "python3"):
            p = os.path.join(venv_dir, sub, name)
            if os.path.isfile(p):
                return p
    return None


def main():
    t_start = time.time()

    # ── 查找对象 ──
    aircraft = mesh_graph.find_aircraft()
    engines_left, engines_right = mesh_graph.find_all_engines()
    all_engines = engines_left + engines_right

    if aircraft is None:
        print("[错误] 未找到蒙皮网格对象")
        return
    if not all_engines:
        print("[错误] 未找到发动机对象 (用于定位尾焰核心)")
        return

    print(f"对象: Aircraft='{aircraft.name}'")
    for i, eng in enumerate(all_engines):
        print(f"       Engine[{i}]='{eng.name}'")

    # ── 模型缩放 (所有操作之前) ──
    if config.MODEL_SCALE != 1.0:
        print(f"\n[校准] 应用模型缩放: ×{config.MODEL_SCALE}")
        for obj in [aircraft] + all_engines:
            if obj is None:
                continue
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            obj.scale *= config.MODEL_SCALE
            bpy.ops.object.transform_apply(scale=True)
            obj.select_set(False)
        print("[校准] 缩放完成，scale 已应用")

    # ── 导出合并 mesh ──
    tmpdir = tempfile.gettempdir()
    input_npz = os.path.join(tmpdir, "_blir_calibrate_input.npz")

    print("\n[校准] 合并 mesh 并导出...")
    export_merged_mesh(input_npz, aircraft, all_engines)

    # ── 查找 .venv Python ──
    python_exe = _find_venv_python()
    standalone = os.path.join(_project_root, "new_pipeline",
                              "calibrate_compute.py")
    if python_exe is None:
        print("[校准] .venv 未找到，回退到 Blender 内计算 (较慢)...")
        _run_in_blender(input_npz)
        return
    if not os.path.isfile(standalone):
        print(f"[校准] 外部脚本未找到: {standalone}")
        print("[校准] 回退到 Blender 内计算...")
        _run_in_blender(input_npz)
        return

    # ── 调用外部校准引擎 ──
    print(f"\n[校准] 启动外部校准引擎: {python_exe}")
    t_run = time.time()
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        result = subprocess.run(
            [python_exe, standalone, input_npz],
            capture_output=True, text=True, timeout=600,
            encoding='utf-8', errors='replace', env=env,
        )
    except FileNotFoundError:
        print(f"[校准] Python 未找到: {python_exe}")
        _run_in_blender(input_npz)
        return
    except subprocess.TimeoutExpired:
        print("[校准] 超时 (600s)")
        return

    elapsed = time.time() - t_run
    print(f"[校准] 外部引擎耗时: {elapsed:.1f} s")

    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")
    if result.stderr:
        print(f"  [stderr]: {result.stderr[:500]}")

    if result.returncode != 0:
        print(f"[校准] 外部进程失败 (code={result.returncode})")
        return

    print(f"\n[校准] 总耗时: {time.time() - t_start:.1f} s")
    print("请将输出的 q_o 值填入 config.py 的 Q_O 参数。")

    # 清理
    try:
        os.remove(input_npz)
    except OSError:
        pass


def _run_in_blender(input_npz):
    """Blender 内回退: 纯 Python 二分搜索 (较慢, 但可用)."""
    print("[校准] Blender 内回退模式...")
    from new_pipeline import diffusion as _diff_mod
    from new_pipeline import calibrate_compute as _cc

    data = np.load(input_npz)
    _cc.run_bisection(data)
    data.close()


if __name__ == "__main__":
    main()
