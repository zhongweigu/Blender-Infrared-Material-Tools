"""
外部计算脚本: 统一网格稳态温度计算

用法: python compute_standalone.py input.npz output.npz

无 Blender 依赖。使用 numba JIT 加速 Gauss-Seidel。

算法 (pipeline.md §§3-9 + §13.1):
  1. 加载网格 CSR + 配置参数
  2. 【关键】添加跨边界结构连接 (add_cross_boundary_bridges)
  3. 热源面片识别 (最近 5% 发动机面片)
  4. 热平衡求解 T_s (二分法)
  5. Gauss-Seidel 扩散 (算术平均权重)
  6. 气动加热 ΔT = T_amb × 0.16 × M²
  7. 温度→辐亮度 (Planck 经验近似公式7)
  8. 能量衰减 η = τ₀π / [4K_e²(1-β'/β_p)²]
  9. 保存结果

不计算: 环境辐射(§10), 大气衰减(§11), 探测器方向因子(§12)
"""

import sys
import os
import time
import numpy as np

# ── 项目路径 ──
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from new_pipeline.calibrate_compute import (
    build_weights,
    _gauss_seidel_sweep,
    _gauss_seidel_sweep_reverse,
    solve_heat_balance,
    compute_radiance,
    add_cross_boundary_bridges,
)


# ══════════════════════════════════════════════════════════════════════════════
# CSR → 邻接表重建
# ══════════════════════════════════════════════════════════════════════════════

def rebuild_adjacency(offsets, indices, edge_lens):
    """
    从 CSR 结构重建 list of lists 邻接表 + edge_lengths dict。

    Returns:
        neighbors: list of lists
        edge_lengths: dict {(i, j): length}
    """
    n = len(offsets) - 1
    neighbors = []
    edge_lengths = {}

    for i in range(n):
        start, end = int(offsets[i]), int(offsets[i + 1])
        nbrs = []
        for k in range(start, end):
            j = int(indices[k])
            nbrs.append(j)
            if (i, j) not in edge_lengths:
                el = float(edge_lens[k])
                edge_lengths[(i, j)] = el
                edge_lengths[(j, i)] = el
        neighbors.append(nbrs)

    return neighbors, edge_lengths


# ══════════════════════════════════════════════════════════════════════════════
# Gauss-Seidel driver
# ══════════════════════════════════════════════════════════════════════════════

def run_diffusion(T_init, offsets, nbr_idx, weights, fixed_set,
                  tol, max_iter, decay=0.0, T_amb=280.0):
    """Gauss-Seidel 扩散。返回 (T, iterations, max_change)。"""
    n = len(T_init)
    T = T_init.copy().astype(np.float64)
    is_fixed = np.zeros(n, dtype=np.int32)
    for f in fixed_set:
        is_fixed[f] = 1

    for iteration in range(1, max_iter + 1):
        # 正向 + 反向扫描消除方向偏差
        if iteration % 2 == 1:
            c1 = _gauss_seidel_sweep(T, offsets, nbr_idx, weights, is_fixed, decay, T_amb)
            c2 = _gauss_seidel_sweep_reverse(T, offsets, nbr_idx, weights, is_fixed, decay, T_amb)
        else:
            c2 = _gauss_seidel_sweep_reverse(T, offsets, nbr_idx, weights, is_fixed, decay, T_amb)
            c1 = _gauss_seidel_sweep(T, offsets, nbr_idx, weights, is_fixed, decay, T_amb)
        max_change = max(c1, c2)
        if max_change < tol:
            return T, iteration, max_change
    return T, max_iter, max_change


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) != 3:
        print("用法: python compute_standalone.py input.npz output.npz")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    t0 = time.time()
    print(f"[compute] 加载: {input_path}")
    data = np.load(input_path)

    # ── 读取网格数据 ──
    centers = data['centers'].astype(np.float64)
    areas = data['areas'].astype(np.float64)
    offsets = data['offsets']
    indices = data['indices']
    edge_lens = data['edge_lens']
    exhaust_arr = data['exhaust_positions']
    engine_mask = data['engine_mask']

    n_faces = len(centers)
    print(f"[compute] 面片总数: {n_faces}, 发动机: {int(engine_mask.sum())}")

    # ── 读取配置参数 ──
    T_exhaust = float(data['T_EXHAUST'])
    q_o = float(data['Q_O'])
    T_aircraft_init = float(data['T_AIRCRAFT_INIT'])
    T_amb = float(data['T_AMB'])
    emissivity = float(data['EMISSIVITY'])
    k_struct = float(data['K_STRUCTURE'])
    a_struct = float(data['A_STRUCTURE'])
    sigma = float(data['SIGMA'])
    hs_tol = float(data['HEAT_SOURCE_TOL'])
    diffusion_tol = float(data['DIFFUSION_TOL'])
    max_iterations = int(data['MAX_ITERATIONS'])
    diffusion_decay = float(data.get('DIFFUSION_DECAY', 0.0))
    q_i = float(data['Q_I'])
    mach = float(data['MACH_NUMBER'])
    lambda_1 = float(data['LAMBDA_1'])
    lambda_2 = float(data['LAMBDA_2'])
    tau0 = float(data.get('TAU0', 0.85))
    K_e = float(data.get('K_E', 2.0))
    beta_ratio = float(data.get('BETA_RATIO', 0.0))
    cross_max_pairs = int(data.get('CROSS_BOUNDARY_MAX_PAIRS', 5))
    cross_max_dist = float(data.get('CROSS_BOUNDARY_MAX_DISTANCE', 5.0))

    # ── 重建邻接表 ──
    neighbors, edge_lengths = rebuild_adjacency(offsets, indices, edge_lens)

    # ── 【关键】跨边界结构连接 ──
    print(f"\n[compute] 跨边界桥接 (max_pairs={cross_max_pairs}, max_dist={cross_max_dist:.1f} m)...")
    n_bridges = add_cross_boundary_bridges(
        neighbors, edge_lengths, centers, engine_mask,
        max_pairs=cross_max_pairs, max_distance=cross_max_dist
    )
    print(f"  跨边界桥接: {n_bridges} 条边")

    # ── 重新构建 CSR (含桥接边) ──
    n = len(neighbors)
    new_offsets = np.zeros(n + 1, dtype=np.int32)
    counts = [len(nbrs) for nbrs in neighbors]
    new_offsets[1:] = np.cumsum(counts)
    total_edges = new_offsets[-1]

    new_indices = np.empty(total_edges, dtype=np.int32)
    new_edge_lens = np.empty(total_edges, dtype=np.float64)

    for i in range(n):
        start = new_offsets[i]
        nbrs = neighbors[i]
        for k, j in enumerate(nbrs):
            new_indices[start + k] = j
            new_edge_lens[start + k] = edge_lengths.get((i, j), 1.0)

    # ── 1. 热源面片识别 ──
    print(f"\n[compute] 识别热源面片 (T_o={T_exhaust} K, Q_O={q_o:.4f} W)...")

    n_engine = int(engine_mask.sum())
    if n_engine == 0:
        print("[compute] 错误: 无发动机面片")
        data.close()
        sys.exit(1)

    n_source_per_exhaust = max(5, min(200, int(n_engine * 0.05)))
    engine_indices = np.where(engine_mask)[0]

    source_faces = set()
    exhaust_positions = [exhaust_arr[i] for i in range(len(exhaust_arr))]
    for ep in exhaust_positions:
        dists = np.linalg.norm(centers[engine_indices] - ep, axis=1)
        nearest = engine_indices[np.argsort(dists)[:n_source_per_exhaust]]
        source_faces.update(nearest.tolist())

    print(f"  热源面片: {len(source_faces)} (每尾焰 {n_source_per_exhaust})")

    # ── 2. 热平衡求解 T_s ──
    print(f"\n[compute] 求解热源面片 T_s...")
    T_source_dict = {}
    for fi in source_faces:
        c = centers[fi]
        dists = [float(np.linalg.norm(c - ep)) for ep in exhaust_positions]
        nearest_idx = int(np.argmin(dists))
        L_N = dists[nearest_idx]
        T_s = solve_heat_balance(
            T_o=T_exhaust, L_N=L_N, A_j=float(areas[fi]), q_o=q_o,
            T_amb=T_amb, emissivity=emissivity,
            k_struct=k_struct, a_struct=a_struct, sigma=sigma,
            q_i=q_i, tol=hs_tol
        )
        T_source_dict[fi] = T_s

    T_s_vals = np.array(list(T_source_dict.values()))
    print(f"  T_s: [{T_s_vals.min():.0f}, {T_s_vals.max():.0f}] K mean={T_s_vals.mean():.0f} K")

    # ── 3. 初始化温度场 ──
    print(f"\n[compute] 初始化温度场 (蒙皮初始={T_aircraft_init} K)...")
    T = np.full(n_faces, T_aircraft_init, dtype=np.float64)
    for fi, T_s in T_source_dict.items():
        T[fi] = T_s

    # ── 4. Gauss-Seidel 扩散 ──
    print(f"\n[compute] Gauss-Seidel 扩散 (tol={diffusion_tol}, max_iter={max_iterations})...")

    # 算术平均权重: 所有边 G=1
    G = np.ones(total_edges, dtype=np.float64)
    w = build_weights(G, new_offsets)

    t_diff = time.time()
    T, iterations, max_change = run_diffusion(
        T, new_offsets, new_indices, w, source_faces,
        tol=diffusion_tol, max_iter=max_iterations,
        decay=diffusion_decay, T_amb=T_amb,
    )
    print(f"  完成: {iterations} 次迭代, {time.time() - t_diff:.1f} s, max ΔT={max_change:.6f} K")
    print(f"  扩散后 T range: [{T.min():.1f}, {T.max():.1f}] K")

    skin_mask = ~engine_mask
    if skin_mask.sum() > 0:
        skin_T = T[skin_mask]
        print(f"  蒙皮 T: [{skin_T.min():.1f}, {skin_T.max():.1f}] K mean={skin_T.mean():.1f} K")

    T_diffusion = T.copy()

    # ── 5. 氨动加热 ──
    delta_aero = T_aircraft_init * 0.16 * mach * mach
    T += delta_aero
    print(f"\n[compute] 气动加热: M={mach}, ΔT={delta_aero:.2f} K")
    print(f"  T range: [{T.min():.1f}, {T.max():.1f}] K mean={T.mean():.1f} K")

    T_aero = T.copy()

    # ── 6. 温度→辐亮度 (Planck 公式7) ──
    print(f"\n[compute] 温度→辐亮度 (波段 {lambda_1*1e6:.0f}-{lambda_2*1e6:.0f} μm)...")
    L_self = compute_radiance(T, emissivity, lambda_1, lambda_2)
    print(f"  L_self: [{L_self.min():.2f}, {L_self.max():.2f}] W/(m²·sr) mean={L_self.mean():.2f}")

    L_radiance = L_self.copy()

    # ── 7. 能量衰减 (公式4) ──
    denom = 4.0 * K_e * K_e * (1.0 - beta_ratio) ** 2
    if denom < 1e-9:
        denom = 1e-9
    eta = tau0 * np.pi / denom
    L = L_self * eta
    print(f"\n[compute] 能量衰减: τ₀={tau0}, K_e={K_e}, η={eta:.4f}")
    print(f"  L_out: [{L.min():.2f}, {L.max():.2f}] W/(m²·sr)")

    # ── 8. 统计 ──
    if skin_mask.sum() > 0:
        skin_T = T[skin_mask]
        print(f"  蒙皮: [{skin_T.min():.0f}, {skin_T.max():.0f}] K mean={skin_T.mean():.0f} K")
    if engine_mask.sum() > 0:
        eng_T = T[engine_mask]
        print(f"  发动机: [{eng_T.min():.0f}, {eng_T.max():.0f}] K mean={eng_T.mean():.0f} K")

    # ── 9. 保存结果 ──
    print(f"\n[compute] 保存: {output_path}")
    np.savez_compressed(output_path,
        T=T.astype(np.float64),
        L=L.astype(np.float64),
        iterations=np.int32(iterations),
        max_change=np.float64(max_change),
        T_diffusion=T_diffusion.astype(np.float64),
        T_aero=T_aero.astype(np.float64),
        L_radiance=L_radiance.astype(np.float64),
    )
    data.close()

    elapsed = time.time() - t0
    print(f"[compute] 总耗时: {elapsed:.1f} s")


if __name__ == '__main__':
    main()