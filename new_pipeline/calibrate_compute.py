"""
外部校准引擎: 二分搜索 q_o 使发动机表面均温 ≈ 350 K。

用法 (由 calibrate_qo.py 调用):
    python calibrate_compute.py input.npz

纯数值计算, 无 Blender 依赖。使用 numba JIT 加速 Gauss-Seidel。
"""

import sys
import time
import numpy as np

# ── numba ──
try:
    from numba import njit
    HAS_NUMBA = True
    print("[calibrate] numba JIT 已启用")
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        return lambda f: f
    print("[calibrate] numba 未安装, 使用纯 numpy")


# ══════════════════════════════════════════════════════════════════════════════
# 基础数学工具
# ══════════════════════════════════════════════════════════════════════════════

@njit
def _point_dist(a, b):
    d0 = a[0] - b[0]
    d1 = a[1] - b[1]
    d2 = a[2] - b[2]
    return np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)


@njit
def _gauss_seidel_sweep(T, offsets, nbr_idx, weights, is_fixed, decay, T_amb):
    """单次 Gauss-Seidel 扫描, 返回 max|ΔT|.

    T[i]_new = (1-decay) * Σ(w_k * T[nbr_k]) + decay * T_amb
    """
    n = len(T)
    max_change = 0.0
    one_minus_decay = 1.0 - decay
    for i in range(n):
        if is_fixed[i]:
            continue
        start = offsets[i]
        end = offsets[i + 1]
        if start == end:
            continue
        s = 0.0
        for k in range(start, end):
            s += weights[k] * T[nbr_idx[k]]
        s = one_minus_decay * s + decay * T_amb
        change = s - T[i]
        if change < 0.0:
            change = -change
        if change > max_change:
            max_change = change
        T[i] = s
    return max_change


@njit
def _gauss_seidel_sweep_reverse(T, offsets, nbr_idx, weights, is_fixed, decay, T_amb):
    """单次 Gauss-Seidel 反向扫描 (n-1 → 0), 返回 max|ΔT|.

    与前向扫描配合使用以消除扫描方向偏差。
    """
    n = len(T)
    max_change = 0.0
    one_minus_decay = 1.0 - decay
    for i in range(n - 1, -1, -1):
        if is_fixed[i]:
            continue
        start = offsets[i]
        end = offsets[i + 1]
        if start == end:
            continue
        s = 0.0
        for k in range(start, end):
            s += weights[k] * T[nbr_idx[k]]
        s = one_minus_decay * s + decay * T_amb
        change = s - T[i]
        if change < 0.0:
            change = -change
        if change > max_change:
            max_change = change
        T[i] = s
    return max_change


# ══════════════════════════════════════════════════════════════════════════════
# CSR 构建
# ══════════════════════════════════════════════════════════════════════════════

def compute_G_flat(centers, offsets, indices, edge_lens, k, thickness):
    """计算每条邻接边的热导 G = k·t·L_edge / d_ij."""
    n = len(centers)
    total = int(offsets[-1])
    G_flat = np.empty(total, dtype=np.float64)
    kt = k * thickness
    for i in range(n):
        start, end = int(offsets[i]), int(offsets[i + 1])
        ci = centers[i]
        for idx in range(start, end):
            j = indices[idx]
            d = _point_dist(ci, centers[j])
            if d < 1e-9:
                d = 1e-9
            G_flat[idx] = kt * float(edge_lens[idx]) / d
    return G_flat


def build_weights(G_flat, offsets):
    """归一化热导为权重 (每行求和=1)."""
    n = len(offsets) - 1
    weights = np.empty_like(G_flat, dtype=np.float64)
    for i in range(n):
        start, end = int(offsets[i]), int(offsets[i + 1])
        row_sum = G_flat[start:end].sum()
        if row_sum > 1e-20:
            weights[start:end] = G_flat[start:end] / row_sum
        else:
            weights[start:end] = 0.0
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# 温度→辐射转换 (pipeline.md §9, 原文 (7) 式)
# ══════════════════════════════════════════════════════════════════════════════

# 辐射常数 (SI)
_C1 = 3.7418e-16   # 2πhc²  W·m²
_C2 = 1.4388e-2    # hc/k    m·K

def compute_radiance(T, emissivity, lambda_1, lambda_2):
    """每面片波段积分辐亮度 L_self (原文 (6)/(7) 式)."""
    n = len(T)
    eps_over_pi = emissivity / np.pi
    x1 = 1.0 / lambda_1
    x2 = 1.0 / lambda_2

    L = np.empty(n, dtype=np.float64)
    for i in range(n):
        Ti = T[i]
        if Ti <= 0.0:
            L[i] = 0.0
            continue
        a = _C2 / Ti
        inv_a = 1.0 / a
        # φ(x) = exp(-a/x) * [x³ + 3/(a·(x² + 2/(a·(x + 1/a))))]
        e1 = np.exp(-a / x1)
        inner1 = x1 + inv_a
        inner1 = x1 * x1 + 2.0 / (a * inner1)
        phi1 = e1 * (x1 * x1 * x1 + 3.0 / (a * inner1))
        e2 = np.exp(-a / x2)
        inner2 = x2 + inv_a
        inner2 = x2 * x2 + 2.0 / (a * inner2)
        phi2 = e2 * (x2 * x2 * x2 + 3.0 / (a * inner2))
        L[i] = eps_over_pi * _C1 * inv_a * (phi1 - phi2)
    return L


# ══════════════════════════════════════════════════════════════════════════════
# 大气传输衰减 (pipeline.md §10, 原文 (8)/(9) 式)
# ══════════════════════════════════════════════════════════════════════════════

@njit
def apply_atmospheric_attenuation(L, centers, detector_pos, mu):
    """对每个面片施加大气衰减: L_detected = e^(-μ·R) · L_self.

    Args:
        L: (N,) 面片辐亮度数组 W/(m²·sr)
        centers: (N, 3) 面片中心坐标 (m)
        detector_pos: (3,) 探测器世界坐标 (m)
        mu: 大气平均衰减系数 (m⁻¹)

    Returns:
        L_atten: (N,) 衰减后辐亮度数组 W/(m²·sr)
    """
    n = len(L)
    L_atten = np.empty_like(L)
    dp0, dp1, dp2 = detector_pos[0], detector_pos[1], detector_pos[2]
    for i in range(n):
        c = centers[i]
        dx = c[0] - dp0
        dy = c[1] - dp1
        dz = c[2] - dp2
        R = np.sqrt(dx * dx + dy * dy + dz * dz)
        tau = np.exp(-mu * R)
        L_atten[i] = L[i] * tau
    return L_atten


# ══════════════════════════════════════════════════════════════════════════════
# 环境辐射 (pipeline.md §10, 原文 (8)-(15) 式)
# ══════════════════════════════════════════════════════════════════════════════

@njit
def _compute_sun_dir(elevation, azimuth):
    """太阳方向单位向量 (elevation=高度角, azimuth=方位角)."""
    cos_el = np.cos(elevation)
    return (cos_el * np.sin(azimuth),
            cos_el * np.cos(azimuth),
            np.sin(elevation))


def compute_environment_radiance(centers, normals, emissivity, config):
    """每面片环境反射辐亮度 L_refl.

    Args:
        centers: (N, 3) 面片中心 (m)
        normals: (N, 3) 面片单位法向 (世界坐标)
        emissivity: 表面发射率 ε
        config: dict with keys:
            I0, P, h, azimuth, n_day, e, T_air, f_fi, alpha_1, sigma

    Returns:
        L_refl: (N,) 反射辐亮度 W/(m²·sr)
    """
    I0 = float(config['I0'])
    P = float(config['P'])
    h = float(config['h'])
    azimuth = float(config['azimuth'])
    n_day = int(config['n_day'])
    e_wvp = float(config['e'])
    T_air = float(config['T_air'])
    f_fi = float(config['f_fi'])
    alpha_1 = float(config['alpha_1'])
    sigma = float(config['sigma'])

    # 太阳方向
    sx, sy, sz = _compute_sun_dir(h, azimuth)

    # 大气质量 m = 1/sin(h), 限制最小值防止 h→0 时发散
    sin_h = np.sin(h)
    if sin_h < 0.017:
        sin_h = 0.017
    m = 1.0 / sin_h

    # 日地距离修正 ξ
    xi = 1.0 + 0.034 * np.cos(2.0 * np.pi / 365.0 * n_day)

    # 太阳直射辐照度 I_d = ξ·I₀·P^m —— 原文 (9) 式
    I_d = xi * I0 * P ** m

    # 天空辐照度 I_sky = (a + b√e)·σ·T_air⁴ —— 原文 (13) 式
    a, b = 0.58, 0.061
    eps_sky = a + b * np.sqrt(e_wvp)
    I_sky = eps_sky * sigma * (T_air ** 4)

    # 地面辐照度 I_e = α₁·I₀·f_fi —— 原文 (15) 式
    I_e = alpha_1 * I0 * f_fi

    # Berlage 分母
    ln_P = np.log(P)
    denom = 1.0 - 1.4 * ln_P
    if abs(denom) < 1e-9:
        denom = 1e-9

    n = len(centers)
    L_refl = np.empty(n, dtype=np.float64)
    inv_pi = 1.0 / np.pi
    refl = 1.0 - emissivity

    for i in range(n):
        nx, ny, nz = normals[i, 0], normals[i, 1], normals[i, 2]

        # 太阳直射项: I_d·cos(θ_i) —— 原文 (8) 式第一项
        cos_theta_i = nx * sx + ny * sy + nz * sz
        if cos_theta_i < 0.0:
            cos_theta_i = 0.0
        E_sun = I_d * cos_theta_i

        # 太阳散射项: I_sc —— 原文 (12) 式
        # θ = 面片与水平面的夹角, cos²(θ/2) = (1 + |nz|)/2
        cos_half_theta_sq = 0.5 * (1.0 + abs(nz))
        I_sc = 0.5 * I0 * sin_h * (1.0 - P ** m) / denom * cos_half_theta_sq

        # 总入射辐照度 E_i —— 原文 (8) 式
        E_i = E_sun + I_sc + I_sky + I_e

        # 反射辐亮度 L_refl = (1-ε)/π · E_i
        L_refl[i] = refl * inv_pi * E_i

    return L_refl


def compute_detector_directional(L_total, centers, normals,
                                  detector_pos, detector_los=None):
    """探测器方向辐亮度 L_cam = L_total · cos(θ_c) · cos(θ_s) —— 原文 (17) 式.

    Args:
        L_total: (N,) 半球总辐亮度 W/(m²·sr)
        centers: (N, 3) 面片中心 (m)
        normals: (N, 3) 面片单位法向
        detector_pos: (3,) 探测器世界坐标 (m)
        detector_los: (3,) 探测器视线方向单位向量, None=自动指向面片中心均值

    Returns:
        L_cam: (N,) 探测器方向辐亮度 W/(m²·sr)
    """
    det_pos = np.asarray(detector_pos, dtype=np.float64)
    n = len(centers)

    # 面片→探测器方向向量 (未归一化)
    d = det_pos - centers  # (N, 3)
    dists = np.sqrt((d * d).sum(axis=1))  # (N,)
    dists = np.where(dists < 1e-9, 1e-9, dists)
    d_hat = d / dists[:, np.newaxis]  # (N, 3) unit vectors

    # cos(θ_s) = clamp(n_i · d_hat_i, 0, 1)
    cos_theta_s = (normals[:, 0] * d_hat[:, 0] +
                   normals[:, 1] * d_hat[:, 1] +
                   normals[:, 2] * d_hat[:, 2])
    cos_theta_s = np.clip(cos_theta_s, 0.0, 1.0)

    # 探测器视线方向
    if detector_los is None:
        los = d_hat.mean(axis=0)  # 指向面片几何中心均值
        los_norm = np.sqrt((los * los).sum())
        los = los / max(los_norm, 1e-9)
    else:
        los = np.asarray(detector_los, dtype=np.float64)

    # cos(θ_c) = d_hat_i · los
    cos_theta_c = (d_hat[:, 0] * los[0] +
                   d_hat[:, 1] * los[1] +
                   d_hat[:, 2] * los[2])
    cos_theta_c = np.clip(cos_theta_c, 0.0, 1.0)

    return L_total * cos_theta_c * cos_theta_s


def ensure_connectivity(neighbors, edge_lengths, centers, source_faces):
    """Connect faces in components that have no heat source to the nearest
    face in a source-containing component.

    Faces in a connected component without any heat source will never receive
    heat during diffusion — they stay at their initial temperature forever.
    This function finds those faces and bridges them to the source component.

    Args:
        neighbors: list of lists (mutated in place)
        edge_lengths: dict (mutated in place)
        centers: (N, 3) face centers
        source_faces: set of face indices that are heat sources

    Returns:
        n_bridged: number of faces that were bridged
    """
    n = len(neighbors)
    if n == 0:
        return 0

    # BFS from source faces to find all reachable faces
    reachable = np.zeros(n, dtype=bool)
    queue = list(source_faces)
    for s in queue:
        reachable[s] = True
    head = 0
    while head < len(queue):
        fi = queue[head]
        head += 1
        for nj in neighbors[fi]:
            if not reachable[nj]:
                reachable[nj] = True
                queue.append(nj)

    n_reachable = int(reachable.sum())
    if n_reachable == n:
        return 0  # all faces connected

    # Unreachable faces → bridge each to nearest reachable face
    reachable_idx = np.where(reachable)[0]
    unreachable_idx = np.where(~reachable)[0]
    reachable_centers = centers[reachable_idx]

    # Typical edge length for weight normalization
    pos_lens = [abs(v) for v in edge_lengths.values() if v > 0]
    typical_len = float(np.median(pos_lens)) if pos_lens else 1.0

    bridged = 0
    for ui in unreachable_idx:
        ci = centers[ui]
        dists = np.sum((reachable_centers - ci) ** 2, axis=1)
        j = reachable_idx[int(np.argmin(dists))]
        proxy_len = float(np.sqrt(dists.min()))

        neighbors[ui].append(j)
        neighbors[j].append(ui)
        # Negative value = connectivity bridge, magnitude = distance.
        # Weight will be scaled by typical_len / distance.
        edge_lengths[(ui, j)] = -proxy_len
        edge_lengths[(j, ui)] = -proxy_len
        bridged += 1

    print(f"[mesh_graph] 连通性修复: {bridged} 个面片桥接到热源分量 "
          f"(总面片={n}, 可达={n_reachable}, 不可达={n - n_reachable})")
    return bridged


# ══════════════════════════════════════════════════════════════════════════════
# 热平衡方程求解
# ══════════════════════════════════════════════════════════════════════════════

def solve_heat_balance(T_o, L_N, A_j, q_o, T_amb, emissivity,
                       k_struct, a_struct, sigma, q_i=0.0, tol=1e-3):
    """二分法求解热源面片稳态温度 T_s.

    方程:  F(T) = (T_o-T)/R_N + q_o + q_i - εσ₀(T⁴-T_amb⁴)A_j = 0
    其中 R_N = L_N / (k_struct * a_struct)  —— a_struct 是结构件截面积(真实值)
    """
    if L_N < 1e-9:
        L_N = 1e-9
    R_N = L_N / (k_struct * a_struct)
    eps_sigma = emissivity * sigma
    T_amb4 = T_amb ** 4

    def F(T):
        conduction = (T_o - T) / R_N
        radiation = eps_sigma * (T ** 4 - T_amb4) * A_j
        return conduction + q_o + q_i - radiation

    lo, hi = T_amb, T_o

    # F 单调递减: F(lo) 通常 > 0, F(hi) 可能 < 0 或 > 0
    if F(lo) <= 0.0:
        return T_amb  # 辐射散热 ≥ 供热 → 环境温度

    if F(hi) >= 0.0:
        # q_o 太大, 根在 T_o 以上 → 搜索上界
        hi = T_o
        for _ in range(50):
            hi *= 1.5
            if F(hi) <= 0.0 or hi > 10000.0:
                break
        if F(hi) >= 0.0:
            return hi  # 饱和

    for _ in range(300):
        mid = (lo + hi) * 0.5
        fmid = F(mid)
        if abs(fmid) < tol:
            return mid
        if fmid > 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def solve_all_source_faces(source_indices, centers, exhaust_pos, areas,
                           q_o, T_o, T_amb, emissivity, k_struct,
                           a_struct, sigma, q_i=0.0, tol=1e-3):
    """对全部热源面片求解 T_s. 返回 dict: face_idx -> T_s."""
    T_source = {}
    for fi in source_indices:
        L_N = float(np.linalg.norm(centers[fi] - exhaust_pos))
        T_s = solve_heat_balance(
            T_o=T_o, L_N=L_N, A_j=float(areas[fi]), q_o=q_o,
            T_amb=T_amb, emissivity=emissivity,
            k_struct=k_struct, a_struct=a_struct, sigma=sigma, q_i=q_i, tol=tol
        )
        T_source[fi] = T_s
    return T_source


# ══════════════════════════════════════════════════════════════════════════════
# Gauss-Seidel 扩散
# ══════════════════════════════════════════════════════════════════════════════

def run_diffusion(T_init, offsets, nbr_idx, weights, fixed_set,
                  tol, max_iter, decay=0.0, T_amb=280.0):
    """Gauss-Seidel 扩散, 返回 (T, iterations, max_change)."""
    T = T_init.copy().astype(np.float64)
    n = len(T)
    is_fixed = np.zeros(n, dtype=np.int32)
    for f in fixed_set:
        is_fixed[f] = 1

    for iteration in range(1, max_iter + 1):
        max_change = _gauss_seidel_sweep(T, offsets, nbr_idx, weights, is_fixed, decay, T_amb)
        if max_change < tol:
            return T, iteration, max_change
    return T, max_iter, max_change


# ══════════════════════════════════════════════════════════════════════════════
# 单次计算 (给定 q_o)
# ══════════════════════════════════════════════════════════════════════════════

def compute_with_qo(data, q_o, k_skin, skin_thickness,
                    diffusion_tol, max_iterations):
    """用给定的 q_o 值运行一次完整计算, 返回发动机面片均温."""
    centers = data['centers']
    areas = data['areas']
    offsets = data['offsets']
    indices = data['indices']
    edge_lens = data['edge_lens']
    exhaust_pos = data['exhaust_pos']
    engine_mask = data['engine_mask']

    T_o = float(data['T_EXHAUST'])
    T_amb = float(data['T_AMB'])
    emissivity = float(data['EMISSIVITY'])
    k_struct = float(data['K_STRUCTURE'])
    a_struct = float(data['A_STRUCTURE'])
    sigma = float(data['SIGMA'])
    hs_tol = float(data['HEAT_SOURCE_TOL'])
    q_i = float(data['Q_I'])
    decay = float(data.get('DIFFUSION_DECAY', 0.0))

    # 1. 诊断: 打印距离分布
    print(f"\n[calibrate] 尾焰核心: ({exhaust_pos[0]:.4f}, {exhaust_pos[1]:.4f}, {exhaust_pos[2]:.4f})")
    dists_all = np.linalg.norm(centers - exhaust_pos, axis=1)
    dists_eng = dists_all[engine_mask]
    print(f"[calibrate] 全部面片到核心: min={dists_all.min():.4f}, "
          f"median={np.median(dists_all):.3f}, max={dists_all.max():.3f} m")
    print(f"[calibrate] 发动机面片到核心: min={dists_eng.min():.4f}, "
          f"median={np.median(dists_eng):.3f}, max={dists_eng.max():.3f} m")
    for r in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        n_all = int((dists_all <= r).sum())
        n_eng = int((dists_eng <= r).sum())
        print(f"[calibrate]   半径 {r:.2f}m: 全部={n_all} 面, 发动机={n_eng} 面")

    # 热源面片: 自动选取距尾焰核心最近的发动机面片 (5%, 最少5最多200)
    n_engine = engine_mask.sum()
    n_source = max(5, min(200, int(n_engine * 0.05)))
    engine_indices = np.where(engine_mask)[0]
    sorted_eng = engine_indices[np.argsort(dists_all[engine_indices])]
    source_faces = set(sorted_eng[:n_source])
    print(f"  热源面片: 最近 {n_source} 个发动机面片 "
          f"(共 {n_engine} 个发动机面片)")

    if len(source_faces) == 0:
        print(f"  [警告] q_o={q_o:.3f}: 未找到热源面片")
        return 0.0

    # 热平衡求解 T_s
    T_s_dict = solve_all_source_faces(
        list(source_faces), centers, exhaust_pos, areas,
        q_o=q_o, T_o=T_o, T_amb=T_amb, emissivity=emissivity,
        k_struct=k_struct, a_struct=a_struct, sigma=sigma, q_i=q_i, tol=hs_tol
    )

    # 初始化温度场
    n_faces = len(centers)
    T = np.full(n_faces, T_amb, dtype=np.float64)
    for fi, T_s in T_s_dict.items():
        T[fi] = T_s

    # Gauss-Seidel 扩散（算术平均）
    total_edges = int(offsets[-1])
    G = np.ones(total_edges, dtype=np.float64)
    w = build_weights(G, offsets)

    T, iters, change = run_diffusion(
        T, offsets, indices, w, set(source_faces),
        tol=diffusion_tol, max_iter=max_iterations,
        decay=decay, T_amb=T_amb,
    )

    # 发动机区域统计
    engine_T = T[engine_mask]
    if len(engine_T) == 0:
        return 0.0

    eng_mean = float(np.mean(engine_T))
    eng_min = float(np.min(engine_T))
    eng_max = float(np.max(engine_T))

    T_s_vals = np.array(list(T_s_dict.values()))
    print(f"  q_o={q_o:8.3f} W | "
          f"T_s=[{T_s_vals.min():.0f}, {T_s_vals.max():.0f}] K | "
          f"迭代={iters:5d} | "
          f"发动机均温={eng_mean:.2f} K  [{eng_min:.1f}, {eng_max:.1f}]")

    return eng_mean


# ══════════════════════════════════════════════════════════════════════════════
# 二分搜索
# ══════════════════════════════════════════════════════════════════════════════

def run_bisection(data):
    """二分搜索 q_o 使发动机表面均温 → TARGET_ENGINE_T."""
    target = float(data['TARGET_ENGINE_T'])
    k_skin = float(data['K_SKIN'])
    skin_thickness = float(data['SKIN_THICKNESS'])
    diffusion_tol = float(data['DIFFUSION_TOL'])
    max_iterations = int(data['MAX_ITERATIONS'])

    print(f"\n[calibrate] 二分搜索 q_o, 目标: 发动机均温 = {target} K")
    print(f"[calibrate] {'='*60}")

    # 搜索范围: q_o ∈ [0.01, 500.0] W
    q_lo, q_hi = 0.01, 500.0

    # 检查下界
    t_lo = compute_with_qo(data, q_lo, k_skin, skin_thickness,
                           diffusion_tol, max_iterations)
    if t_lo > target:
        print(f"\n[calibrate] q_o={q_lo} 时发动机均温已达 {t_lo:.2f} K > {target} K")
        print(f"[calibrate] 无需 q_o, 传导已足够。q_o ≈ 0")
        return

    if t_lo <= 0:
        print("[calibrate] 错误: 未找到发动机面片或热源面片")
        return

    # 检查上界
    t_hi = compute_with_qo(data, q_hi, k_skin, skin_thickness,
                           diffusion_tol, max_iterations)
    if t_hi < target:
        print(f"\n[calibrate] q_o={q_hi} 时发动机均温仅 {t_hi:.2f} K < {target} K")
        print(f"[calibrate] 需要更大 q_o, 扩展搜索...")
        for _ in range(20):
            q_hi *= 2.0
            t_hi = compute_with_qo(data, q_hi, k_skin, skin_thickness,
                                   diffusion_tol, max_iterations)
            if t_hi >= target:
                break
        if t_hi < target:
            print(f"[calibrate] q_o={q_hi} 仍不足, 可能无法达到目标温度")
            return

    # 二分
    for bisect_iter in range(30):
        q_mid = (q_lo + q_hi) * 0.5
        t_mid = compute_with_qo(data, q_mid, k_skin, skin_thickness,
                                diffusion_tol, max_iterations)

        if t_mid <= 0:
            break

        if abs(t_mid - target) < 0.5:
            q_lo = q_mid
            break

        if t_mid > target:
            q_hi = q_mid
        else:
            q_lo = q_mid

        if q_hi - q_lo < 0.01:
            break

    # 精炼: 在 [q_lo-1, q_lo+1] 范围再做一次精细二分
    print(f"\n[calibrate] {'='*60}")
    print(f"[calibrate] 精炼搜索...")
    q_refine_lo = max(0.01, q_lo - 2.0)
    q_refine_hi = q_lo + 2.0
    for _ in range(15):
        q_mid = (q_refine_lo + q_refine_hi) * 0.5
        t_mid = compute_with_qo(data, q_mid, k_skin, skin_thickness,
                                diffusion_tol, max_iterations)
        if t_mid > target:
            q_refine_hi = q_mid
        else:
            q_refine_lo = q_mid
        if q_refine_hi - q_refine_lo < 0.005:
            break

    q_final = (q_refine_lo + q_refine_hi) * 0.5
    t_final = compute_with_qo(data, q_final, k_skin, skin_thickness,
                              diffusion_tol, max_iterations)

    print(f"\n[calibrate] {'='*60}")
    print(f"[calibrate] ✅ 校准完成")
    print(f"[calibrate]    q_o = {q_final:.4f} W")
    print(f"[calibrate]    发动机均温 = {t_final:.2f} K (目标 {target} K)")
    print(f"[calibrate]    请在 config.py 中设置: Q_O = {q_final:.4f}")
    print(f"[calibrate] {'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) != 2:
        print("用法: python calibrate_compute.py input.npz")
        sys.exit(1)

    input_path = sys.argv[1]
    t0 = time.time()
    print(f"[calibrate] 加载: {input_path}")
    data = np.load(input_path)
    run_bisection(data)
    data.close()
    print(f"\n[calibrate] 总耗时: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
