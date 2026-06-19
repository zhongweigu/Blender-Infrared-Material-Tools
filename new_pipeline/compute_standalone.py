"""Standalone unified-mesh steady-state temperature computation.

Usage: python compute_standalone.py input.npz output.npz

No Blender dependency. Uses numba JIT for Gauss-Seidel acceleration.
Imports core numerics from calibrate_compute.py (same heat-balance formula).

Algorithm:
  1. Load unified mesh CSR + config from input.npz
  2. Auto-select heat source faces (nearest 5% engine faces per exhaust)
  3. Solve T_s via bisection using calibrated Q_O
  4. Gauss-Seidel diffusion on full mesh (heat source faces = Dirichlet BCs)
  5. Aerodynamic heating
  6. T → L_self (Planck spectral radiance)
  7. L_refl (environment reflection: solar + sky + earth) — Eq.(8)-(15)
  8. L_total = L_self + L_refl — Eq.(16)
  9. L_cam = L_total · cos(θ_c) · cos(θ_s) — Eq.(17)
  10. Atmospheric attenuation on L_cam
  11. Save T, L + stats to output.npz
"""

import sys
import os
import time
import numpy as np

# ── path ──
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from new_pipeline.calibrate_compute import (
    compute_G_flat,
    build_weights,
    _gauss_seidel_sweep,
    solve_heat_balance,
    compute_radiance,
    compute_environment_radiance,
    compute_detector_directional,
    apply_atmospheric_attenuation,
)


# ══════════════════════════════════════════════════════════════════════════════
# Gauss-Seidel driver (wrapper around jitted sweep)
# ══════════════════════════════════════════════════════════════════════════════

def run_diffusion(T_init, offsets, nbr_idx, weights, fixed_set,
                  tol, max_iter):
    """Gauss-Seidel diffusion. Returns (T, iterations, max_change)."""
    n = len(T_init)
    T = T_init.copy().astype(np.float64)
    is_fixed = np.zeros(n, dtype=np.int32)
    for f in fixed_set:
        is_fixed[f] = 1

    for iteration in range(1, max_iter + 1):
        max_change = _gauss_seidel_sweep(T, offsets, nbr_idx, weights, is_fixed)
        if max_change < tol:
            return T, iteration, max_change
    return T, max_iter, max_change


# ══════════════════════════════════════════════════════════════════════════════
# Main
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

    # ── Read mesh data ──
    centers = data['centers'].astype(np.float64)
    areas = data['areas'].astype(np.float64)
    offsets = data['offsets']
    indices = data['indices']
    edge_lens = data['edge_lens']
    exhaust_arr = data['exhaust_positions']  # (N_exh, 3) float32
    engine_mask = data['engine_mask']

    n_faces = len(centers)
    print(f"[compute] 面片总数: {n_faces}, 发动机: {engine_mask.sum()}")

    # ── Read config ──
    T_exhaust = float(data['T_EXHAUST'])
    q_o = float(data['Q_O'])
    T_aircraft_init = float(data['T_AIRCRAFT_INIT'])
    T_amb = float(data['T_AMB'])
    emissivity = float(data['EMISSIVITY'])
    k_skin = float(data['K_SKIN'])
    k_struct = float(data['K_STRUCTURE'])
    a_struct = float(data['A_STRUCTURE'])
    skin_thickness = float(data['SKIN_THICKNESS'])
    sigma = float(data['SIGMA'])
    hs_tol = float(data['HEAT_SOURCE_TOL'])
    diffusion_tol = float(data['DIFFUSION_TOL'])
    max_iterations = int(data['MAX_ITERATIONS'])
    q_i = float(data['Q_I'])
    mach = float(data['MACH_NUMBER'])

    # ── 1. Heat source faces ───────────────────────────────────────────
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

    print(f"  热源面片: {len(source_faces)} "
          f"(每尾焰 {n_source_per_exhaust}, 发动机共 {n_engine} 面)")

    # ── 2. Solve T_s for each heat source face ─────────────────────────
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
    print(f"  T_s: [{T_s_vals.min():.0f}, {T_s_vals.max():.0f}] K "
          f"mean={T_s_vals.mean():.0f} K")

    # ── 3. Initialize temperature field ────────────────────────────────
    print(f"\n[compute] 初始化温度场 (蒙皮初始={T_aircraft_init} K)...")
    T = np.full(n_faces, T_aircraft_init, dtype=np.float64)
    for fi, T_s in T_source_dict.items():
        T[fi] = T_s

    # ── 4. Cross-boundary structural bridges ───────────────────────────
    print(f"\n[compute] 添加发动机→蒙皮结构连接桥...")
    eng_idx_all = np.where(engine_mask)[0]
    ac_idx_all = np.where(~engine_mask)[0]
    cross_pairs = []
    if len(eng_idx_all) > 0 and len(ac_idx_all) > 0:
        ac_c = centers[ac_idx_all]
        k_nbr = min(3, len(ac_idx_all))
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(ac_c)
            for ei in eng_idx_all:
                dists, nearby = tree.query(centers[ei], k=k_nbr)
                if k_nbr == 1:
                    dists, nearby = [float(dists)], [int(nearby)]
                for d, ni in zip(dists, nearby):
                    d = float(d)
                    if d < 2.0:
                        cross_pairs.append((int(ei), int(ac_idx_all[ni]), d))
        except ImportError:
            for ei in eng_idx_all:
                dists = np.linalg.norm(ac_c - centers[ei], axis=1)
                order = np.argsort(dists)[:k_nbr]
                for ni in order:
                    d = float(dists[ni])
                    if d < 2.0:
                        cross_pairs.append((int(ei), int(ac_idx_all[ni]), d))

    # Rebuild CSR to include structural edges
    n = len(centers)
    new_counts = [int(offsets[i + 1] - offsets[i]) for i in range(n)]
    for ei, si, _ in cross_pairs:
        new_counts[ei] += 1
        new_counts[si] += 1

    new_offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(new_counts, out=new_offsets[1:])
    new_total = int(new_offsets[-1])
    new_indices = np.zeros(new_total, dtype=np.int32)
    new_edge_lens = np.zeros(new_total, dtype=np.float64)

    pos = [0] * n
    for i in range(n):
        start, end = int(offsets[i]), int(offsets[i + 1])
        for k in range(start, end):
            p = new_offsets[i] + pos[i]
            new_indices[p] = indices[k]
            new_edge_lens[p] = float(edge_lens[k])
            pos[i] += 1
    for ei, si, dist in cross_pairs:
        p = new_offsets[ei] + pos[ei]
        new_indices[p] = si
        new_edge_lens[p] = -dist  # negative = structural edge
        pos[ei] += 1
        p = new_offsets[si] + pos[si]
        new_indices[p] = ei
        new_edge_lens[p] = -dist
        pos[si] += 1

    print(f"  结构桥: {len(cross_pairs)} 对 (共 {len(cross_pairs)*2} 条边)")

    # ── 5. Gauss-Seidel diffusion ──────────────────────────────────────
    print(f"[compute] Gauss-Seidel 扩散 "
          f"(tol={diffusion_tol}, max_iter={max_iterations})...")

    # 算术平均：所有权重相等
    total_edges = int(new_offsets[-1])
    G = np.ones(total_edges, dtype=np.float64)
    w = build_weights(G, new_offsets)

    t_diff = time.time()
    T, iterations, max_change = run_diffusion(
        T, new_offsets, new_indices, w, source_faces,
        tol=diffusion_tol, max_iter=max_iterations
    )
    print(f"  完成: {iterations} 次迭代, {time.time() - t_diff:.1f} s, "
          f"max ΔT = {max_change:.6f} K")
    print(f"  扩散后 T range: [{T.min():.1f}, {T.max():.1f}] K")

    # ── 6. Aerodynamic heating ─────────────────────────────────────────
    delta_aero = T_aircraft_init * 0.16 * mach * mach
    T += delta_aero
    print(f"\n[compute] 气动加热: M={mach}, ΔT = +{delta_aero:.2f} K")
    print(f"  T range: [{T.min():.1f}, {T.max():.1f}] K  mean={T.mean():.1f} K")

    # ── 7. Temperature → Self Radiance ─────────────────────────────────
    lambda_1 = float(data['LAMBDA_1'])
    lambda_2 = float(data['LAMBDA_2'])
    print(f"\n[compute] 温度→自身辐亮度 (波段 {lambda_1*1e6:.0f}-{lambda_2*1e6:.0f} μm)...")
    L_self = compute_radiance(T, emissivity, lambda_1, lambda_2)
    print(f"  L_self range: [{L_self.min():.2f}, {L_self.max():.2f}] W/(m²·sr)  mean={L_self.mean():.2f}")

    # ── 8. Environment reflection radiation ─────────────────────────────
    print(f"\n[compute] 环境反射辐亮度...")
    normals = data['normals'].astype(np.float64)
    env_config = {
        'I0': float(data['I0']),
        'P': float(data['P']),
        'h': float(data['h']),
        'azimuth': float(data['azimuth']),
        'n_day': int(data['n_day']),
        'e': float(data['e']),
        'T_air': float(data['T_air']),
        'f_fi': float(data['f_fi']),
        'alpha_1': float(data['alpha_1']),
        'sigma': float(data['SIGMA']),
    }
    L_refl = compute_environment_radiance(centers, normals, emissivity, env_config)
    print(f"  L_refl range: [{L_refl.min():.2f}, {L_refl.max():.2f}] W/(m²·sr)  mean={L_refl.mean():.2f}")

    L = L_self + L_refl
    print(f"  L_total range: [{L.min():.2f}, {L.max():.2f}] W/(m²·sr)  mean={L.mean():.2f}")

    # ── 9. Detector directional radiation ──────────────────────────────
    print(f"\n[compute] 探测器方向辐射 (17)...")
    det_pos = data['detector_pos']
    has_los = bool(data['has_los'])
    det_los = data['detector_los'].astype(np.float64) if has_los else None
    L = compute_detector_directional(L, centers, normals, det_pos, det_los)
    print(f"  L_cam range: [{L.min():.2f}, {L.max():.2f}] W/(m²·sr)  mean={L.mean():.2f}")

    # ── 10. Atmospheric attenuation ──────────────────────────────────────
    mu_atm = float(data['MU_ATM'])
    print(f"\n[compute] 大气衰减: μ={mu_atm:.2e} m⁻¹, "
          f"探测器=({det_pos[0]:.0f}, {det_pos[1]:.0f}, {det_pos[2]:.0f})")
    L = apply_atmospheric_attenuation(L, centers, det_pos, mu_atm)
    dists = np.linalg.norm(centers - det_pos, axis=1)
    tau_vals = np.exp(-mu_atm * dists)
    print(f"  距离范围: [{dists.min():.0f}, {dists.max():.0f}] m, "
          f"τ range: [{tau_vals.min():.3f}, {tau_vals.max():.3f}]")
    print(f"  L_detected range: [{L.min():.2f}, {L.max():.2f}] W/(m²·sr)")

    # ── 11. Per-region stats ───────────────────────────────────────────
    skin_mask = ~engine_mask
    if skin_mask.sum() > 0:
        skin_T = T[skin_mask]
        print(f"  蒙皮: [{skin_T.min():.0f}, {skin_T.max():.0f}] K  mean={skin_T.mean():.0f} K")
    if engine_mask.sum() > 0:
        eng_T = T[engine_mask]
        print(f"  发动机: [{eng_T.min():.0f}, {eng_T.max():.0f}] K  mean={eng_T.mean():.0f} K")

    # ── 12. Save results ───────────────────────────────────────────────
    print(f"\n[compute] 保存结果: {output_path}")
    np.savez_compressed(output_path,
        T=T.astype(np.float64),
        L=L.astype(np.float64),
        iterations=np.int32(iterations),
        max_change=np.float64(max_change),
    )
    data.close()

    elapsed = time.time() - t0
    print(f"[compute] 总耗时: {elapsed:.1f} s")


if __name__ == '__main__':
    main()
