import math
import numpy as np
from new_pipeline import config


def solve_face_temperature(T_o, L_N, A_j, T_amb=None, emissivity=None,
                           k_struct=None, A_struct=None, q_i=None, q_o=None,
                           tol=None):
    """Solve for steady-state face temperature T_s using bisection.

    Equation from pipeline.md:
        F(T) = (T_o - T) / R_N  +  q_o + q_i  -  ε·σ₀·(T⁴ - T_amb⁴)·A_j  =  0

    where R_N = L_N / (k_struct * A_struct)

    Args:
        T_o: exhaust/heat source temperature (K)
        L_N: distance from exhaust to face center (m)
        A_j: face area (m²)
        T_amb: ambient radiation sink temperature (K)
        emissivity: surface IR emissivity ε
        k_struct: thermal conductivity of connecting structure (W/(m·K))
        A_struct: effective cross-sectional area of connecting structure (m²)
        q_i: incident radiation term (W)
        q_o: engine heat power per face (W), from calibration
        tol: convergence tolerance on |F(T)| (W)

    Returns:
        T_s: steady-state face temperature (K)
    """
    if T_amb is None:
        T_amb = config.T_AMB
    if emissivity is None:
        emissivity = config.EMISSIVITY
    if k_struct is None:
        k_struct = config.K_STRUCTURE
    if A_struct is None:
        A_struct = config.A_STRUCTURE
    if q_i is None:
        q_i = config.Q_I
    if q_o is None:
        q_o = config.Q_O if config.Q_O is not None else 0.0
    if tol is None:
        tol = config.HEAT_SOURCE_TOL

    sigma = config.SIGMA
    eps_sigma = emissivity * sigma

    # Thermal resistance
    if L_N < 1e-9:
        L_N = 1e-9
    R_N = L_N / (k_struct * A_struct)

    T_amb4 = T_amb ** 4

    def F(T):
        conduction = (T_o - T) / R_N
        radiation = eps_sigma * (T ** 4 - T_amb4) * A_j
        return conduction + q_o + q_i - radiation

    # F is monotonic decreasing: F(T_amb) > 0, F(T_o) < 0
    lo = T_amb
    hi = T_o

    # Guard: if F(lo) <= 0, face radiates more than it receives → T_s = T_amb
    if F(lo) <= 0:
        return T_amb
    # Guard: if F(hi) >= 0, conduction overwhelms radiation → T_s = T_o
    if F(hi) >= 0:
        return T_o

    for _ in range(200):
        mid = (lo + hi) * 0.5
        fmid = F(mid)
        if abs(fmid) < tol:
            return mid
        if fmid > 0:
            lo = mid
        else:
            hi = mid

    return (lo + hi) * 0.5


def solve_all_source_faces(source_faces, centers, exhaust_positions, areas,
                           T_o=None, T_amb=None, emissivity=None,
                           k_struct=None, A_struct=None, q_i=None, q_o=None,
                           tol=None):
    """Solve T_s for all heat source faces, using the nearest exhaust for each.

    Args:
        source_faces: list of face indices
        centers: (N, 3) face centers in world space
        exhaust_positions: list of (3,) exhaust positions in world space
        areas: (N,) face areas

    Returns:
        T_source: dict mapping face_index -> T_s (K)
    """
    if T_o is None:
        T_o = config.T_EXHAUST

    T_source = {}
    exhaust_arr = np.array(exhaust_positions)

    for fi in source_faces:
        center = centers[fi]
        dists = np.linalg.norm(exhaust_arr - center, axis=1)
        nearest_idx = int(np.argmin(dists))
        L_N = float(dists[nearest_idx])

        T_s = solve_face_temperature(
            T_o=T_o, L_N=L_N, A_j=float(areas[fi]),
            T_amb=T_amb, emissivity=emissivity,
            k_struct=k_struct, A_struct=A_struct,
            q_i=q_i, q_o=q_o, tol=tol
        )
        T_source[fi] = T_s

    return T_source
