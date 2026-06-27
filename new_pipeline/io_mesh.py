"""Blender-side unified mesh data extraction → .npz for standalone computation.

Exports the already-merged-and-scaled mesh (CSR adjacency, engine mask,
exhaust positions) and all config parameters needed by compute_standalone.py.
"""

import os
import numpy as np
import bpy
from new_pipeline import config
from new_pipeline import mesh_graph


def export_unified_mesh(output_path, merged_obj, exhaust_positions, engine_mask):
    """Export merged mesh data for external compute_standalone.py.

    Args:
        output_path: path to output .npz file
        merged_obj: Blender mesh object (already merged + scaled to real size)
        exhaust_positions: list of (3,) arrays in world space (real scale)
        engine_mask: bool array (N_faces,) — True for engine-origin faces
    """
    centers, areas, face_verts = mesh_graph.get_mesh_data(merged_obj)
    neighbors, edge_lengths = mesh_graph.build_face_adjacency(merged_obj)

    n = len(neighbors)

    # Compute face normals from world-space vertices
    normals = np.empty((n, 3), dtype=np.float32)
    for i, fv in enumerate(face_verts):
        v0, v1, v2 = np.asarray(fv[0]), np.asarray(fv[1]), np.asarray(fv[2])
        nrm = np.cross(v1 - v0, v2 - v0)
        nlen = np.linalg.norm(nrm)
        if nlen > 1e-9:
            nrm /= nlen
        normals[i] = nrm
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

    exh_arr = np.array(exhaust_positions, dtype=np.float32)

    np.savez_compressed(output_path,
        centers=centers.astype(np.float32),
        areas=areas.astype(np.float32),
        normals=normals,
        offsets=offsets,
        indices=indices,
        edge_lens=edge_lens,
        exhaust_positions=exh_arr,
        engine_mask=engine_mask,

        # Config parameters
        T_EXHAUST=np.float32(config.T_EXHAUST),
        Q_O=np.float32(config.Q_O),
        T_AIRCRAFT_INIT=np.float32(config.T_AIRCRAFT_INIT),
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
        DIFFUSION_DECAY=np.float32(config.DIFFUSION_DECAY),
        Q_I=np.float32(config.Q_I),
        MACH_NUMBER=np.float32(config.MACH_NUMBER),
        LAMBDA_1=np.float32(config.LAMBDA_1),
        LAMBDA_2=np.float32(config.LAMBDA_2),
        MU_ATM=np.float32(config.MU_ATM),
        detector_pos=np.array(config.DETECTOR_POS, dtype=np.float32),
        detector_los=(np.array(config.DETECTOR_LOS, dtype=np.float32)
                      if config.DETECTOR_LOS is not None
                      else np.zeros(3, dtype=np.float32)),
        has_los=np.int32(0 if config.DETECTOR_LOS is None else 1),
        env_radiation_enabled=np.int32(config.ENV_RADIATION_ENABLED),
        # Environment radiation
        I0=np.float32(config.SUN_CONSTANT),
        P=np.float32(config.ATM_TRANSPARENCY),
        h=np.float32(config.SUN_ELEVATION),
        azimuth=np.float32(config.SUN_AZIMUTH),
        n_day=np.int32(config.DAY_NUMBER),
        e=np.float32(config.WATER_VAPOR_PRESSURE),
        T_air=np.float32(config.AIR_TEMPERATURE),
        f_fi=np.float32(config.EARTH_ANGLE_COEFF),
        alpha_1=np.float32(config.ALPHA_1),
        # Energy degradation
        TAU0=np.float32(config.TAU0),
        K_E=np.float32(config.K_E),
        BETA_RATIO=np.float32(config.BETA_RATIO),
    )
    print(f"[io_mesh] 已导出统一网格: {output_path} ({n} 面)")


def import_results(results_path):
    """Load temperature results from standalone computation.

    Returns:
        dict with keys: T, iterations, max_change.
        T is a float64 array (N_faces,). Returns None if file not found.
    """
    if not os.path.isfile(results_path):
        print(f"[io_mesh] 结果文件不存在: {results_path}")
        return None

    data = np.load(results_path)
    results = {
        'T': data['T'].astype(np.float64),
        'L': data['L'].astype(np.float64),
        'iterations': int(data['iterations']),
        'max_change': float(data['max_change']),
        # 中间过程数据
        'T_diffusion': data['T_diffusion'].astype(np.float64),
        'T_aero': data['T_aero'].astype(np.float64),
        'L_radiance': data['L_radiance'].astype(np.float64),
    }
    data.close()
    print(f"[io_mesh] 已读回结果: {results_path}")
    return results
