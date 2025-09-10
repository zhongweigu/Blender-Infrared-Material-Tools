import math, sys, os, bpy
import numpy as np

script_dir = os.path.dirname(bpy.data.filepath)
if script_dir not in sys.path:
    sys.path.append(script_dir)
from bl_IR import config


def calculate(temp_K: float):
    if config.METHOD == "stefan_boltzmann":
        return stefan_boltzmann(temp_K)
    elif config.METHOD == "plank_law":
        return plank_law(temp_K)
    return 0


def stefan_boltzmann(temp_K: float, emissivity: float = config.emissivity, sigma: float = config.sigma) -> float:
    """
    计算辐射功率，W/m²
    R = ε * σ * T^4
    """
    return emissivity * sigma * (temp_K ** 4)


def plank_law(temp_K: float, emissivity: float = config.emissivity, h: float = config.h, c: float = config.c,
              wavelength: float = config.wavelength, kB: float = config.kB) -> float:
    B_lambda = (2 * h * c ** 2) / (wavelength ** 5) / (math.exp((h * c) / (wavelength * kB * temp_K)) - 1)
    return emissivity * B_lambda


def engine_influence(v_co, eng_pos, heat, decay):
    """指数衰减发动机热影响"""
    v_co = np.array(v_co)
    d = np.linalg.norm(v_co - eng_pos)
    return heat * math.exp(-decay * d)


def sun_rad(v):
    normal = np.array(v.normal)
    normal = normal / np.linalg.norm(normal)

    sun_dot = max(0.0, np.dot(normal, -config.sun_dir))
    solar_term = config.solar_delta * sun_dot
    return solar_term


def aero_rad(v):
    normal = np.array(v.normal)
    normal = normal / np.linalg.norm(normal)

    wind_dot = max(0.0, np.dot(normal, config.forward_dir))
    aero_term = config.aero_delta * wind_dot
    return aero_term


def cfd_analysis(obj_name, temp_K, v_world, pos_L, pos_R, jet_dirs=config.JET_DIRS,
                 JET_MAX_AXIAL_LENGTH: float = config.JET_MAX_Y_LENGTH,
                 JET_AXIAL_DECAY: float = config.JET_Y_DECAY,
                 JET_RADIAL_SIGMA: float = config.JET_RADIAL_SIGMA,
                 JET_CENTERLINE_DT0: float = config.JET_CENTERLINE_DT0,
                 BACKFLOW_SCALE: float = 0.2,
                 BACKFLOW_DECAY_FACTOR: float = 0.8):
    if not config.CONSIDER_CFD:
        return temp_K

    jet_increase = 0.0
    for ec, jet_dir in zip([pos_L, pos_R], jet_dirs):
        rel_vec = (v_world[0] - ec[0], v_world[1] - ec[1], v_world[2] - ec[2])

        axial_dist = rel_vec[0] * jet_dir[0] + rel_vec[1] * jet_dir[1] + rel_vec[2] * jet_dir[2]

        # 计算径向距离
        rel_len2 = rel_vec[0] ** 2 + rel_vec[1] ** 2 + rel_vec[2] ** 2
        radial2 = rel_len2 - axial_dist ** 2
        radial_dist = math.sqrt(max(0.0, radial2))

        if 0.0 <= axial_dist <= JET_MAX_AXIAL_LENGTH:
            # 正方向喷流
            axial_decay = math.exp(-axial_dist / JET_AXIAL_DECAY)
            radial_decay = math.exp(-(radial_dist ** 2) / (2.0 * JET_RADIAL_SIGMA ** 2))
            jet_increase = max(jet_increase, JET_CENTERLINE_DT0 * axial_decay * radial_decay)

        elif axial_dist < 0.0:
            # 反方向扰动（弱）
            axial_decay = math.exp(-abs(axial_dist) / (JET_AXIAL_DECAY * BACKFLOW_DECAY_FACTOR))
            radial_decay = math.exp(-(radial_dist ** 2) / (2.0 * JET_RADIAL_SIGMA ** 2))
            jet_increase = max(jet_increase, JET_CENTERLINE_DT0 * BACKFLOW_SCALE * axial_decay * radial_decay)

    if obj_name == "Aircraft":
        temp_K = max(temp_K, recovery_temperature(config.ambient_temp_C + 273.15))

    return max(config.ambient_temp_C + 273.15 + jet_increase, temp_K)


def recovery_temperature(T_inf_K):
    return T_inf_K * (1.0 + config.RECOVERY_FACTOR * (config.GAMMA - 1.0) / 2.0 * config.MACH ** 2)


def atmospheric_transmittance(R, kappa=0.01):
    return math.exp(-kappa * R)
