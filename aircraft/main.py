import math

import bpy
import numpy as np
import os
import sys

# 导入模块
script_dir = r"D:\codes\MTIR-Blender-InfraRed-Material-Tools\aircraft"
if script_dir not in sys.path:
    sys.path.append(script_dir)
from bl_IR import config
from bl_IR import material
from bl_IR import radiation
from bl_IR import location
from bl_IR import camera

# 获取发动机位置
pos_L, pos_R = location.get_engines_location()
print("Pos_L:", pos_L)
print("Pos_R:", pos_R)

# -----------------------
# 主要方法,应用红外材质
# -----------------------
def apply_ir_material(obj_name, engine_heat_delta):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        print(f"跳过: 找不到网格对象 '{obj_name}' 或类型不是 MESH")
        return

    mesh = obj.data
    T_inf_K = config.ambient_temp_C + 273.15
    cam_loc = np.array(config.CAMERA_POS)

    # ====== 调试容器 ======
    stats = {
        "R": [],
        "E_self": [],
        "E_sun": [],
        "E_aero": [],
        "E_jet": [],
        "E_total_raw": [],
        "E_cfd": [],
        "tau": [],
        "geom": [],
        "final": []
    }

    for v in mesh.vertices:
        v_world = np.array(obj.matrix_world @ v.co)

        # ---------- 自身 ----------
        E_self = radiation.calculate(T_inf_K)

        # ---------- 太阳 ----------
        E_sun = 0.0
        if config.CONSIDER_SUN:
            cos_theta = radiation.sun_rad(v)
            I_sun = 1360.0 * 0.7
            E_sun = config.emissivity * I_sun * cos_theta

        # ---------- 气动 ----------
        E_aero = 0.0
        if config.CONSIDER_AERO:
            T_recover = radiation.recovery_temperature(T_inf_K)
            E_aero = radiation.calculate(T_recover) - radiation.calculate(T_inf_K)

        # ---------- 发动机 ----------
        E_jet = 0.0
        if config.CONSIDER_AERO:
            dTL = radiation.engine_influence(v_world, pos_L, heat=200, decay=0.7)
            dTR = radiation.engine_influence(v_world, pos_R, heat=200, decay=0.7)
            dT_jet = max(dTL, dTR)
            E_jet = radiation.calculate(T_inf_K + dT_jet) - radiation.calculate(T_inf_K)

        # ---------- 总 ----------
        E_total = E_self + E_sun + E_aero + E_jet

        # ---------- CFD ----------
        T_cfd = radiation.cfd_analysis(obj_name, T_inf_K, v_world, pos_L, pos_R)
        E_cfd = radiation.calculate(T_cfd)
        E_total = max(E_total, E_cfd)

        # ---------- 距离 ----------
        R = np.linalg.norm(cam_loc - v_world)

        # ---------- 大气 ----------
        if config.USE_ATMOS_CORR:
            tau_R = radiation.atmospheric_transmittance(R, kappa=config.KAPPA)
            geom_factor = 1.0 / (1.0 + 0.001 * R * R)

            T_bg_K = config.T_BACKGROUND_C + 273.15
            E_bg = radiation.calculate(T_bg_K)

            rad = tau_R * E_total * geom_factor + (1 - tau_R) * E_bg
        else:
            tau_R = 1.0
            geom_factor = 1.0
            rad = E_total

        # ---------- 噪声 ----------
        if config.CONSIDER_NOISE:
            noise_sigma = config.NOISE_LEVEL * rad
            rad += np.random.normal(0, noise_sigma)

        # ====== 收集数据 ======
        stats["R"].append(R)
        stats["E_self"].append(E_self)
        stats["E_sun"].append(E_sun)
        stats["E_aero"].append(E_aero)
        stats["E_jet"].append(E_jet)
        stats["E_total_raw"].append(E_self + E_sun + E_aero + E_jet)
        stats["E_cfd"].append(E_cfd)
        stats["tau"].append(tau_R)
        stats["geom"].append(geom_factor)
        stats["final"].append(rad)

    # ====== 打印统计 ======
    def log_stat(name, arr):
        arr = np.array(arr)
        print(f"{name:12s} | min={arr.min():.3e} max={arr.max():.3e} mean={arr.mean():.3e}")

    print(f"\n===== DEBUG: {obj_name} =====")
    log_stat("R", stats["R"])
    log_stat("E_self", stats["E_self"])
    log_stat("E_sun", stats["E_sun"])
    log_stat("E_aero", stats["E_aero"])
    log_stat("E_jet", stats["E_jet"])
    log_stat("E_total", stats["E_total_raw"])
    log_stat("E_cfd", stats["E_cfd"])
    log_stat("tau", stats["tau"])
    log_stat("geom", stats["geom"])
    log_stat("FINAL", stats["final"])

    # ====== 应用材质 ======
    material.assign(obj, mesh, stats["final"])



# -----------------------
# 批量处理
# -----------------------
for name in ["Aircraft", "Engin_L", "Engin_R"]:
    obj = bpy.data.objects.get(name)
    if obj and obj.type == 'MESH':
        obj.data.materials.clear()
        print(f"{name} 材质已清空")

for name, eng_delta in config.obj_names.items():
    apply_ir_material(name, eng_delta)

# camera.render_ir_image("//ir_render.png", cam_location=config.CAMERA_POS,
#                        cam_rotation=(math.radians(60), 0, math.radians(30)))
