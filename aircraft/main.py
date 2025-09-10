import math

import bpy
import numpy as np
import os
import sys

# 导入模块
script_dir = os.path.dirname(bpy.data.filepath)
if script_dir not in sys.path:
    sys.path.append(script_dir)
from bl_IR import config
from bl_IR import material
from bl_IR import radiation
from bl_IR import location
from bl_IR import camera

# 获取发动机位置
pos_L, pos_R = location.get_engines_location()


# -----------------------
# 主要方法,应用红外材质
# -----------------------
def apply_ir_material(obj_name, engine_heat_delta):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        print(f"跳过: 找不到网格对象 '{obj_name}' 或类型不是 MESH")
        return

    mesh = obj.data
    radiation_values = []
    T_inf_K = config.ambient_temp_C + 273.15

    # 遍历顶点
    for v in mesh.vertices:
        v_world = np.array(obj.matrix_world @ v.co)
        # ---------- 自身辐射 ----------
        E_self = radiation.calculate(T_inf_K)  # 物体自身热辐射

        # ---------- 太阳项 ----------
        E_sun = 0.0
        if config.CONSIDER_SUN:
            cos_theta = radiation.sun_rad(v)  # 假设返回 0~1
            I_sun = 1360.0 * 0.7  # 简化：地球大气透过率 0.7
            E_sun = config.emissivity * I_sun * cos_theta

        # ---------- 气动加热 ----------
        E_aero = 0.0
        if config.CONSIDER_AERO:
            T_recover = radiation.recovery_temperature(T_inf_K)
            E_aero = radiation.calculate(T_recover) - radiation.calculate(T_inf_K)
            # 航空热力学中 恢复温度（adiabatic wall temperature）

        # ---------- 发动机喷流 ----------
        E_jet = 0.0
        if config.CONSIDER_AERO:
            dTL = radiation.engine_influence(v_world, pos_L, heat=200, decay=0.7)
            dTR = radiation.engine_influence(v_world, pos_R, heat=200, decay=0.7)
            dT_jet = max(dTL, dTR)
            E_jet = radiation.calculate(T_inf_K + dT_jet) - radiation.calculate(T_inf_K)

        # ---------- 总辐射 ----------
        E_total = E_self + E_sun + E_aero + E_jet

        # CFD 修正
        T_cfd = radiation.cfd_analysis(obj_name, T_inf_K, v_world, pos_L, pos_R)
        E_total = max(E_total, radiation.calculate(T_cfd))

        # ---------- 大气修正 (考虑距离) ----------
        if config.USE_ATMOS_CORR:
            T_bg_K = config.T_BACKGROUND_C + 273.15
            E_bg = radiation.calculate(T_bg_K)

            # 计算距离 (传感器位置 - 当前点位置)
            cam_loc = np.array(config.CAMERA_POS)  # 需要在 config 里定义
            R = np.linalg.norm(cam_loc - v_world)

            tau_R = radiation.atmospheric_transmittance(R, kappa=config.KAPPA)

            # 几何扩散项
            geom_factor = 1.0 / (1.0 + 0.001*R*R)

            rad = tau_R * E_total * geom_factor + (1.0 - tau_R) * E_bg
        else:
            rad = E_total

        # ---------- 传感器噪声 ----------
        if config.CONSIDER_NOISE:
            noise_sigma = config.NOISE_LEVEL * rad  # 例如 5% 噪声
            rad += np.random.normal(0, noise_sigma)

        radiation_values.append(rad)

    # 控制台输出范围
    min_rad = min(radiation_values)
    max_rad = max(radiation_values)
    print(f"[{obj_name}] 辐射强度范围: {min_rad:.3e} 到 {max_rad:.3e} W/m²")

    # -----------------------
    # 创建材质
    # -----------------------
    material.assign(obj, mesh, radiation_values)
    print(f"[{obj_name}] 红外热成像材质已应用")


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

camera.render_ir_image("//ir_render.png", cam_location=config.CAMERA_POS,
                       cam_rotation=(math.radians(60), 0, math.radians(30)))
