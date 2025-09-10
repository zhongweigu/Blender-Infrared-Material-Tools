import bpy
import math
from mathutils import Vector

# ---------- 对象与基础参数 ----------
obj_names = {
    "Aircraft": 0.0,
    "Engin_L": 200.0,
    "Engin_R": 200.0
}

ambient_temp_C = -50.0
solar_delta = 25.0
aero_delta = 10.0
emissivity = 0.85
sigma = 5.670374419e-8

# 太阳与前向
sun_dir = Vector((0, -1.0, -0.2)).normalized()
forward_dir = Vector((0, 1, 0))

# ---------- 解析近似 CFD 场参数 ----------
JET_CENTERLINE_DT0 = 500.0      # 发动机喷流中心线最大升温（K）
JET_Y_DECAY = 15.0              # 沿 y 的指数衰减长度（m）
JET_RADIAL_SIGMA = 2.5          # 径向高斯标准差（m）
JET_MAX_Y_LENGTH = 30.0         # 喷流影响最大长度（m）

# 边界层/气动加热（恢复温度）
GAMMA = 1.4
PR = 0.71
MACH = 0.8
RECOVERY_FACTOR = PR ** (1.0/3.0)  # ~0.88
def recovery_temperature(T_inf_K):
    return T_inf_K * (1.0 + RECOVERY_FACTOR * (GAMMA - 1.0) / 2.0 * MACH**2)

# ---------- 大气修正 ----------
USE_ATMOS_CORR = True
TAU = 0.85
T_BACKGROUND_C = 15.0

# ---------- 发动机中心 ----------
def get_engine_centers():
    centers = []
    for ename in ["Engin_L", "Engin_R"]:
        o = bpy.data.objects.get(ename)
        if o and o.type == 'MESH':
            centers.append(o.matrix_world.translation.copy())
    return centers

ENGINE_CENTERS = get_engine_centers()

# ---------- 主流程 ----------
def apply_ir_material(obj_name, engine_heat_delta):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        print(f"跳过: 找不到网格对象 '{obj_name}' 或类型不是 MESH")
        return

    mesh = obj.data
    radiation_values = []

    # 环境与恢复温度
    T_inf_K = ambient_temp_C + 273.15
    T_recover_K = recovery_temperature(T_inf_K)

    mw = obj.matrix_world
    for v in mesh.vertices:
        nrm = v.normal.normalized()
        sun_dot = max(0.0, nrm.dot(-sun_dir))
        solar_term = solar_delta * sun_dot
        wind_dot = max(0.0, nrm.dot(forward_dir))
        aero_term = aero_delta * wind_dot

        # ---- 原经验温度 ----
        T_exp_C = ambient_temp_C + solar_term + aero_term + engine_heat_delta
        T_exp_K = T_exp_C + 273.15

        # ---- 解析 CFD 温度场 ----
        Pw = mw @ v.co
        jet_increase = 0.0
        for ec in ENGINE_CENTERS:
            dy = 14 + (ec.y - Pw.y)
            if 0.0 <= dy <= JET_MAX_Y_LENGTH:
                radial = math.hypot(Pw.x - ec.x, Pw.z - ec.z)
                axial_decay = math.exp(-dy / JET_Y_DECAY)
                radial_decay = math.exp(-(radial**2) / (2.0 * JET_RADIAL_SIGMA**2))
                jet_increase = max(jet_increase, JET_CENTERLINE_DT0 * axial_decay * radial_decay)

        if obj_name == "Aircraft":
            T_exp_K = max(T_exp_K, T_recover_K)

        T_mix_K = max(T_inf_K + jet_increase, T_exp_K)

        # ---- 辐射计算 (σT^4) ----
        E_emit = emissivity * sigma * (T_mix_K ** 4)
        if USE_ATMOS_CORR:
            T_bg_K = T_BACKGROUND_C + 273.15
            E_bg = emissivity * sigma * (T_bg_K ** 4)
            rad = TAU * E_emit + (1.0 - TAU) * E_bg
        else:
            rad = E_emit

        radiation_values.append(rad)

    # ---- 材质节点 ----
    min_rad = min(radiation_values)
    max_rad = max(radiation_values)
    print(f"[{obj_name}] 辐射强度范围: {min_rad:.3e} 到 {max_rad:.3e}")

    mat = bpy.data.materials.new(f"IR_Emission_{obj_name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    attr_rad = nodes.new("ShaderNodeAttribute")
    attr_rad.attribute_name = "Radiation"

    map_range = nodes.new("ShaderNodeMapRange")
    map_range.inputs['From Min'].default_value = min_rad
    map_range.inputs['From Max'].default_value = max_rad
    map_range.inputs['To Min'].default_value = 0.1
    map_range.inputs['To Max'].default_value = 1.0

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].color = (0, 0, 1, 1)
    color_ramp.color_ramp.elements[1].color = (1, 0, 0, 1)

    links.new(attr_rad.outputs["Fac"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    if "Radiation" not in mesh.attributes:
        mesh.attributes.new(name="Radiation", type='FLOAT', domain='POINT')
    for i, rad in enumerate(radiation_values):
        mesh.attributes["Radiation"].data[i].value = rad

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    print(f"[{obj_name}] 红外热成像材质已应用 (analytic)")

# ---------- 执行 ----------
for name in ["Aircraft", "Engin_L", "Engin_R"]:
    obj = bpy.data.objects.get(name)
    if obj and obj.type == 'MESH':
        obj.data.materials.clear()
        print(f"{name} 材质已清空")

for name, eng_delta in obj_names.items():
    apply_ir_material(name, eng_delta)
