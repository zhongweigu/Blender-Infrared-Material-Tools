import bpy
import math
import numpy as np

# -----------------------
# 配置参数
# -----------------------
class Config:
    ambient_temp_C = -50.0
    solar_delta = 25.0
    aero_delta = 10.0
    emissivity = 0.85
    sigma = 5.670374419e-8

config = Config()

# 物体与温度加成
obj_names = {
    "Aircraft": 0.0,   # 主机体无直接加成
    "Engin_L": 140.0,  # 左发动机固定加成
    "Engin_R": 140.0   # 右发动机固定加成
}

# 方向向量
sun_dir = np.array([0, -1, 0])
sun_dir = sun_dir / np.linalg.norm(sun_dir)
forward_dir = np.array([0, 1, 0])

# 获取发动机位置
def get_obj_position(name):
    obj = bpy.data.objects.get(name)
    return np.array(obj.location) if obj else np.zeros(3)

pos_L = get_obj_position("Engin_L")
pos_R = get_obj_position("Engin_R")
print(f"Engin positions: {pos_L}, {pos_R}")

# -----------------------
# 模块化函数
# -----------------------
def calc_radiation(temp_C):
    """温度转红外辐射"""
    temp_K = temp_C + 273.15
    return config.emissivity * config.sigma * (temp_K ** 4)

def engine_influence(v_co, eng_pos, heat=200.0, decay=1.0):
    """指数衰减发动机热影响"""
    v_co = np.array(v_co)
    d = np.linalg.norm(v_co - eng_pos)
    return heat * math.exp(-decay * d)

# -----------------------
# 应用红外材质
# -----------------------
def apply_ir_material(obj_name, engine_heat_delta):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        print(f"跳过: 找不到网格对象 '{obj_name}' 或类型不是 MESH")
        return

    mesh = obj.data
    temps_K = []
    radiation_values = []

    # 遍历顶点
    for v in mesh.vertices:
        normal = np.array(v.normal)
        normal = normal / np.linalg.norm(normal)

        sun_dot = max(0.0, np.dot(normal, -sun_dir))
        solar_term = config.solar_delta * sun_dot

        wind_dot = max(0.0, np.dot(normal, forward_dir))
        aero_term = config.aero_delta * wind_dot

        v_world = np.array(obj.matrix_world @ v.co)

        if obj_name == "Aircraft":
            # 环境 + 太阳 + 气动 + 发动机影响
            influence_L = engine_influence(v_world, pos_L, heat=200, decay=0.7)
            influence_R = engine_influence(v_world, pos_R, heat=200, decay=0.7)
            temp_C = config.ambient_temp_C + solar_term + aero_term + influence_L + influence_R
        else:
            # 发动机: 环境 + 太阳 + 气动 + 固定发动机加成
            temp_C = config.ambient_temp_C + solar_term + aero_term + engine_heat_delta

        temps_K.append(temp_C + 273.15)
        radiation_values.append(calc_radiation(temp_C))

    # 范围
    min_rad = min(radiation_values)
    max_rad = max(radiation_values)
    print(f"[{obj_name}] 辐射强度范围: {min_rad:.3e} 到 {max_rad:.3e} W/m²")

    # -----------------------
    # 创建材质
    # -----------------------
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
    # 使用固定范围而不是 min_rad/max_rad
    map_range.inputs['From Min'].default_value = calc_radiation(config.ambient_temp_C)
    map_range.inputs['From Max'].default_value = calc_radiation(config.ambient_temp_C + 200)
    map_range.inputs['To Min'].default_value = 0.0
    map_range.inputs['To Max'].default_value = 1.0

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.interpolation = 'EASE'
    color_ramp.color_ramp.elements[0].color = (0, 0, 1, 1)  # 蓝
    mid = color_ramp.color_ramp.elements.new(0.5)
    mid.color = (1, 1, 0, 1)  # 黄
    color_ramp.color_ramp.elements[1].color = (1, 0, 0, 1)  # 红

    links.new(attr_rad.outputs["Fac"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    # 添加Radiation属性
    if "Radiation" not in mesh.attributes:
        mesh.attributes.new(name="Radiation", type='FLOAT', domain='POINT')
    for i, rad in enumerate(radiation_values):
        mesh.attributes["Radiation"].data[i].value = rad

    # 分配材质
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    print(f"[{obj_name}] 红外热成像材质已应用")

# -----------------------
# 批量处理
# -----------------------
for name in ["Aircraft", "Engin_L", "Engin_R"]:
    obj = bpy.data.objects.get(name)
    if obj and obj.type == 'MESH':
        obj.data.materials.clear()
        print(f"{name} 材质已清空")

for name, eng_delta in obj_names.items():
    apply_ir_material(name, eng_delta)
