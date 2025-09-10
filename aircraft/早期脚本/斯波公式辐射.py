import bpy
from mathutils import Vector

# 物体与温度参数
obj_names = {
    "Aircraft": 0.0,         # 主机体无发动机温度加成
    "Engin_L": 200.0,        # 左发动机加成
    "Engin_R": 200.0         # 右发动机加成
}

ambient_temp_C = -50.0
solar_delta = 25.0
aero_delta = 10.0
emissivity = 0.85
sigma = 5.670374419e-8

# 固定方向
sun_dir = Vector((0, -1.0, -0.2)).normalized()  # 太阳方向
forward_dir = Vector((0, 1, 0))                 # 飞机前向

def apply_ir_material(obj_name, engine_heat_delta):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        print(f"跳过: 找不到网格对象 '{obj_name}' 或类型不是 MESH")
        return

    mesh = obj.data

    temps_K = []
    radiation_values = []

    # 逐顶点计算温度和辐射
    for v in mesh.vertices:
        normal = v.normal.normalized()
        sun_dot = max(0.0, normal.dot(-sun_dir))
        solar_term = solar_delta * sun_dot
        wind_dot = max(0.0, normal.dot(forward_dir))
        aero_term = aero_delta * wind_dot
        temp_C = ambient_temp_C + solar_term + aero_term + engine_heat_delta
        temp_K = temp_C + 273.15
        temps_K.append(temp_K)
        rad = emissivity * sigma * (temp_K ** 4)
        radiation_values.append(rad)

    # 范围
    min_rad = min(radiation_values)
    max_rad = max(radiation_values)
    print(f"[{obj_name}] 辐射强度范围: {min_rad:.3e} 到 {max_rad:.3e} W/m²")

    # 创建材质
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
    color_ramp.color_ramp.elements[0].color = (0, 0, 1, 1)  # 冷色
    color_ramp.color_ramp.elements[1].color = (1, 0, 0, 1)  # 热色

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

# 批量处理所有对象

for name in ["Aircraft", "Engin_L", "Engin_R"]:
    obj = bpy.data.objects.get(name)
    if obj and obj.type == 'MESH':
        obj.data.materials.clear()
        print(f"{name} 材质已清空")

for name, eng_delta in obj_names.items():
    apply_ir_material(name, eng_delta)
