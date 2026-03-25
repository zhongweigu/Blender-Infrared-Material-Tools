import os
import sys

import bpy

script_dir = os.path.dirname(bpy.data.filepath)
if script_dir not in sys.path:
    sys.path.append(script_dir)
from bl_IR import config
from bl_IR import radiation

GLOBAL_MIN = 2.0e6
GLOBAL_MAX = 3.5e6

def assign(obj, mesh, radiation_values, mode=config.OUTPUT_MODE):
    # -----------------------
    # 创建材质
    # -----------------------
    mat = bpy.data.materials.new(f"IR_Emission_{config.obj_names}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = config.emission_strength * 2.0
    attr_rad = nodes.new("ShaderNodeAttribute")
    attr_rad.attribute_name = "Radiation"

    map_range = nodes.new("ShaderNodeMapRange")
    min = radiation.calculate(config.ambient_temp_C + 273.15)       # 以大气温度为基准设定材质表现
    max = radiation.calculate(config.ambient_temp_C + 200 + 273.15)

    map_range.inputs['From Min'].default_value = min
    map_range.inputs['From Max'].default_value = max
    map_range.inputs['To Min'].default_value = 0.0
    map_range.inputs['To Max'].default_value = 1.0

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.interpolation = 'EASE'

    if mode == 0:
        # 彩色红外材质
        color_ramp.color_ramp.elements[0].color = (0, 0, 1, 1)  # 蓝
        mid = color_ramp.color_ramp.elements.new(0.5)
        mid.color = (1, 1, 0, 1)  # 黄
        color_ramp.color_ramp.elements[1].color = (1, 0, 0, 1)  # 红
    elif mode == 1:
        # 黑白材质
        color_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)  # 黑
        color_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)  # 白

    # -----------------------
    # 节点连接
    # -----------------------
    links.new(attr_rad.outputs["Fac"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    # -----------------------
    # 写入顶点辐射
    # -----------------------
    if "Radiation" not in mesh.attributes:
        mesh.attributes.new(name="Radiation", type='FLOAT', domain='POINT')
    for i, rad in enumerate(radiation_values):
        mesh.attributes["Radiation"].data[i].value = rad

    # -----------------------
    # 分配材质
    # -----------------------
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
