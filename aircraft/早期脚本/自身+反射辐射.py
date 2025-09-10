import bpy
from mathutils import Vector
from math import exp, cos, sin, radians, sqrt, pi, log

# 物理常量
σ = 5.67e-8      # Stefan-Boltzmann常数 (W/m²K⁴)
c1 = 3.7418e-16  # 第一辐射常数 (W·m²)
c2 = 1.4388e-2   # 第二辐射常数 (m·K)

def calc_self_radiation(ε0, T, λ_min, λ_max):
    """计算自身辐射强度(W/m²)"""
    # 使用简化公式
    λ_avg = (λ_min + λ_max) / 2
    # 普朗克定律近似
    return ε0 * σ * T**4 * (c2 / (λ_avg * T)) * exp(c2 / (λ_avg * T)) / (exp(c2 / (λ_avg * T)) - 1)**2

def calc_reflected_radiation():
    """计算环境反射辐射强度(W/m²)"""
    # 大气参数
    P = 0.7         # 大气透过率
    n = 180         # 一年中的天数
    I0 = 1353       # 太阳常数
    ρ_E = 0.7       # 地球反射系数
    f_Ei = 0.7      # 地球辐射角系数
    
    # 太阳直接辐射
    ξ = 1 + 0.034 * cos(2*pi*n/365)
    m = 1/sin(radians(45))
    I_d = ξ * I0 * P**m
    
    # 散射辐射
    P_safe = max(P, 0.01)
    I_sc = 0.5 * I0 * sin(radians(45)) * (1 - P**m) / (1 - 1.4*log(P_safe)) * cos(radians(22.5))**2
    
    # 天空散射辐射
    ε_sky = 0.58 + 0.061*sqrt(10.0)  # 固定水蒸气10.0
    I_sky = ε_sky * σ * 288**4
    
    # 地球辐射
    I_e = (1 - ρ_E) * I0 / 4 * f_Ei
    
    return I_d + I_sc + I_sky + I_e

def temperature_to_radiation(T):
    """根据温度计算辐射强度(W/m²)"""
    ε0 = 0.65  # 表面发射率
    λ_min = 8 if T < 400 else 3
    λ_max = 12 if T < 400 else 5
    return calc_self_radiation(ε0, T, λ_min, λ_max) + calc_reflected_radiation()

def create_temperature_material(obj):
    """创建温度可视化材质 (完全避免删除操作)"""
    mat_name = "Aircraft_Temperature_Material"
    
    # 使用新材质或创建
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
    
    # 获取节点树
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # 查找或创建必要的节点
    attr_node = None
    color_ramp = None
    emission = None
    output = None
    
    # 尝试查找现有节点
    for node in nodes:
        if node.type == 'ATTRIBUTE':
            attr_node = node
        elif node.type == 'VALTORGB':
            color_ramp = node
        elif node.type == 'EMISSION':
            emission = node
        elif node.type == 'OUTPUT_MATERIAL':
            output = node
    
    # 创建缺失的节点
    if not attr_node:
        attr_node = nodes.new('ShaderNodeAttribute')
        attr_node.attribute_name = "TempColor"
        attr_node.location = (0, 300)
    
    if not color_ramp:
        color_ramp = nodes.new('ShaderNodeValToRGB')
        color_ramp.location = (200, 300)
        color_ramp.color_ramp.interpolation = 'EASE'
    
    if not emission:
        emission = nodes.new('ShaderNodeEmission')
        emission.location = (400, 300)
        emission.inputs['Strength'].default_value = 5.0
    
    if not output:
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (600, 300)
    
    # 设置颜色渐变点 (不删除现有元素，直接覆盖)
    # 创建冷色到暖色的渐变
    colors = [
        (0.0, (0.05, 0.2, 1.0, 1.0)),   # 深蓝 (低温)
        (0.15, (0.2, 0.6, 1.0, 1.0)),   # 浅蓝
        (0.3, (0.0, 1.0, 1.0, 1.0)),    # 青色
        (0.5, (0.5, 1.0, 0.0, 1.0)),    # 黄绿
        (0.7, (1.0, 1.0, 0.0, 1.0)),    # 明黄
        (0.85, (1.0, 0.5, 0.0, 1.0)),   # 橙色
        (1.0, (1.0, 0.0, 0.0, 1.0))     # 红色 (高温)
    ]
    
    # 确保有足够的元素
    while len(color_ramp.color_ramp.elements) < len(colors):
        color_ramp.color_ramp.elements.new(0.0)
    
    # 更新现有元素
    for i, (pos, color) in enumerate(colors):
        if i < len(color_ramp.color_ramp.elements):
            elem = color_ramp.color_ramp.elements[i]
            elem.position = pos
            elem.color = color
    
    # 删除多余元素
    while len(color_ramp.color_ramp.elements) > len(colors):
        color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[-1])
    
    # 连接节点
    if not any(link for link in links if link.from_node == attr_node and link.to_node == color_ramp):
        links.new(attr_node.outputs['Color'], color_ramp.inputs['Fac'])
    
    if not any(link for link in links if link.from_node == color_ramp and link.to_node == emission):
        links.new(color_ramp.outputs['Color'], emission.inputs['Color'])
    
    if not any(link for link in links if link.from_node == emission and link.to_node == output):
        links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # 应用材质到对象
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def generate_temperature_distribution():
    """生成飞机表面温度分布可视化(简化版)"""
    try:
        obj = bpy.context.active_object
        if not obj or obj.type != 'MESH':
            raise Exception("请选择一个网格物体")
        
        mesh = obj.data
        if len(mesh.vertices) == 0:
            raise Exception("网格没有顶点数据")
        
        # 计算世界空间中的Y轴范围
        min_y = float('inf')
        max_y = float('-inf')
        for vert in mesh.vertices:
            global_pos = obj.matrix_world @ vert.co
            if global_pos.y < min_y: min_y = global_pos.y
            if global_pos.y > max_y: max_y = global_pos.y
        
        y_range = max(max_y - min_y, 0.001)
        
        # 创建颜色属性
        attr_name = "TempColor"
        if attr_name not in mesh.attributes:
            attr = mesh.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')
        else:
            attr = mesh.attributes[attr_name]
        
        # 计算并设置温度颜色
        for i, vert in enumerate(mesh.vertices):
            global_pos = obj.matrix_world @ vert.co
            progress = (global_pos.y - min_y) / y_range
            T = 300 + 400 * progress  # 300K到700K线性变化
            
            # 简化温度可视化 - 直接映射温度到颜色
            normalized_T = min(max((T - 300) / 400, 0.0), 1.0)
            
            # 温度颜色映射(蓝到红)
            if normalized_T < 0.3:
                # 蓝色区域
                r = 0.0
                g = normalized_T / 0.3 * 0.6
                b = 1.0 - normalized_T / 0.3 * 0.2
            elif normalized_T < 0.7:
                # 绿-黄区域
                segment = (normalized_T - 0.3) / 0.4
                r = segment
                g = 1.0 - segment * 0.5
                b = 0.0
            else:
                # 橙-红区域
                segment = (normalized_T - 0.7) / 0.3
                r = 1.0
                g = 1.0 - segment * 1.0
                b = 0.0
            
            attr.data[i].color = (r, g, b, 1.0)
        
        # 创建温度可视化材质
        create_temperature_material(obj)
        
        # 配置渲染设置
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.view_settings.view_transform = 'Standard'
        bpy.context.scene.view_settings.look = 'Medium High Contrast'
        
        print("温度可视化完成! 温度沿Y轴从300K(蓝色)到700K(红色)渐变")
        
        # 设置简单世界环境
        if bpy.context.scene.world is None:
            world = bpy.data.worlds.new("SimpleWorld")
            bpy.context.scene.world = world
            world.use_nodes = False
            world.color = (0.05, 0.05, 0.05)  # 深灰色背景
        
        return True
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# 执行主函数
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

# 生成可视化
success = generate_temperature_distribution()

if success:
    print("温度可视化已成功创建，请按F12渲染查看效果!")
else:
    print("温度可视化创建失败")