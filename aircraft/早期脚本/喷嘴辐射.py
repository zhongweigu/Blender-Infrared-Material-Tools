import bpy
import numpy as np
from math import exp
from mathutils import Vector

# 常量定义 (SI单位) - 与机身相同
c1 = 3.7418e-16  # 第一辐射常数 (W·m²)
c2 = 1.4388e-2   # 第二辐射常数 (m·K)
ε_M = 0.9         # 发射率 (0-1)
θ_M = 0           # 观察角度 (0度表示正对表面)
λ1 = 8e-6         # 波长下限 (8μm)
λ2 = 14e-6        # 波长上限 (14μm)
S_M = 1.0         # 表面比例因子

def planck_law(λ, T):
    """计算普朗克辐射定律在给定波长和温度下的辐射强度"""
    exponent = c2 / (λ * T)
    if exponent > 700:  # 防止数值溢出
        return 0
    return c1 / (λ**5 * (exp(exponent) - 1))

def integrate_radiation(T, num_points=100):
    """数值积分计算辐射强度积分项"""
    dλ = (λ2 - λ1) / num_points
    integral = 0.0
    
    for i in range(num_points):
        λ = λ1 + (i + 0.5) * dλ  # 中点积分法
        term = planck_law(λ, T)
        integral += term * dλ
    
    return integral

def calculate_IM(T):
    """计算完整辐射强度 I_M"""
    integral = integrate_radiation(T)
    return (ε_M / np.pi) * integral * S_M * np.cos(θ_M)

def apply_nozzle_thermal_material(obj, max_rad):
    """应用喷嘴专用的热辐射材质，基于辐射强度"""
    # 创建新材质
    mat_name = "NozzleRadiationMaterial"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        # 清除所有现有节点
        nodes.clear()
    else:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
    
    links = mat.node_tree.links
    
    # 1. 辐射强度属性节点
    attr_node = nodes.new('ShaderNodeAttribute')
    attr_node.attribute_name = "Radiation"
    attr_node.location = (-800, 0)
    
    # 2. 映射范围节点 - 喷嘴辐射范围更高
    map_node = nodes.new('ShaderNodeMapRange')
    map_node.location = (-600, 0)
    # 设置动态调整范围，实际值会在主函数中更新
    map_node.inputs['From Min'].default_value = 1e-8
    map_node.inputs['From Max'].default_value = max_rad * 1.1  # 使用传入的最大辐射值
    map_node.inputs['To Min'].default_value = 0.0
    map_node.inputs['To Max'].default_value = 1.0
    
    # 3. 颜色渐变节点 - 基于辐射强度设置颜色
    color_ramp = nodes.new('ShaderNodeValToRGB')
    color_ramp.location = (-400, 0)
    color_ramp.color_ramp.interpolation = 'LINEAR'
    
    # 确保有5个控制点
    if len(color_ramp.color_ramp.elements) < 5:
        # 添加新的控制点
        for _ in range(5 - len(color_ramp.color_ramp.elements)):
            color_ramp.color_ramp.elements.new(0.1)
    
    # 设置控制点位置和颜色 - 喷嘴特有的辐射色彩方案
    # 冷色（低辐射）- 深蓝色
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.3, 1)
    
    # 暖色（中辐射）- 橙红色
    color_ramp.color_ramp.elements[1].position = 0.3
    color_ramp.color_ramp.elements[1].color = (1, 0.4, 0, 1)
    
    # 热色（中高辐射）- 亮红色
    color_ramp.color_ramp.elements[2].position = 0.6
    color_ramp.color_ramp.elements[2].color = (1, 0.1, 0, 1)
    
    # 极热（高辐射）- 橙黄色
    color_ramp.color_ramp.elements[3].position = 0.8
    color_ramp.color_ramp.elements[3].color = (1, 1, 0.5, 1)
    
    # 最热（极高辐射）- 白色
    color_ramp.color_ramp.elements[4].position = 1.0
    color_ramp.color_ramp.elements[4].color = (1, 1, 1, 1)
    
    # 4. 自发光节点
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (-200, 0)
    
    # 5. 透明BSDF节点
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    transparent.location = (-200, 200)
    
    # 6. 混合着色器节点
    mix_shader = nodes.new('ShaderNodeMixShader')
    mix_shader.location = (0, 100)
    
    # 7. 材质输出节点
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (200, 100)
    
    # 连接节点
    links.new(attr_node.outputs["Color"], map_node.inputs["Value"])
    links.new(map_node.outputs["Result"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], emission.inputs["Color"])
    
    # 辐射强度映射值控制发光强度
    links.new(map_node.outputs["Result"], emission.inputs["Strength"])
    
    # 辐射强度映射值控制混合系数
    links.new(map_node.outputs["Result"], mix_shader.inputs["Fac"])
    
    # 连接自发光和透明节点到混合着色器
    links.new(emission.outputs["Emission"], mix_shader.inputs[1])
    links.new(transparent.outputs["BSDF"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])
    
    # 应用材质到喷嘴对象
    if obj.data.materials:
        # 使用现有材质槽
        for slot in obj.material_slots:
            slot.material = mat
    else:
        # 添加新材质槽
        obj.data.materials.append(mat)
    
    return mat

def setup_nozzle_temperature(obj, nozzle_type="typical"):
    """为喷嘴设置独特的温度分布和辐射强度"""
    mesh = obj.data
    vertices = mesh.vertices
    
    # 确保有温度属性
    if "Temperature" not in mesh.attributes:
        mesh.attributes.new(name="Temperature", type='FLOAT', domain='POINT')
    
    # 确保有辐射强度属性
    if "Radiation" not in mesh.attributes:
        mesh.attributes.new(name="Radiation", type='FLOAT', domain='POINT')
    
    # 计算几何中心
    if vertices:
        center = Vector((0, 0, 0))
        for v in vertices:
            center += v.co
        center /= len(vertices)
    else:
        center = Vector((0, 0, 0))
    
    # 获取最大距离（用于归一化）
    max_distance = 0.0
    if vertices:
        max_distance = max((v.co - center).length for v in vertices)
    if max_distance == 0:
        max_distance = 1.0
    
    # 喷嘴温度分布模型
    radiation_values = []  # 存储所有辐射值
    for i, vert in enumerate(vertices):
        # 计算到中心的距离
        distance = (vert.co - center).length
        
        # 不同类型的喷嘴温度分布模式
        if nozzle_type == "typical":
            # 典型喷嘴：中心热，边缘冷却
            max_temp = 2400
            min_temp = 600
            distance_ratio = distance / max_distance
            temp = min_temp + (max_temp - min_temp) * (1 - distance_ratio)
        
        elif nozzle_type == "cooled":
            # 冷却型喷嘴：入口热，出口冷
            # 假设模型轴向为Z轴
            min_z = min(v.co.z for v in vertices)
            max_z = max(v.co.z for v in vertices)
            height_range = max_z - min_z
            if height_range == 0:
                height_range = 1.0
            z_ratio = (vert.co.z - min_z) / height_range
            temp = 500 + 2000 * (1 - z_ratio)  # 入口2500K，出口500K
        
        else:  # rocket
            # 火箭喷嘴：喷口极端高温
            max_temp = 3000
            min_temp = 700
            distance_factor = min(1.0, distance / max_distance)
            temp = min_temp + (max_temp - min_temp) * (1 - distance_factor)**2
        
        # 确保温度在合理范围内
        temp = min(3500, max(500, temp))
        mesh.attributes["Temperature"].data[i].value = temp
        
        # 计算辐射强度并存储
        rad_value = calculate_IM(temp)
        mesh.attributes["Radiation"].data[i].value = rad_value
        radiation_values.append(rad_value)
    
    # 返回最大辐射值
    max_rad = max(radiation_values) if radiation_values else 1e-5
    min_rad = min(radiation_values) if radiation_values else 0
    
    return max_rad, min_rad, vertices

def setup_scene():
    """设置场景参数 - 为喷嘴优化"""
    scene = bpy.context.scene
    
    # 设置渲染引擎为Cycles以获得更好的热辐射效果
    scene.render.engine = 'CYCLES'
    
    # 降低环境光照
    if scene.world and scene.world.node_tree:
        if "Background" in scene.world.node_tree.nodes:
            scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.1
    
    # 增加采样数
    if hasattr(scene, 'cycles') and scene.cycles:
        scene.cycles.samples = 128
    
    # 添加全局辉光效果（通过合成器）
    scene.use_nodes = True
    tree = scene.node_tree
    if tree is None:
        tree = scene.node_tree = bpy.context.scene.node_tree
    
    glare_node = tree.nodes.get("NozzleGlare")
    if not glare_node:
        glare_node = tree.nodes.new('CompositorNodeGlare')
        glare_node.name = "NozzleGlare"
        glare_node.glare_type = 'FOG_GLOW'
        glare_node.threshold = 0.1
        glare_node.size = 8
        
    # 连接到合成
    rl_node = tree.nodes.get('Render Layers')
    comp_node = tree.nodes.get('Composite')
    if rl_node and comp_node:
        tree.links.new(rl_node.outputs[0], glare_node.inputs[0])
        tree.links.new(glare_node.outputs[0], comp_node.inputs[0])

def refresh_viewport():
    """强制刷新3D视图"""
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
                    space.shading.use_scene_world = True
                    space.shading.use_scene_lights = True
    
    bpy.context.view_layer.update()
    bpy.context.scene.frame_current = bpy.context.scene.frame_current

def create_nozzle_report(obj, max_rad, min_rad, vertices):
    """创建喷嘴热分析报告"""
    if vertices:
        # 获取网格数据
        mesh = obj.data
        
        # 获取温度属性
        if "Temperature" not in mesh.attributes:
            print(f"警告: 对象 '{obj.name}' 没有温度属性")
            return
            
        temps = [mesh.attributes["Temperature"].data[i].value for i in range(len(vertices))]
        min_temp = min(temps)
        max_temp = max(temps)
        avg_temp = sum(temps) / len(temps)
        
        # 计算最大辐射强度点
        rads = [mesh.attributes["Radiation"].data[i].value for i in range(len(vertices))]
        max_rad_value = max(rads)
        max_rad_temp = temps[rads.index(max_rad_value)]
        max_rad_index = rads.index(max_rad_value)
        
        print(f"喷嘴 '{obj.name}' 热辐射分析报告:")
        print(f" - 温度范围: {min_temp:.1f}K 到 {max_temp:.1f}K")
        print(f" - 辐射强度范围: {min_rad:.3e} W/sr 到 {max_rad:.3e} W/sr")
        print(f" - 最大辐射强度点: {max_rad_value:.3e} W/sr ({max_rad_temp:.1f}K)")
        print(f" - 位置: {vertices[max_rad_index].co}")
    else:
        print("警告: 喷嘴网格没有顶点数据")

def main(nozzle_objects):
    """处理所有喷嘴对象"""
    print("\n===== 开始设置喷嘴热成像（辐射强度模式）=====")
    
    # 设置场景
    setup_scene()
    
    # 处理每个喷嘴对象
    for obj in nozzle_objects:
        if not obj:
            print("警告: 遇到无效对象, 跳过")
            continue
            
        # 确保对象是网格类型
        if obj.type != 'MESH':
            print(f"转换 {obj.name} 为网格对象...")
            try:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.convert(target='MESH')
                print(f"成功将 {obj.name} 转换为网格对象")
            except Exception as e:
                print(f"无法转换 {obj.name}: {e}")
                continue
        
        print(f"\n处理喷嘴: {obj.name}")
        
        # 设置喷嘴特定温度分布并计算辐射强度
        max_rad, min_rad, vertices = setup_nozzle_temperature(obj, nozzle_type="rocket")
        
        # 应用基于辐射强度的喷嘴材质
        mat = apply_nozzle_thermal_material(obj, max_rad)
        
        # 更新材质中的映射范围
        if mat and mat.use_nodes:
            nodes = mat.node_tree.nodes
            for node in nodes:
                if node.type == 'MAP_RANGE':
                    # 设置实际辐射强度范围
                    node.inputs['From Min'].default_value = min_rad
                    node.inputs['From Max'].default_value = max_rad
        
        # 生成热辐射分析报告
        create_nozzle_report(obj, max_rad, min_rad, vertices)
    
    # 刷新视图
    refresh_viewport()
    
    # 设置完成
    print("\n所有喷嘴热成像设置完成（辐射强度模式）!")
    print("提示: 按F12键进行最终渲染查看热辐射效果")

# 执行主程序
if __name__ == "__main__":
    # 清空控制台
    print("\n" * 5)
    
    # 选择喷嘴对象
    # 方法1：选择当前活动对象
    # nozzle_objects = [bpy.context.active_object] if bpy.context.active_object else []
    
    # 方法2：选择所有名称包含"nozzle"的对象
    nozzle_objects = [obj for obj in bpy.context.scene.objects if "nozzle" in obj.name.lower()]
    
    # 如果没有找到，尝试使用选定对象
    if not nozzle_objects:
        nozzle_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    
    if nozzle_objects:
        main(nozzle_objects)
    else:
        print("错误：未找到喷嘴对象！")
        print("请在场景中选择或创建喷嘴对象")