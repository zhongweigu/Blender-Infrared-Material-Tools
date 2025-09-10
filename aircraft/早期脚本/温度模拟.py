import bpy
import numpy as np
from math import exp

# 常量定义 (SI单位)
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

def apply_thermal_material(obj):
    """应用热辐射材质到对象"""
    # 创建新材质
    mat_name = "InfraredThermalMaterial"
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
    
    # 创建节点结构
    # 1. 温度属性节点
    attr_node = nodes.new('ShaderNodeAttribute')
    attr_node.attribute_name = "Temperature"
    attr_node.location = (-800, 0)
    
    # 2. 映射范围节点 - 将温度值映射到0-1范围
    map_node = nodes.new('ShaderNodeMapRange')
    map_node.location = (-600, 0)
    map_node.inputs['From Min'].default_value = 300  # 最低温度(K)
    map_node.inputs['From Max'].default_value = 800  # 最高温度(K)
    map_node.inputs['To Min'].default_value = 0.0
    map_node.inputs['To Max'].default_value = 1.0
    
    # 3. 颜色渐变节点
    color_ramp = nodes.new('ShaderNodeValToRGB')
    color_ramp.location = (-400, 0)
    
    # 设置温度颜色渐变
    # 蓝色(冷) -> 青色 -> 绿色 -> 黄色 -> 红色(热) -> 白色(最热)
    color_ramp.color_ramp.interpolation = 'LINEAR'
    
    # 确保有4个控制点
    if len(color_ramp.color_ramp.elements) < 4:
        # 添加新的控制点
        for _ in range(4 - len(color_ramp.color_ramp.elements)):
            color_ramp.color_ramp.elements.new(0)
    
    # 设置控制点位置和颜色
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0, 0, 1, 1)  # 蓝色 - 冷
    
    color_ramp.color_ramp.elements[1].position = 0.2
    color_ramp.color_ramp.elements[1].color = (0, 1, 1, 1)  # 青色
    
    color_ramp.color_ramp.elements[2].position = 0.5
    color_ramp.color_ramp.elements[2].color = (0, 1, 0, 1)  # 绿色
    
    color_ramp.color_ramp.elements[3].position = 0.8
    color_ramp.color_ramp.elements[3].color = (1, 1, 0, 1)  # 黄色
    
    # 添加第五个点用于红色到白色的过渡
    color_ramp.color_ramp.elements.new(1.0)
    color_ramp.color_ramp.elements[4].position = 1.0
    color_ramp.color_ramp.elements[4].color = (1, 1, 1, 1)  # 白色 - 最热
    
    # 4. 自发光节点
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (-200, 0)
    emission.inputs[1].default_value = 1.0  # 高强度的自发光
    
    # 5. 材质输出节点
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (0, 0)
    
    # 连接节点
    links.new(attr_node.outputs["Color"], map_node.inputs["Value"])
    links.new(map_node.outputs["Result"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    
    # 应用材质到对象
    if obj.data.materials:
        # 使用现有材质槽
        for slot in obj.material_slots:
            slot.material = mat
    else:
        # 添加新材质槽
        obj.data.materials.append(mat)
    
    # 强制视图更新
    bpy.context.view_layer.update()

def setup_scene():
    """设置场景物理参数"""
    scene = bpy.context.scene
    
    # 设置渲染引擎为Cycles以更好显示自发光
    scene.render.engine = 'CYCLES'
    
    # 设置透明度
    scene.render.film_transparent = True
    
    # 设置视图转换
    scene.view_settings.view_transform = 'Standard'
    
    # 添加辉光效果 - 兼容不同Blender版本
    if scene.render.engine == 'BLENDER_EEVEE':
        try:
            # Blender 2.8-2.92 版本
            if hasattr(scene.eevee, 'use_bloom'):
                scene.eevee.use_bloom = True
                scene.eevee.bloom_threshold = 0.1
                scene.eevee.bloom_intensity = 1.0
                scene.eevee.bloom_knee = 0.5
            
            # Blender 3.0+ 版本
            elif hasattr(scene.eevee, 'bloom'):
                scene.eevee.use_bloom = True
                scene.eevee.bloom.threshold = 0.1
                scene.eevee.bloom.intensity = 1.0
                scene.eevee.bloom.knee = 0.5
        except AttributeError:
            print("警告：当前Blender版本可能不支持辉光设置")
    
    # 设置后期处理辉光
    scene.render.use_compositing = True
    scene.render.use_sequencer = False

def refresh_viewport():
    """强制刷新3D视图"""
    # 切换到材质预览模式
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            # 设置着色模式为材质预览
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
                    space.shading.use_scene_world = True
                    space.shading.use_scene_lights = True
                    space.shading.show_object_outline = False
    
    # 强制重绘所有视图
    bpy.context.view_layer.update()
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

def main():
    # 获取当前活动对象
    obj = bpy.context.active_object
    
    # 检查是否有选中对象
    if not obj:
        print("错误：没有选中任何对象！请先选择一个网格对象")
        return
    
    # 检查对象类型是否为网格
    if obj.type != 'MESH':
        # 尝试自动转换为网格
        try:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.convert(target='MESH')
            print(f"已将 {obj.name} 转换为网格对象")
        except Exception as e:
            print(f"错误：选中的对象 '{obj.name}' 不是网格类型（{obj.type}），且无法转换：{str(e)}")
            return
    
    print(f"正在处理网格对象: {obj.name}")
    
    # 获取网格数据
    mesh = obj.data
    
    # 确保网格有温度属性
    if "Temperature" not in mesh.attributes:
        mesh.attributes.new(name="Temperature", type='FLOAT', domain='POINT')
    
    # 计算物体边界范围
    vertices = mesh.vertices
    min_z = min(v.co.z for v in vertices) if vertices else 0
    max_z = max(v.co.z for v in vertices) if vertices else 0
    height_range = max(max_z - min_z, 0.1)  # 防止除以零
    
    print(f"物体高度范围: {min_z:.2f} 到 {max_z:.2f} (范围: {height_range:.2f})")
    
    # 设置合理的温度分布 (300K~800K)
    for i, vert in enumerate(vertices):
        # 计算高度比例 (0到1)
        height_factor = (vert.co.z - min_z) / height_range
        
        # 限制在0-1范围内
        height_factor = max(0.0, min(1.0, height_factor))
        
        # 计算温度 (从300K到800K)
        temperature = 300 + height_factor * 500
        
        # 确保温度在合理范围内
        temperature = min(2000, max(100, temperature))
        
        mesh.attributes["Temperature"].data[i].value = temperature
        # 可选：打印第一个顶点的温度值
        if i == 0:
            print(f"顶点0: 高度 {vert.co.z:.2f}, 温度 {temperature:.1f}K")
    
    # 应用热辐射材质
    apply_thermal_material(obj)
    
    # 计算并打印部分点的辐射强度 (验证)
    if vertices:
        # 底部顶点
        bottom_temp = mesh.attributes["Temperature"].data[0].value
        bottom_IM = calculate_IM(bottom_temp)
        
        # 顶部顶点
        top_temp = mesh.attributes["Temperature"].data[-1].value
        top_IM = calculate_IM(top_temp)
        
        # 中间顶点
        mid_idx = len(vertices) // 2
        mid_temp = mesh.attributes["Temperature"].data[mid_idx].value
        mid_IM = calculate_IM(mid_temp)
        
        print(f"底部顶点: T = {bottom_temp:.1f}K, I_M = {bottom_IM:.3e} W/sr")
        print(f"中间顶点: T = {mid_temp:.1f}K, I_M = {mid_IM:.3e} W/sr")
        print(f"顶部顶点: T = {top_temp:.1f}K, I_M = {top_IM:.3e} W/sr")
    else:
        print("警告：网格中没有顶点")
    
    # 设置场景
    setup_scene()
    
    # 刷新视图
    refresh_viewport()
    
    print("红外热成像设置完成！")
    print("提示：如果效果不明显，尝试增加自发光强度或调高辉光效果")

# 执行主程序
if __name__ == "__main__":
    # 开始前清除控制台输出
    print("\n" * 5)
    print("===== 开始红外热成像设置 =====")
    main()