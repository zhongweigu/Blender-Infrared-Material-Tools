"""
清除所有材质，背景置白，架好摄像机。
在 Blender Scripting 工作区 Alt+P 运行，然后手动截图。
"""
import bpy
from mathutils import Vector

# 1. 清除所有材质
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        obj.data.materials.clear()
for mat in list(bpy.data.materials):
    if mat.users == 0:
        bpy.data.materials.remove(mat)

# 2. 背景置白
world = bpy.context.scene.world
if not world:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links
bg = nodes.get("Background")
if not bg:
    bg = nodes.new("ShaderNodeBackground")
out = nodes.get("World Output")
if not out:
    out = nodes.new("ShaderNodeWorldOutput")
    links.new(bg.outputs["Background"], out.inputs["Surface"])
bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
bg.inputs["Strength"].default_value = 1.0

# 3. 摄像机 — 和其他 process images 一样的位置
cam_name = "Debug_Camera"
if cam_name in bpy.data.objects:
    cam = bpy.data.objects[cam_name]
else:
    cam_data = bpy.data.cameras.new(cam_name)
    cam = bpy.data.objects.new(cam_name, cam_data)
    bpy.context.collection.objects.link(cam)

cam_loc = (42, 72, -25)
target = (0, 0, 1)
cam.location = cam_loc
direction = Vector(target) - Vector(cam_loc)
cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
bpy.context.scene.camera = cam

# 4. 视口线框模式 + 选择飞机
aircraft = bpy.data.objects.get("Aircraft")
if aircraft:
    for obj in bpy.context.view_layer.objects:
        obj.select_set(False)
    aircraft.select_set(True)
    bpy.context.view_layer.objects.active = aircraft

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'WIREFRAME'
                # 隐藏游标、网格地板、坐标轴
                space.overlay.show_cursor = False
                space.overlay.show_floor = False
                space.overlay.show_axis_x = False
                space.overlay.show_axis_y = False
                space.overlay.show_axis_z = False
                space.overlay.show_object_origins = False

print("完成。材质已清除，背景已置白，摄像机已就位。")
print(f"摄像机: {cam_loc}, 朝向 {target}")
print("Numpad 0 → 摄像机视角，然后手动截图。")
