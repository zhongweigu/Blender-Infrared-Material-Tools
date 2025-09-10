import bpy
import math

# ---------- 新增：渲染函数 ----------
def render_ir_image(output_path, cam_location=(0, -30, 10), cam_rotation=(math.radians(75), 0, 0)):
    """
    设置摄像机位置和角度，并渲染红外图像保存
    :param output_path: 输出图片路径 (绝对路径或相对路径)
    :param cam_location: 摄像机位置 (x, y, z)
    :param cam_rotation: 摄像机旋转欧拉角 (rx, ry, rz)，单位：弧度
    """
    # 获取或创建摄像机
    if "IR_Camera" in bpy.data.objects:
        cam = bpy.data.objects["IR_Camera"]
    else:
        cam_data = bpy.data.cameras.new("IR_Camera")
        cam = bpy.data.objects.new("IR_Camera", cam_data)
        bpy.context.collection.objects.link(cam)

    # 设置位置和角度
    cam.location = cam_location
    cam.rotation_euler = cam_rotation

    # 设为活动摄像机
    bpy.context.scene.camera = cam

    # 渲染设置
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'  # 或 'BLENDER_EEVEE'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.filepath = output_path

    # 渲染
    bpy.ops.render.render(write_still=True)
    print(f"渲染完成，图像已保存到 {output_path}")