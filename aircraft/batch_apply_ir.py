"""
批量为ShapeNet.v2中的.obj模型应用红外材质

用法:
    在Blender中运行: Alt+P 或 Run Script
    或命令行: blender -b --python ./aircraft/batch_apply_ir.py

输入: ShapeNet.v2文件夹路径 (包含多个synset子文件夹，每个子文件夹下有models/*.obj)
输出: output/文件夹，保存每个模型的.blend文件
"""

import bpy
import numpy as np
import os
import sys
import math
from tqdm import tqdm

# ============== 配置 ==============
# ShapeNet.v2 根目录，修改为你的实际路径
SHAPENET_ROOT = r"D:\codes\MTIR-Blender-InfraRed-Material-Tools\Shapenet.v2"

# 输出目录
OUTPUT_DIR = r"D:\codes\MTIR-Blender-InfraRed-Material-Tools\output"

# bl_IR 模块路径
BL_IR_PATH = r"D:\codes\MTIR-Blender-InfraRed-Material-Tools\aircraft"

# 是否渲染图像（可选）
RENDER_IMAGE = False
# ============== 配置结束 ==============


def setup_paths():
    """将bl_IR模块添加到路径"""
    if BL_IR_PATH not in sys.path:
        sys.path.append(BL_IR_PATH)


def enable_obj_import():
    """Blender 4.x 使用 wm.obj_import，不再需要手动启用插件"""
    pass


def import_obj(obj_path):
    """导入单个.obj文件，返回导入的对象"""
    # 确保OBJ导入插件已启用
    enable_obj_import()

    # 清除现有场景
    clear_scene()

    # 导入obj (Blender 4.x 使用 wm.obj_import)
    try:
        bpy.ops.import_scene.obj(filepath=obj_path)
    except AttributeError:
        bpy.ops.wm.obj_import(filepath=obj_path)

    # 获取导入的对象（通常是第一个）
    imported_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

    if not imported_objs:
        return None

    # 如果有多个对象，合并它们
    if len(imported_objs) > 1:
        bpy.ops.object.join()
        obj = bpy.context.active_object
    else:
        obj = imported_objs[0]

    # 重命名为Aircraft，以便main.py的逻辑识别
    obj.name = "Aircraft"

    # 确保是mesh类型
    if obj.type != 'MESH':
        return None

    return obj


def clear_scene():
    """清除所有对象"""
    # 取消所有选择
    bpy.ops.object.select_all(action='DESELECT')

    # 删除所有对象
    for obj in bpy.data.objects:
        obj.select_set(True)
    bpy.ops.object.delete()

    # 清除未使用的数据
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def apply_ir_material_to_object(obj):
    """直接调用main.py的apply_ir_material为Aircraft对象应用红外材质"""
    import main

    # 由于ShapeNet飞机没有独立引擎，临时禁用发动机相关计算
    from bl_IR import config
    original_consider_aero = config.CONSIDER_AERO
    config.CONSIDER_AERO = False

    # 调用main.py的逻辑，engine_heat_delta设为0
    main.apply_ir_material("Aircraft", engine_heat_delta=0.0)

    # 恢复原始设置
    config.CONSIDER_AERO = original_consider_aero


def render_ir_image(output_path, cam_location=(15.0, -30.0, 20.0)):
    """渲染IR图像"""
    from bl_IR import camera
    camera.render_ir_image(
        output_path,
        cam_location=cam_location,
        cam_rotation=(math.radians(60), 0, math.radians(30))
    )


def save_blend(output_path):
    """保存.blend文件"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    print(f"已保存: {output_path}")


def get_all_obj_files(shapenet_root):
    """递归查找所有.obj文件"""
    obj_files = []
    for root, dirs, files in os.walk(shapenet_root):
        for file in files:
            if file.endswith(".obj"):
                obj_files.append(os.path.join(root, file))
    return obj_files


def process_single_model(obj_path, output_dir, render=False):
    """处理单个模型"""
    from bl_IR import config  # 用于获取CAMERA_POS

    # 获取模型标识名（用于输出文件名）
    model_name = os.path.basename(os.path.dirname(os.path.dirname(obj_path)))  # ShapeNet结构: synsetID/models/model.obj，取synsetID

    # 导入模型
    obj = import_obj(obj_path)
    if obj is None:
        return False

    # 应用IR材质
    apply_ir_material_to_object(obj)

    # 渲染（可选）
    if render:
        render_path = os.path.join(output_dir, f"{model_name}_render.png")
        render_ir_image(render_path, cam_location=config.CAMERA_POS)

    # 保存blend
    blend_path = os.path.join(output_dir, f"{model_name}.blend")
    save_blend(blend_path)

    return True


def main():
    """主函数"""
    setup_paths()

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有obj文件
    obj_files = get_all_obj_files(SHAPENET_ROOT)

    if not obj_files:
        print(f"在 {SHAPENET_ROOT} 中未找到.obj文件")
        return

    print(f"找到 {len(obj_files)} 个.obj文件，开始处理...")

    # 统计
    success_count = 0
    fail_count = 0

    # tqdm进度条
    with tqdm(total=len(obj_files), desc="处理模型", unit="个", ncols=80) as pbar:
        for obj_path in obj_files:
            model_name = os.path.basename(os.path.dirname(os.path.dirname(obj_path)))
            pbar.set_postfix_str(model_name[:30])

            try:
                if process_single_model(obj_path, OUTPUT_DIR, RENDER_IMAGE):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception:
                fail_count += 1

            pbar.update(1)

    print(f"\n完成！成功: {success_count}, 失败: {fail_count}")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
