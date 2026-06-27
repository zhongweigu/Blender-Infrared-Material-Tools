"""
可视化模块 —— 将温度/辐射值映射为材质颜色并渲染输出

管线: per-face values → per-vertex averaging → Vertex Attribute →
      Map Range → Color Ramp → Emission → Output → Render
"""

import os
import bpy
import math
import numpy as np
from new_pipeline import config


# ============================================================
# 顶点转换: per-face → per-vertex
# ============================================================

def face_to_vertex(mesh, face_values):
    """Average per-face values onto vertices.

    Each vertex gets the mean of all adjacent face values.
    """
    n_verts = len(mesh.vertices)
    vert_sum = np.zeros(n_verts)
    vert_count = np.zeros(n_verts, dtype=np.int32)

    for fi, poly in enumerate(mesh.polygons):
        val = face_values[fi]
        for vi in poly.vertices:
            vert_sum[vi] += val
            vert_count[vi] += 1

    vert_count = np.maximum(vert_count, 1)
    return vert_sum / vert_count


# ============================================================
# 材质: 数值 → 颜色
# ============================================================

def assign_value_material(obj, mesh, per_face_values, attr_name="Value",
                           color_mode="thermal", vmin=None, vmax=None,
                           mat_name="IR_Value_Mat"):
    """Create a shader material that colors the mesh by per-face scalar values.

    Node chain: VertexAttribute(attr_name) → MapRange(vmin,vmax→0,1) →
               Gamma(Power) → ColorRamp → Emission → Output

    Gamma exponent and emission strength are read from config.py.
    """
    # Normalize range
    arr = np.asarray(per_face_values)
    if vmin is None:
        vmin = float(arr.min())
    if vmax is None:
        vmax = float(arr.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    # Convert to per-vertex and write attribute
    vert_vals = face_to_vertex(mesh, per_face_values)

    if attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name])

    attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
    attr.data.foreach_set('value', vert_vals.tolist())

    # Create material + node tree
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # --- Nodes ---
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (800, 0)

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (600, 0)
    emission.inputs["Strength"].default_value = float(config.RENDER_EMISSION_STRENGTH)

    attr_node = nodes.new("ShaderNodeAttribute")
    attr_node.location = (-400, 0)
    attr_node.attribute_name = attr_name

    map_range = nodes.new("ShaderNodeMapRange")
    map_range.location = (-200, 0)
    map_range.inputs['From Min'].default_value = vmin
    map_range.inputs['From Max'].default_value = vmax
    map_range.inputs['To Min'].default_value = 0.0
    map_range.inputs['To Max'].default_value = 1.0

    # ── Gamma boost: Power node with exponent < 1 brightens midtones ──
    gamma_node = nodes.new("ShaderNodeMath")
    gamma_node.location = (100, 0)
    gamma_node.operation = 'POWER'
    gamma_node.inputs[1].default_value = float(config.RENDER_GAMMA)

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.location = (400, 0)
    color_ramp.color_ramp.interpolation = 'LINEAR'

    _setup_color_ramp(color_ramp, color_mode)

    # --- Links ---
    links.new(attr_node.outputs["Fac"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], gamma_node.inputs[0])
    links.new(gamma_node.outputs["Value"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    # --- Assign to object ---
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return mat


def _setup_color_ramp(color_ramp, mode):
    """Populate ColorRamp stops based on mode."""
    els = color_ramp.color_ramp.elements

    if mode == "thermal":
        # FLIR-style: blue → cyan → green → yellow → red
        els[0].position = 0.0
        els[0].color = (0.0, 0.0, 0.5, 1.0)  # deep blue
        els[1].position = 1.0
        els[1].color = (1.0, 0.0, 0.0, 1.0)  # red

        stops = [
            (0.15, (0.0, 0.5, 1.0, 1.0)),   # cyan
            (0.35, (0.0, 1.0, 0.0, 1.0)),   # green
            (0.60, (1.0, 1.0, 0.0, 1.0)),   # yellow
            (0.85, (1.0, 0.5, 0.0, 1.0)),   # orange
        ]
        for pos, color in stops:
            el = els.new(pos)
            el.color = color

    elif mode == "bw":
        t = float(config.RENDER_BW_THRESHOLD)
        g = float(config.RENDER_BW_BASE_GRAY)
        els[0].position = 0.0
        els[0].color = (g, g, g, 1.0)
        # threshold stop: below this value everything stays flat gray
        el_mid = els.new(t)
        el_mid.color = (g + 0.02, g + 0.02, g + 0.02, 1.0)
        els[1].position = 1.0
        els[1].color = (1.0, 1.0, 1.0, 1.0)

    else:
        raise ValueError(f"Unknown color mode: {mode}")


# ============================================================
# 摄像机 / 渲染
# ============================================================

def _look_at_rotation(cam_location, target):
    """Compute Euler rotation so camera at cam_location looks at target."""
    from mathutils import Vector
    direction = Vector(target) - Vector(cam_location)
    # Track to: -Z points at target, Y is up
    quat = direction.to_track_quat('-Z', 'Y')
    return quat.to_euler()


def setup_camera(cam_location, target=(0, 0, 2),
                  cam_name="IR_Camera",
                  resolution=(1920, 1080)):
    """Create or update a render camera with Cycles, looking at target."""
    if cam_name in bpy.data.objects:
        cam = bpy.data.objects[cam_name]
    else:
        cam_data = bpy.data.cameras.new(cam_name)
        cam = bpy.data.objects.new(cam_name, cam_data)
        bpy.context.collection.objects.link(cam)

    cam.location = cam_location
    cam.rotation_euler = _look_at_rotation(cam_location, target)
    bpy.context.scene.camera = cam

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]

    return cam


def render_to_file(output_path):
    """Render current scene to PNG."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"[visualize] 渲染完成: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 多视角渲染预设 (真实尺度飞机, ~50m 长)
#   - 目标点 target=(0, 0, 2) 为飞机中心偏上
#   - Y+ 为机头方向, Y- 为机尾
# ══════════════════════════════════════════════════════════════════════════════

MULTIVIEW_PRESETS = [
    # (镜头名称, 相机位置, 目标点)
    ("01_right_side",       (0,    -80,  10),  (0, 0, 3)),
    ("02_front_quarter",    (40,   -70,  12),  (0, 0, 3)),
    ("03_front_high",       (35,   -55,  40),  (0, 0, 2)),
    ("04_rear_quarter",     (-35,   60,  10),  (0, 0, 3)),
    ("05_top_down",         (0,      5,  55),  (0, 0, 2)),
    ("06_belly_front",      (25,   -45, -20),  (0, 0, 2)),
    ("07_nose_close",       (15,   -30,   5),  (0, 5, 2)),
    ("08_tail_exhaust",     (-10,    5,   5),  (0, -20, 3)),
]


def render_multiview(obj, mesh, face_values, output_dir, base_name="radiance",
                      color_mode="thermal", vmin=None, vmax=None,
                      attr_name="Radiance"):
    """从多个预设视角渲染辐射/温度分布图。

    Args:
        obj: Blender mesh object
        mesh: mesh data block
        face_values: (N_faces,) array of scalar values
        output_dir: directory for output PNGs
        base_name: prefix for output filenames
        color_mode: 'thermal' or 'bw'
        vmin, vmax: normalization override
        attr_name: vertex attribute name
    """
    print(f"\n[visualize] 多视角渲染 ({len(MULTIVIEW_PRESETS)} 个镜头)...")
    clear_scene_materials()

    assign_value_material(
        obj, mesh, face_values,
        attr_name=attr_name,
        color_mode=color_mode,
        vmin=vmin, vmax=vmax,
        mat_name="IR_Radiance",
    )

    os.makedirs(output_dir, exist_ok=True)

    for cam_name, location, target in MULTIVIEW_PRESETS:
        setup_camera(location, target=target, cam_name=cam_name)
        output_path = os.path.join(output_dir, f"{base_name}_{cam_name}.png")
        render_to_file(output_path)


# ============================================================
# 便捷入口
# ============================================================

def clear_scene_materials():
    """Remove all material slots from all mesh objects and purge orphan materials."""
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
    # Purge orphan materials
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)


def render_temperature(obj, mesh, face_values, output_path,
                        color_mode="thermal", cam_location=None,
                        target=None, vmin=None, vmax=None,
                        attr_name="Radiance"):
    """Full pipeline: assign value material → setup camera → render.

    Args:
        obj: Blender mesh object (aircraft)
        mesh: mesh data
        face_values: (N_faces,) array of scalar values
        output_path: PNG output path
        color_mode: 'thermal' or 'bw'
        cam_location: camera world position (default: right side, 80m away)
        target: look-at point (default: aircraft center)
        vmin, vmax: normalization override (auto-compute if None)
        attr_name: vertex attribute name
    """
    print(f"\n[visualize] 清除旧材质...")
    clear_scene_materials()

    print(f"[visualize] 分配材质 ({color_mode})...")
    assign_value_material(
        obj, mesh, face_values,
        attr_name=attr_name,
        color_mode=color_mode,
        vmin=vmin, vmax=vmax,
        mat_name="IR_Radiance",
    )
    print(f"  数值范围: {vmin or face_values.min():.2f} ~ {vmax or face_values.max():.2f}")

    if cam_location is None:
        cam_location = (0, -80, 10)
    if target is None:
        target = (0, 0, 3)

    # White background
    scene = bpy.context.scene
    world = scene.world
    bg_restore = None
    if world and world.use_nodes:
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_restore = (
                bg_node.inputs['Color'].default_value[:],
                bg_node.inputs['Strength'].default_value,
            )
            bg_node.inputs['Color'].default_value = (1, 1, 1, 1)
            bg_node.inputs['Strength'].default_value = 1.0

    setup_camera(cam_location, target=target)
    render_to_file(output_path)

    if bg_restore:
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Color'].default_value = bg_restore[0]
            bg_node.inputs['Strength'].default_value = bg_restore[1]
