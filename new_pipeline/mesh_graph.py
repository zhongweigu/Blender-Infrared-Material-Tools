import bpy
import numpy as np
from collections import defaultdict
from new_pipeline import config


def get_mesh_data(obj):
    """Extract face centers, areas (world space), and per-face vertex lists from a mesh object."""
    mesh = obj.data
    matrix = obj.matrix_world
    n_faces = len(mesh.polygons)

    centers = np.empty((n_faces, 3))
    areas = np.empty(n_faces)
    face_verts = []  # [(v0_world, v1_world, v2_world, ...), ...]

    for i, poly in enumerate(mesh.polygons):
        verts_local = [mesh.vertices[v].co for v in poly.vertices]
        verts_world = [matrix @ v for v in verts_local]

        # World-space center: average of world-space vertices
        centers[i] = sum(verts_world, start=np.zeros(3)) / len(verts_world)

        # World-space area via cross product (works for convex polygons)
        if len(verts_world) >= 3:
            v0, v1, v2 = verts_world[0], verts_world[1], verts_world[2]
            area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) * 0.5
            areas[i] = area
        else:
            areas[i] = 0.0

        face_verts.append(verts_world)

    return centers, areas, face_verts


def build_face_adjacency(obj):
    """Build neighbor lists and shared edge lengths for each face.

    Args:
        obj: Blender mesh object (must have obj.data for polygons/vertices)

    Returns:
        neighbors: list of lists, neighbors[i] = [face_idx, ...]
        edge_lengths: dict, edge_lengths[(i, j)] = shared edge length in world space
    """
    mesh = obj.data
    matrix = obj.matrix_world
    n_faces = len(mesh.polygons)

    # Map edge_key -> list of face indices
    edge_faces = defaultdict(list)
    for i, poly in enumerate(mesh.polygons):
        for ek in poly.edge_keys:
            edge_faces[ek].append(i)

    neighbors = [[] for _ in range(n_faces)]
    edge_lengths = {}

    for ek, faces in edge_faces.items():
        if len(faces) == 2:
            i, j = faces
            v1 = matrix @ mesh.vertices[ek[0]].co
            v2 = matrix @ mesh.vertices[ek[1]].co
            edge_len = (v2 - v1).length

            neighbors[i].append(j)
            neighbors[j].append(i)
            edge_lengths[(i, j)] = edge_len
            edge_lengths[(j, i)] = edge_len

    # ── Distance-based fallback for orphan faces ──
    # Faces with 0 edge-neighbors (steep/narrow faces at mesh seams) get
    # connected to their nearest face by center distance.
    orphan_indices = [i for i in range(n_faces) if len(neighbors[i]) == 0]
    if orphan_indices:
        # Compute all face centers in world space
        centers = np.empty((n_faces, 3), dtype=np.float64)
        for i, poly in enumerate(mesh.polygons):
            c = poly.center
            wc = matrix @ c
            centers[i] = (wc.x, wc.y, wc.z)

        for i in orphan_indices:
            ci = centers[i]
            # Find nearest non-orphan face (prefer well-connected faces)
            dists = np.sum((centers - ci) ** 2, axis=1)
            dists[i] = np.inf  # exclude self
            # Exclude other orphans to avoid orphan clusters
            for oi in orphan_indices:
                dists[oi] = np.inf
            j = int(np.argmin(dists))
            if np.isfinite(dists[j]):
                neighbors[i].append(j)
                neighbors[j].append(i)
                # Use center distance as proxy edge length
                proxy_len = float(np.sqrt(dists[j]))
                edge_lengths[(i, j)] = proxy_len
                edge_lengths[(j, i)] = proxy_len
        print(f"[mesh_graph] 距离回退: {len(orphan_indices)} 个孤立面片已连接到最近邻")

    return neighbors, edge_lengths


from new_pipeline.calibrate_compute import ensure_connectivity  # noqa: E402


def compute_conductances(centers, neighbors, edge_lengths, k=None, thickness=None):
    """Compute thermal conductances between adjacent faces.

    G_ij = k * t * L_edge / d_ij

    Where:
        d_ij = |center_i - center_j| (m)
        L_edge = length of shared edge (m)
        k = thermal conductivity of skin (W/(m·K))
        t = skin thickness (m)

    Returns:
        conductances: dict, conductances[(i, j)] = G_ij
    """
    if k is None:
        k = config.K_SKIN
    if thickness is None:
        thickness = config.SKIN_THICKNESS

    conductances = {}
    kt = k * thickness

    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if (i, j) in conductances:
                continue  # Already computed for reverse direction
            d_ij = np.linalg.norm(centers[i] - centers[j])
            if d_ij < 1e-9:
                d_ij = 1e-9
            L_edge = edge_lengths.get((i, j), 0.0)
            G = kt * L_edge / d_ij
            conductances[(i, j)] = G
            conductances[(j, i)] = G

    return conductances


def find_object(candidates):
    """Find a mesh object in the scene matching any of the given candidate names.

    Tries exact match first, then case-insensitive, then substring match.
    Returns the first matching MESH object, or None.
    """
    # 1) Exact match
    for name in candidates:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == 'MESH':
            return obj

    # 2) Case-insensitive match
    name_lower = {obj.name.lower(): obj for obj in bpy.data.objects if obj.type == 'MESH'}
    for name in candidates:
        match = name_lower.get(name.lower())
        if match is not None:
            return match

    # 3) Substring match (case-insensitive)
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        obj_lower = obj.name.lower()
        for name in candidates:
            if name.lower() in obj_lower:
                return obj

    return None


def find_aircraft():
    """Find the aircraft mesh object with flexible name matching.

    Priority:
      1. Exact match "model_normalized" (机身专用，不含.001/.002后缀)
      2. Contains "air" (case-insensitive) → Airliner, Aircraft, etc.
      3. Common body names: body, fuselage, plane, AIRFRAME
    """
    # 1. 精确匹配 model_normalized (不含后缀)
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            name_lower = obj.name.lower()
            if name_lower == "model_normalized":
                return obj

    # 2. 包含 "air" 或其他机身关键词
    candidates = ["air", "body", "fuselage", "plane", "airframe", "model"]
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        name_lower = obj.name.lower()
        # 排除发动机后缀变体
        if name_lower.startswith("model_normalized.") or name_lower.startswith("model_normalized_"):
            continue
        for name in candidates:
            if name.lower() in name_lower:
                return obj

    return None


def find_engine_left():
    """Find the left engine mesh object with flexible name matching."""
    candidates = ["Engin_L", "engin_l", "Engine_L", "engine_l",
                  "EngineL", "enginL", "Eng_L", "eng_l",
                  "engine_left", "Engine_Left", "ENGINE_L",
                  "model_normalized.001", "Model_Normalized.001"]
    return find_object(candidates)


def find_engine_right():
    """Find the right engine mesh object with flexible name matching."""
    candidates = ["Engin_R", "engin_r", "Engine_R", "engine_r",
                  "EngineR", "enginR", "Eng_R", "eng_r",
                  "engine_right", "Engine_Right", "ENGINE_R",
                  "model_normalized.002", "Model_Normalized.002"]
    return find_object(candidates)


def _centroid_x(obj):
    """World-space X centroid of a mesh object."""
    verts = obj.data.vertices
    mw = obj.matrix_world
    return sum((mw @ v.co).x for v in verts) / len(verts)


def find_all_engines():
    """Find all engine mesh objects, classified by X centroid.

    Engines are identified by:
      1. Name contains "engin" (case-insensitive)
      2. Name matches "model_normalized.001", ".002", ".003"... (ShapeNet后缀)
    Assigned left/right based on world-space X centroid.
    Falls back to vertex-count heuristic if no engine-named objects exist.

    Returns:
        (left_engines, right_engines): tuple of lists, each sorted by |X|
    """
    aircraft = find_aircraft()

    engine_patterns = ["engin"]

    candidates = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if aircraft and obj == aircraft:
            continue
        name_lower = obj.name.lower()

        # 1. 包含 "engin" 关键词
        if any(pat in name_lower for pat in engine_patterns):
            candidates.append(obj)
            continue

        # 2. model_normalized 后缀变体 (.001, .002, .003...)
        if name_lower.startswith("model_normalized.") or name_lower.startswith("model_normalized_"):
            candidates.append(obj)

    if not candidates:
        meshes = sorted(
            [(obj, len(obj.data.vertices)) for obj in bpy.data.objects if obj.type == 'MESH'],
            key=lambda x: x[1], reverse=True,
        )
        candidates = [obj for obj, _ in meshes[1:]]

    left, right = [], []
    for obj in candidates:
        x = _centroid_x(obj)
        if x < 0:
            left.append((x, obj))
        else:
            right.append((x, obj))

    left.sort(key=lambda p: p[0])
    right.sort(key=lambda p: p[0])

    return [obj for _, obj in left], [obj for _, obj in right]


def find_exhaust_position(engine_obj):
    """Find the engine exhaust position in world space.

    The exhaust is at the rear of the engine (minimum Y in world space).
    Returns the average position of vertices in the rear-most portion.
    """
    if engine_obj is None or engine_obj.type != 'MESH':
        return None

    mesh = engine_obj.data
    matrix = engine_obj.matrix_world

    # Get all world-space vertex Y coordinates
    world_y = np.array([(matrix @ v.co).y for v in mesh.vertices])
    threshold = np.percentile(world_y, 10)  # Bottom 10% by Y

    # Collect vertices in the rear portion
    rear_verts = []
    for v in mesh.vertices:
        wv = matrix @ v.co
        if wv.y <= threshold:
            rear_verts.append(wv)

    if not rear_verts:
        return None

    return sum(rear_verts, start=np.zeros(3)) / len(rear_verts)


def find_engine_center(engine_obj):
    """Find the engine body centroid in world space (all vertices average)."""
    if engine_obj is None or engine_obj.type != 'MESH':
        return None
    mesh = engine_obj.data
    matrix = engine_obj.matrix_world
    world_verts = [matrix @ v.co for v in mesh.vertices]
    if not world_verts:
        return None
    return sum(world_verts, start=np.zeros(3)) / len(world_verts)


def get_engine_characteristic_length(engine_obj):
    """Compute the characteristic length of an engine mesh (max bounding box dimension)."""
    if engine_obj is None or engine_obj.type != 'MESH':
        return 0.1  # fallback
    mesh = engine_obj.data
    matrix = engine_obj.matrix_world
    world_verts = np.array([matrix @ v.co for v in mesh.vertices])
    if len(world_verts) == 0:
        return 0.1
    bbox_size = world_verts.max(axis=0) - world_verts.min(axis=0)
    return float(np.max(bbox_size))


def get_engine_radius(engine_obj):
    """Compute approximate engine radius (half of max XZ extent)."""
    if engine_obj is None or engine_obj.type != 'MESH':
        return 0.05
    mesh = engine_obj.data
    matrix = engine_obj.matrix_world
    world_verts = np.array([matrix @ v.co for v in mesh.vertices])
    if len(world_verts) == 0:
        return 0.05
    x_extent = world_verts[:, 0].max() - world_verts[:, 0].min()
    z_extent = world_verts[:, 2].max() - world_verts[:, 2].min()
    return float(max(x_extent, z_extent)) * 0.5


def auto_exhaust_radius(engine_l, engine_r):
    """Compute adaptive exhaust heat source radius from engine dimensions."""
    eng_objs = [e for e in (engine_l, engine_r) if e is not None]
    if not eng_objs:
        return 0.1  # fallback

    avg_radius = np.mean([get_engine_radius(e) for e in eng_objs])
    r = avg_radius * config.EXHAUST_RADIUS_MULT
    return max(r, 0.01)


def decimate_obj(original_obj, ratio):
    """Create a decimated copy of a mesh object for faster computation.

    The original object and its data are untouched.
    Returns a new object that must be cleaned up by the caller.

    Args:
        original_obj: Blender mesh object
        ratio: fraction of faces to keep (1.0 = no reduction, 0.15 = 15%)

    Returns:
        (new_obj, is_new) — new_obj is the decimated object; is_new is False
        if ratio >= 1.0 and the original was returned as-is.
    """
    if ratio >= 1.0:
        return original_obj, False

    # Duplicate mesh data
    new_mesh = original_obj.data.copy()

    # Create a temp object for the operator
    temp = bpy.data.objects.new("_temp_decimate", new_mesh)
    bpy.context.collection.objects.link(temp)
    bpy.context.view_layer.objects.active = temp

    # Decimate in edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.decimate(ratio=ratio)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create final object and remove temp
    result = bpy.data.objects.new(original_obj.name + "_preview", temp.data)
    bpy.context.collection.objects.link(result)
    result.matrix_world = original_obj.matrix_world

    bpy.data.objects.remove(temp, do_unlink=True)

    return result, True


def upsample_temperatures(src_centers, src_temps, dst_obj):
    """Map per-face temperatures from a decimated mesh back to the original mesh.

    Uses nearest-neighbor lookup: each original face gets the temperature
    of the closest decimated face center.
    """
    dst_centers, _, _ = get_mesh_data(dst_obj)
    # Build KD-tree of source centers for fast nearest-neighbor
    from scipy.spatial import cKDTree
    tree = cKDTree(src_centers)
    _, idx = tree.query(dst_centers)
    return np.asarray(src_temps)[idx]


try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def upsample_temperatures_fallback(src_centers, src_temps, dst_centers):
    """Map temperatures using vectorized brute-force nearest neighbor.

    For use when scipy is not available (e.g. Blender's bundled Python).
    """
    # (n_dst, 3) vs (n_src, 3) → (n_dst, n_src) distances
    # Use batch processing to avoid O(n²) memory blowup
    n_dst = len(dst_centers)
    result = np.empty(n_dst)
    batch = 1000
    for start in range(0, n_dst, batch):
        end = min(start + batch, n_dst)
        diff = dst_centers[start:end, np.newaxis, :] - src_centers[np.newaxis, :, :]
        dists = np.sum(diff * diff, axis=2)  # squared distances
        nearest = np.argmin(dists, axis=1)
        result[start:end] = np.asarray(src_temps)[nearest]
    return result


def upsample_temperatures(src_centers, src_temps, dst_obj):
    """Map per-face temperatures from a decimated mesh back to the original mesh."""
    dst_centers, _, _ = get_mesh_data(dst_obj)
    if cKDTree is not None:
        tree = cKDTree(src_centers)
        _, idx = tree.query(dst_centers)
        return np.asarray(src_temps)[idx]
    return upsample_temperatures_fallback(src_centers, src_temps, dst_centers)


def cleanup_decimated(obj, is_new):
    """Remove a decimated object and its mesh if it was created by decimate_obj."""
    if not is_new:
        return
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def find_heat_source_faces(centers, heat_pos, radius):
    """Find face indices within a given radius of a heat source position."""
    if heat_pos is None:
        return []

    dists = np.linalg.norm(centers - heat_pos, axis=1)
    return list(np.where(dists <= radius)[0])


def find_faces_near_mesh(centers, engine_obj, radius=None):
    """Find face indices and distances to the nearest engine vertex.

    Uses KD-tree nearest-neighbor: for each face center, finds the closest
    engine vertex. Much more accurate than a sphere around the engine center —
    it captures the actual contact/proximity region between skin and engine.

    Returns:
        (indices, distances) — two parallel lists. indices[i] is the face
        index, distances[i] is the distance to the nearest engine vertex (m).
    """
    if engine_obj is None or engine_obj.type != 'MESH':
        return [], []
    if radius is None:
        radius = 0.1

    mesh = engine_obj.data
    matrix = engine_obj.matrix_world
    engine_verts = np.array([matrix @ v.co for v in mesh.vertices])

    if cKDTree is not None:
        tree = cKDTree(engine_verts)
        dists, _ = tree.query(centers)
        mask = dists <= radius
        idx = np.where(mask)[0]
        return list(idx), list(dists[mask])

    # Brute-force fallback (no scipy)
    idx_out, dists_out = [], []
    for i, c in enumerate(centers):
        d = float(np.min(np.linalg.norm(engine_verts - c, axis=1)))
        if d <= radius:
            idx_out.append(i)
            dists_out.append(d)
    return idx_out, dists_out


def lookup_engine_face_temp(skin_centers, skin_face_idx, engine_obj, engine_temps):
    """Find the temperature of the nearest engine face to a skin face.

    Uses KD-tree over engine face centers. Returns the engine face temperature
    that should drive the skin face's heat source.
    """
    eng_centers, _, _ = get_mesh_data(engine_obj)
    eng_T = np.asarray(engine_temps)

    if cKDTree is not None:
        tree = cKDTree(eng_centers)
        _, nearest_eng_face = tree.query(skin_centers[skin_face_idx])
        return float(eng_T[nearest_eng_face])

    # Fallback
    c = skin_centers[skin_face_idx]
    dists = np.linalg.norm(eng_centers - c, axis=1)
    return float(eng_T[np.argmin(dists)])


def find_cross_boundary_pairs(centers, engine_mask, max_pairs=5,
                               max_distance=10.0):
    """Find nearest (engine, aircraft) face pairs for structural thermal bridges.

    After joining meshes, engine and aircraft faces share no edges. This
    function finds nearby pairs across the boundary so we can add synthetic
    edges with structural conductance G = k_struct * A_struct / d.

    Args:
        centers: (N, 3) face centers
        engine_mask: (N,) bool array
        max_pairs: max aircraft faces to connect per engine face
        max_distance: max distance for a bridge (m, real scale)

    Returns:
        list of (eng_idx, ac_idx, distance)
    """
    eng_idx = np.where(engine_mask)[0]
    ac_idx = np.where(~engine_mask)[0]

    if len(eng_idx) == 0 or len(ac_idx) == 0:
        return []

    ac_centers = centers[ac_idx]
    pairs = []
    k = min(max_pairs, len(ac_idx))

    if cKDTree is not None:
        tree = cKDTree(ac_centers)
        for ei in eng_idx:
            dists, nearby = tree.query(centers[ei], k=k)
            if k == 1:
                dists, nearby = [float(dists)], [int(nearby)]
            for d, ni in zip(dists, nearby):
                d = float(d)
                if d < max_distance:
                    pairs.append((int(ei), int(ac_idx[ni]), d))
    else:
        for ei in eng_idx:
            dists = np.linalg.norm(ac_centers - centers[ei], axis=1)
            order = np.argsort(dists)[:k]
            for ni in order:
                d = float(dists[ni])
                if d < max_distance:
                    pairs.append((int(ei), int(ac_idx[ni]), d))

    return pairs


def symmetrize_mesh(obj):
    """Symmetrize mesh vertices across X=0 by mirror-averaging.

    For each vertex, finds its nearest neighbor in the X-reflected
    vertex set, then averages both positions to enforce strict
    geometric symmetry about the X=0 plane.

    The mesh topology (faces, edges) is unchanged — only vertex
    positions are modified.
    """
    from mathutils import Vector
    mesh = obj.data
    mw = obj.matrix_world
    n_verts = len(mesh.vertices)

    verts_world = np.array([mw @ v.co for v in mesh.vertices], dtype=np.float64)

    # Reflected point cloud (X flipped)
    reflected = verts_world.copy()
    reflected[:, 0] = -reflected[:, 0]

    # Nearest-neighbor: each vertex → its closest mirror counterpart
    if cKDTree is not None:
        tree = cKDTree(reflected)
        _, nn = tree.query(verts_world)
    else:
        nn = np.zeros(n_verts, dtype=np.int64)
        for i in range(n_verts):
            nn[i] = np.argmin(np.sum((reflected - verts_world[i]) ** 2, axis=1))

    # Partner world position: reflect the reflected point back
    partner = reflected[nn].copy()
    partner[:, 0] = -partner[:, 0]

    # Average original with mirror partner
    new_world = (verts_world + partner) * 0.5

    # Write back in local space
    mw_inv = mw.inverted()
    for i, v in enumerate(mesh.vertices):
        v.co = mw_inv @ Vector(new_world[i])

    # Measure residual asymmetry
    verts_after = np.array([mw @ v.co for v in mesh.vertices], dtype=np.float64)
    asym = np.abs(verts_after[:, 0]).mean()
    old_asym = np.abs(verts_world[:, 0]).mean()
    print(f"[mesh_graph] 对称化: X质心 {verts_world[:, 0].mean():.4f} → "
          f"{verts_after[:, 0].mean():.6f}, "
          f"平均|X| {old_asym:.4f} → {asym:.4f} m")
