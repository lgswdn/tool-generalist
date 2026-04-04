import numpy as np
from skimage import measure
import trimesh
import open3d as o3d
import igl
from scipy.spatial.transform import Rotation as R

# Global voxel grid parameters - all objects use the same coordinate system
GLOBAL_VOXEL_RES = 256
GLOBAL_VOXEL_MIN = np.array([-21.0, -21.0, -21.0])
GLOBAL_VOXEL_MAX = np.array([21.0, 21.0, 21.0])


# Global tracking for head area
_HEAD_AREA = None

class VoxelObject:
    """Wrapper for voxel grid that mimics mesh interface."""
    def __init__(self, grid, bbox=None):
        self.grid = grid
        self.bbox = bbox  # Store bbox at creation time

    def export(self, filename):
        """Convert to mesh and export."""
        print(f"  [export] Adding surface variation...")
        smoothed_grid = add_surface_variation(self.grid, noise_level=0.12)
        print(f"  [export] Converting voxel grid to mesh...")
        mesh = grid_to_mesh(smoothed_grid)
        mesh.export(filename)
        print(f"  [export] Saved to {filename}")

def _create_bowl_mesh(width, depth, thickness, n_sectors=32, n_rings=16):
    """
    Create a watertight bowl mesh via surface of revolution.

    The bowl opening faces +Z (upward), convex bottom at -Z.
    Centered at the origin (bounding box midpoint = (0,0,0)).

    Args:
        width (float): Diameter of the bowl opening.
        depth (float): Depth of the bowl cavity (inside).
        thickness (float): Wall thickness.
        n_sectors (int): Number of angular divisions around the Z axis.
        n_rings (int): Number of latitudinal rings from pole to rim.

    Returns:
        trimesh.Trimesh: A watertight bowl mesh.
    """
    half_width = width / 2.0

    # Derive sphere radius and cap angle from width and depth
    # R*(1 - cos(cap_angle)) = depth, R*sin(cap_angle) = half_width
    # => R = (half_width^2 + depth^2) / (2 * depth)
    R_outer = (half_width ** 2 + depth ** 2) / (2.0 * depth)
    cap_angle = np.arcsin(np.clip(half_width / R_outer, -1.0, 1.0))
    R_inner = R_outer - thickness

    if R_inner <= 0:
        raise ValueError(
            f"Bowl thickness ({thickness}) is too large for the given curvature "
            f"(R={R_outer:.2f}). Reduce thickness or increase width/depth."
        )

    # Strategy: Build a solid of revolution from a 2D cross-section profile.
    # The profile is a closed loop in the (r, z) plane:
    #   bottom_point -> outer surface (up to rim) -> across rim -> inner surface (down to bottom) -> close
    #
    # We revolve this closed profile around the Z axis.
    # The bottom point is on the axis (r=0), so it becomes a single shared vertex.

    # Build the 2D profile points (r, z) going around the cross-section
    # alpha goes from 0 (pole) to cap_angle (rim)
    alphas = np.linspace(0, cap_angle, n_rings + 1)  # includes 0 and cap_angle

    profile = []
    # Outer surface: from pole upward to rim
    for alpha in alphas:
        r = R_outer * np.sin(alpha)
        z = -R_outer * np.cos(alpha)
        profile.append((r, z))
    # Inner surface: from rim back down to pole
    for alpha in reversed(alphas):
        r = R_inner * np.sin(alpha)
        z = -R_inner * np.cos(alpha)
        profile.append((r, z))

    # Remove duplicate at junction (inner pole = near outer pole, both at r≈0)
    # The profile has 2*(n_rings+1) points. The first and last points are both on the axis.
    # We'll use a single vertex for the bottom.
    n_profile = len(profile)

    # --- Build vertices and faces via revolution ---
    vertices = []
    faces = []

    # For each profile point, if r > 0, create n_sectors vertices around Z axis.
    # If r == 0 (on axis), create a single vertex.
    # Track vertex indices for each profile point.
    profile_vertex_indices = []  # For each profile point, list of vertex indices (or single index)

    for pi, (r, z) in enumerate(profile):
        if r < 1e-10:
            # On-axis point: single vertex
            idx = len(vertices)
            vertices.append([0.0, 0.0, z])
            profile_vertex_indices.append([idx] * n_sectors)  # repeat index for easy face lookup
        else:
            # Off-axis point: ring of n_sectors vertices
            indices = []
            for j in range(n_sectors):
                phi = 2.0 * np.pi * j / n_sectors
                idx = len(vertices)
                vertices.append([r * np.cos(phi), r * np.sin(phi), z])
                indices.append(idx)
            profile_vertex_indices.append(indices)

    # Create faces: quad strip between consecutive profile points, revolved
    for pi in range(n_profile - 1):
        idx_curr = profile_vertex_indices[pi]
        idx_next = profile_vertex_indices[pi + 1]

        for j in range(n_sectors):
            j_next = (j + 1) % n_sectors

            v0 = idx_curr[j]
            v1 = idx_curr[j_next]
            v2 = idx_next[j_next]
            v3 = idx_next[j]

            if v0 == v1:
                # Current profile point is on-axis (single vertex) -> triangle
                if v2 != v3:
                    faces.append([v0, v3, v2])
            elif v2 == v3:
                # Next profile point is on-axis (single vertex) -> triangle
                faces.append([v0, v2, v1])
            else:
                # Both off-axis -> quad (two triangles)
                faces.append([v0, v3, v2])
                faces.append([v0, v2, v1])

    # Close the profile loop: connect last profile point back to first
    idx_last = profile_vertex_indices[-1]
    idx_first = profile_vertex_indices[0]
    for j in range(n_sectors):
        j_next = (j + 1) % n_sectors

        v0 = idx_last[j]
        v1 = idx_last[j_next]
        v2 = idx_first[j_next]
        v3 = idx_first[j]

        if v0 == v1:
            if v2 != v3:
                faces.append([v0, v3, v2])
        elif v2 == v3:
            faces.append([v0, v2, v1])
        else:
            faces.append([v0, v3, v2])
            faces.append([v0, v2, v1])

    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)

    # Center the mesh at origin
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    vertices -= center

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # Fix normals to be consistent
    trimesh.repair.fix_normals(mesh)

    print(f"  [_create_bowl_mesh] Created bowl: width={width}, depth={depth}, "
          f"thickness={thickness}, R={R_outer:.2f}, cap_angle={np.degrees(cap_angle):.1f}deg, "
          f"watertight={mesh.is_watertight}")

    return mesh


def _create_arc_mesh(curvature_radius, cross_width, cross_height, arc_angle_deg, n_segments=32):
    """
    Create a watertight arc mesh by sweeping a rectangular cross-section along a circular arc.

    The arc lies in the XZ plane. One end (alpha=0) starts at the TOP (+Z direction).
    The arc curves from +Z into +X and then toward -Z:
        At   0deg: center is at (0, 0, +R)  -- top (attachment end)
        At  90deg: center is at (+R, 0, 0)  -- horizontal right
        At 180deg: center is at (0, 0, -R)  -- bottom
    Centered at the origin (bounding box midpoint = (0,0,0)) after construction.

    Args:
        curvature_radius (float): Radius of the arc centerline.
        cross_width (float): Width of the rectangular cross-section (radial direction).
        cross_height (float): Height of the cross-section (Y direction, perpendicular to arc plane).
        arc_angle_deg (float): Sweep angle in degrees (e.g., 180 for a U-hook, 90 for an L-hook).
        n_segments (int): Number of segments along the arc.

    Returns:
        trimesh.Trimesh: A watertight arc mesh.
    """
    R = curvature_radius
    hw = cross_width / 2.0   # half-width (radial)
    hh = cross_height / 2.0  # half-height (Y)
    arc_angle = np.radians(arc_angle_deg)

    if R <= hw:
        raise ValueError(
            f"Arc curvature_radius ({R}) must be > cross_section_width/2 ({hw}). "
            f"Otherwise the inner surface collapses."
        )

    # Arc starts at +Z and curves into +X.
    # At angle alpha, the arc center is at:
    #   x = R * sin(alpha),  z = R * cos(alpha)
    # Cross-section corners at each angle:
    #   Corner 0 (outer, +Y): ((R + hw)*sin(a), +hh, (R + hw)*cos(a))
    #   Corner 1 (outer, -Y): ((R + hw)*sin(a), -hh, (R + hw)*cos(a))
    #   Corner 2 (inner, -Y): ((R - hw)*sin(a), -hh, (R - hw)*cos(a))
    #   Corner 3 (inner, +Y): ((R - hw)*sin(a), +hh, (R - hw)*cos(a))

    angles = np.linspace(0, arc_angle, n_segments + 1)

    vertices = []
    # For each angle, add 4 corner vertices
    for alpha in angles:
        ca, sa = np.cos(alpha), np.sin(alpha)
        r_outer = R + hw
        r_inner = R - hw
        vertices.append([r_outer * sa, +hh, r_outer * ca])  # corner 0
        vertices.append([r_outer * sa, -hh, r_outer * ca])  # corner 1
        vertices.append([r_inner * sa, -hh, r_inner * ca])  # corner 2
        vertices.append([r_inner * sa, +hh, r_inner * ca])  # corner 3

    faces = []
    n_cross = 4  # corners per cross-section

    # Swept faces: connect consecutive cross-sections
    # 4 faces of the rectangular tube: outer(0-1), bottom(1-2), inner(2-3), top(3-0)
    face_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for seg in range(n_segments):
        base_curr = seg * n_cross
        base_next = (seg + 1) * n_cross
        for c_a, c_b in face_pairs:
            v0 = base_curr + c_a
            v1 = base_curr + c_b
            v2 = base_next + c_b
            v3 = base_next + c_a
            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    # End cap at start (alpha=0): rectangle with corners 0,1,2,3
    base_start = 0
    faces.append([base_start + 0, base_start + 2, base_start + 1])
    faces.append([base_start + 0, base_start + 3, base_start + 2])

    # End cap at end (alpha=arc_angle): rectangle with corners 0,1,2,3
    base_end = n_segments * n_cross
    faces.append([base_end + 0, base_end + 1, base_end + 2])
    faces.append([base_end + 0, base_end + 2, base_end + 3])

    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)

    # Center the mesh at origin
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    vertices -= center

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    trimesh.repair.fix_normals(mesh)

    print(f"  [_create_arc_mesh] Created arc: R={R}, cross={cross_width}x{cross_height}, "
          f"angle={arc_angle_deg}deg, watertight={mesh.is_watertight}")

    return mesh


_AXIS_VECTORS = {
    '+X': np.array([1.0, 0.0, 0.0]),
    '-X': np.array([-1.0, 0.0, 0.0]),
    '+Y': np.array([0.0, 1.0, 0.0]),
    '-Y': np.array([0.0, -1.0, 0.0]),
    '+Z': np.array([0.0, 0.0, 1.0]),
    '-Z': np.array([0.0, 0.0, -1.0]),
}


def _arc_plane_to_rotation_matrix(arc_plane):
    """
    Convert an arc_plane string to a 3x3 rotation matrix that pre-orients the arc.

    arc_plane is a string of exactly two signed-axis tokens, e.g. "+Z+X", "-Y+Z", "+X-Z".
      - Token 1: the direction the arc BODY TRAVELS at its start (alpha=0). Default: +X.
      - Token 2: the direction the arc body travels after 90 degrees of sweep. Default: -Z.

    The rotation matrix R satisfies:
        R @ [1,0,0] = T1  (maps default start tangent +X to T1)
        R @ [0,0,-1] = T2 (maps default 90-deg tangent -Z to T2)
        R @ [0,1,0]  = T1 x T2  (maps plane normal)

    Formula:  R = column_stack([T1, cross(T1, T2), -T2])

    Identity (no rotation): arc_plane="+X-Z"  (arc travels +X at start, -Z at 90deg)
    "+Z+X": arc starts going upward (+Z), curves toward +X after 90 deg.
    "+Z-X": arc starts going upward (+Z), curves toward -X after 90 deg.

    Args:
        arc_plane (str): Two signed-axis tokens. Token 1 = start travel direction,
            Token 2 = travel direction after 90 degrees. Must be orthogonal.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    arc_plane = arc_plane.strip()
    import re
    tokens = re.findall(r'[+\-][XYZxyz]', arc_plane)
    if len(tokens) != 2:
        raise ValueError(
            f"arc_plane '{arc_plane}' must contain exactly two signed-axis tokens "
            f"(e.g. '+X-Z', '+Z+X'). Got: {tokens}"
        )
    tok1 = tokens[0].upper()  # start tangent
    tok2 = tokens[1].upper()  # 90-deg tangent
    if tok1 not in _AXIS_VECTORS or tok2 not in _AXIS_VECTORS:
        raise ValueError(f"Unknown axis token(s): '{tok1}', '{tok2}'. Use +X/-X/+Y/-Y/+Z/-Z.")

    T1 = _AXIS_VECTORS[tok1].copy()
    T2 = _AXIS_VECTORS[tok2].copy()

    if abs(np.dot(T1, T2)) > 1e-6:
        raise ValueError(
            f"arc_plane axes '{tok1}' and '{tok2}' must be orthogonal."
        )

    # R maps default frame (+X start tangent, -Z 90-deg tangent) to target frame.
    # col0 = T1  (maps e_x = default start tangent)
    # col1 = T1 x T2  (maps e_y = default plane normal)
    # col2 = -T2  (maps e_z, since default 90-deg tangent is -e_z)
    normal = np.cross(T1, T2)
    rot_matrix = np.column_stack([T1, normal, -T2])
    return rot_matrix

def primitive(primitive_name, primitive_scale, is_head=False, is_base=False, arc_plane=None):
    """
    Create a 3D primitive mesh, centered at the origin with NO rotation.

    Args:
        primitive_name (str): One of {'cube', 'ball', 'cylinder', 'cone', 'bowl', 'arc'}.
        primitive_scale (list of float):
            - If cube, [sx, sy, sz]
            - If ball, [radius]
            - If cylinder, [radius, height]
            - If cone, [radius, height]
            - If bowl, [width, depth, thickness]
            - If arc, [curvature_radius, cross_section_width, cross_section_height, arc_angle]
        is_head (bool): If True, this primitive is part of the head area.
        is_base (bool): If True, record the center as base center.
        arc_plane (str or None): Optional axis-pair string for arcs only, e.g. "+Z+X".
            Specifies the initial orientation of the arc BEFORE voxelization:
              - Token 1: world direction the arc's start arm points (default '+Z')
              - Token 2: world direction the arc sweeps toward first (default '+X')
            The arc geometry is rotated in-place at the origin before being voxelized.
            Ignored for non-arc primitives.
    Returns:
        VoxelObject: The resulting voxel object.
    """
    primitive_name = primitive_name.lower()
    print(f"  [primitive] Creating {primitive_name} with scale {primitive_scale}...")

    if primitive_name == 'cube':
        if len(primitive_scale) != 3:
            raise ValueError("cube requires 3 scale parameters: [sx, sy, sz].")
        mesh = trimesh.creation.box(extents=primitive_scale)

    elif primitive_name == 'ball':
        if len(primitive_scale) != 1:
            raise ValueError("ball requires 1 scale parameter: [radius].")
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=primitive_scale[0])

    elif primitive_name == 'cylinder':
        if len(primitive_scale) != 2:
            raise ValueError("cylinder requires 2 scale parameters: [radius, height].")
        mesh = trimesh.creation.cylinder(radius=primitive_scale[0], height=primitive_scale[1], sections=32)

    elif primitive_name == 'cone':
        if len(primitive_scale) != 2:
            raise ValueError("cone requires 2 scale parameters: [radius, height].")
        mesh = trimesh.creation.cone(radius=primitive_scale[0], height=primitive_scale[1], sections=32)

    elif primitive_name == 'bowl':
        if len(primitive_scale) != 3:
            raise ValueError("bowl requires 3 scale parameters: [width, depth, thickness].")
        bowl_width, bowl_depth, bowl_thickness = primitive_scale
        mesh = _create_bowl_mesh(bowl_width, bowl_depth, bowl_thickness)

    elif primitive_name == 'arc':
        if len(primitive_scale) != 4:
            raise ValueError("arc requires 4 scale parameters: [curvature_radius, cross_section_width, cross_section_height, arc_angle].")
        arc_R, arc_cw, arc_ch, arc_angle = primitive_scale
        mesh = _create_arc_mesh(arc_R, arc_cw, arc_ch, arc_angle)

    else:
        raise ValueError(f"Unsupported primitive name: {primitive_name}. "
                         "Choose from 'cube', 'ball', 'cylinder', 'cone', 'bowl', or 'arc'.")

    # Apply arc_plane pre-rotation to the mesh geometry (centered at origin) before voxelization.
    # This rotates the entire arc in-place so the bounding box already reflects the new orientation.
    if arc_plane is not None and primitive_name == 'arc':
        rot_matrix = _arc_plane_to_rotation_matrix(arc_plane)
        mesh.vertices = (rot_matrix @ mesh.vertices.T).T
        print(f"  [primitive] Applied arc_plane='{arc_plane}' pre-rotation to arc")

    # Record base center if this is the base primitive
    if is_base:
        bb = mesh.bounds
        global _BASE_CENTER
        _BASE_CENTER = [(bb[0][0]+bb[1][0])/2, (bb[0][1]+bb[1][1])/2, bb[0][2]]
        print(f"  [primitive] Auto-recorded base center: {_BASE_CENTER}")

    # Convert mesh to voxel grid
    print(f"  [primitive] Converting to voxel grid...")
    grid = {}
    grid["res"] = GLOBAL_VOXEL_RES
    grid["data"] = np.zeros((GLOBAL_VOXEL_RES, GLOBAL_VOXEL_RES, GLOBAL_VOXEL_RES), dtype=np.uint8)
    grid["min_bound"] = GLOBAL_VOXEL_MIN
    grid["max_bound"] = GLOBAL_VOXEL_MAX
    grid = add_mesh(grid, mesh, voxel_value=2 if is_head else 1)

    # If empty, create a minimal 1x1x1 voxel at center
    if not np.any(grid["data"]):
        print(f"  [primitive] Warning: empty grid, creating 1x1x1 voxel at center")
        center_idx = GLOBAL_VOXEL_RES // 2
        grid["data"][center_idx, center_idx, center_idx] = 2 if is_head else 1

    # Compute and store bbox at creation time
    occupied = np.argwhere(grid["data"])
    min_idx = occupied.min(axis=0)
    max_idx = occupied.max(axis=0)
    min_bound = np.array(grid["min_bound"])
    max_bound = np.array(grid["max_bound"])
    voxel_size = (max_bound - min_bound) / grid["res"]
    min_world = min_bound + min_idx * voxel_size
    max_world = min_bound + (max_idx + 1) * voxel_size
    bbox = (min_world[0], min_world[1], min_world[2], max_world[0], max_world[1], max_world[2])

    return VoxelObject(grid, bbox=bbox)

def get_position(obj):
    """
    Return the center of the bottom face as (x, y, z).
    Args:
        obj: VoxelObject or trimesh.Trimesh
    Returns:
        tuple: (x, y, z) - center of bottom face
    """
    bb = get_axis_align_bounding_box(obj)
    return ((bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, bb[2])

def get_axis_align_bounding_box(obj):
    """
    Return the axis-aligned bounding box as (min_x, min_y, min_z, max_x, max_y, max_z).

    Args:
        obj: VoxelObject or trimesh.Trimesh

    Returns:
        tuple: (min_x, min_y, min_z, max_x, max_y, max_z)
    """
    if isinstance(obj, VoxelObject):
        if obj.bbox is not None:
            return obj.bbox
        # Fallback for objects without stored bbox
        grid = obj.grid
        data = grid["data"]
        occupied = np.argwhere(data)
        if len(occupied) == 0:
            min_b = grid["min_bound"]
            max_b = grid["max_bound"]
            return (min_b[0], min_b[1], min_b[2], max_b[0], max_b[1], max_b[2])
        min_idx = occupied.min(axis=0)
        max_idx = occupied.max(axis=0)
        res = grid["res"]
        min_bound = np.array(grid["min_bound"])
        max_bound = np.array(grid["max_bound"])
        voxel_size = (max_bound - min_bound) / res
        min_world = min_bound + min_idx * voxel_size
        max_world = min_bound + (max_idx + 1) * voxel_size
        return (min_world[0], min_world[1], min_world[2], max_world[0], max_world[1], max_world[2])
    else:
        bounds = obj.bounds
        (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
        return (min_x, min_y, min_z, max_x, max_y, max_z)


def union_mesh(obj1, obj2):
    """
    Combine two objects using boolean union.
    Works directly with voxel grids - no resampling needed.

    Args:
        obj1: VoxelObject
        obj2: VoxelObject

    Returns:
        VoxelObject: The union of the two objects.
    """
    print("  [union_mesh] Combining voxel grids...")

    grid1 = obj1.grid
    grid2 = obj2.grid

    # Since all grids use the same coordinate system, take maximum value
    new_grid = {
        "res": GLOBAL_VOXEL_RES,
        "data": np.maximum(grid1["data"], grid2["data"]),
        "min_bound": GLOBAL_VOXEL_MIN.copy(),
        "max_bound": GLOBAL_VOXEL_MAX.copy()
    }

    return VoxelObject(new_grid, bbox=obj1.bbox)

def resample_grid(src_grid, tgt_min, tgt_max, tgt_res):
    """Resample source grid to target coordinate system."""
    import time
    start = time.time()

    src_min = src_grid["min_bound"]
    src_max = src_grid["max_bound"]
    src_res = src_grid["res"]
    src_data = src_grid["data"]

    tgt_data = np.zeros((tgt_res, tgt_res, tgt_res), dtype=bool)

    # Map each occupied voxel from source to target
    occupied = np.argwhere(src_data)
    print(f"    [resample_grid] Found {len(occupied)} occupied voxels, resampling...")

    for idx in occupied:
        # Convert source voxel index to world coordinates
        world_pos = src_min + (idx + 0.5) * (src_max - src_min) / src_res

        # Convert world coordinates to target voxel index
        tgt_idx = ((world_pos - tgt_min) / (tgt_max - tgt_min) * tgt_res).astype(int)

        # Check bounds and set
        if np.all(tgt_idx >= 0) and np.all(tgt_idx < tgt_res):
            tgt_data[tuple(tgt_idx)] = True

    elapsed = time.time() - start
    print(f"    [resample_grid] Done in {elapsed:.2f}s")
    return tgt_data

def subtract_mesh(obj1, obj2):
    """
    Subtract obj2 from obj1.
    Works directly with voxel grids - no resampling needed.

    Args:
        obj1: VoxelObject - Base object
        obj2: VoxelObject - Object to subtract

    Returns:
        VoxelObject: obj1 with obj2 subtracted.
    """
    print("  [subtract_mesh] Subtracting voxel grids...")

    grid1 = obj1.grid
    grid2 = obj2.grid

    # Since all grids use the same coordinate system, subtract by zeroing
    new_grid = {
        "res": GLOBAL_VOXEL_RES,
        "data": np.where(grid2["data"] > 0, 0, grid1["data"]),
        "min_bound": GLOBAL_VOXEL_MIN.copy(),
        "max_bound": GLOBAL_VOXEL_MAX.copy()
    }

    return VoxelObject(new_grid, bbox=obj1.bbox)

def union_attach(obj_target, obj_source, target_point, source_point, rotation, rotation_variance, cached_target_bbox=None):
    """
    Attach source object to target object via union operation.

    Args:
        obj_target: VoxelObject - the target object (typically the base or accumulated result)
        obj_source: VoxelObject - the source object to attach
        target_point: [pa, qa, ra] - normalized coordinates [0,1] on target bounding box
        source_point: [pb, qb, rb] - normalized coordinates [0,1] on source bounding box
        rotation: [rx, ry, rz] - euler angles in degrees
        rotation_variance: [vx, vy, vz] - variance in degrees for each axis
        cached_target_bbox: tuple - DEPRECATED, uses obj_target.bbox instead

    Returns:
        VoxelObject: The combined object with source attached to target
    """
    print(f"  [union_attach] Attaching objects...")

    target_bb = obj_target.bbox if obj_target.bbox is not None else get_axis_align_bounding_box(obj_target)
    source_bb = obj_source.bbox if obj_source.bbox is not None else get_axis_align_bounding_box(obj_source)

    # Convert normalized coordinates to world coordinates
    target_pos = np.array([
        target_bb[0] + target_point[0] * (target_bb[3] - target_bb[0]),
        target_bb[1] + target_point[1] * (target_bb[4] - target_bb[1]),
        target_bb[2] + target_point[2] * (target_bb[5] - target_bb[2])
    ])

    source_pos = np.array([
        source_bb[0] + source_point[0] * (source_bb[3] - source_bb[0]),
        source_bb[1] + source_point[1] * (source_bb[4] - source_bb[1]),
        source_bb[2] + source_point[2] * (source_bb[5] - source_bb[2])
    ])

    # Apply random rotation variance
    import random
    rot_euler = np.array(rotation) + np.array([
        random.uniform(-rotation_variance[0], rotation_variance[0]),
        random.uniform(-rotation_variance[1], rotation_variance[1]),
        random.uniform(-rotation_variance[2], rotation_variance[2])
    ])
    rot_euler_rad = np.radians(rot_euler)

    # Rotate source object around its attachment point
    grid_src = obj_source.grid
    grid_tgt = obj_target.grid
    res = grid_src["res"]
    voxel_size = (grid_src["max_bound"] - grid_src["min_bound"]) / res

    # Create rotation matrix from euler angles (ZYX order)
    from scipy.spatial.transform import Rotation as R
    rotation_matrix = R.from_euler('xyz', rot_euler_rad).as_matrix()

    # Rotate voxel grid
    occupied = np.argwhere(grid_src["data"])
    if len(occupied) > 0:
        new_data = np.zeros_like(grid_src["data"])

        for idx in occupied:
            cur_pos = grid_src["min_bound"] + (idx + 0.5) * voxel_size - source_pos
            # Apply rotation
            rotated_pos = rotation_matrix @ cur_pos
            # Translate to target attachment point
            final_pos = rotated_pos + target_pos
            # Convert back to voxel indices
            new_idx = ((final_pos - grid_tgt["min_bound"]) / voxel_size).astype(int)

            if np.all(new_idx >= 0) and np.all(new_idx < res):
                new_data[tuple(new_idx)] = grid_src["data"][tuple(idx)]

        rotated_source = VoxelObject({
            "res": res,
            "data": new_data,
            "min_bound": grid_tgt["min_bound"].copy(),
            "max_bound": grid_tgt["max_bound"].copy()
        })
    else:
        rotated_source = obj_source

    # Union the grids
    return union_mesh(obj_target, rotated_source)


def subtract_attach(obj_target, obj_source, target_point, source_point, rotation, rotation_variance, cached_target_bbox=None):
    """
    Attach and subtract source object from target object.

    Args:
        obj_target: VoxelObject - the target object
        obj_source: VoxelObject - the source object to subtract
        target_point: [pa, qa, ra] - normalized coordinates [0,1] on target bounding box
        source_point: [pb, qb, rb] - normalized coordinates [0,1] on source bounding box
        rotation: [rx, ry, rz] - euler angles in degrees
        rotation_variance: [vx, vy, vz] - variance in degrees for each axis
        cached_target_bbox: tuple - DEPRECATED, uses obj_target.bbox instead

    Returns:
        VoxelObject: The target object with source subtracted
    """
    print(f"  [subtract_attach] Subtracting objects...")

    # Use stored bbox from target object
    target_bb = obj_target.bbox if obj_target.bbox is not None else get_axis_align_bounding_box(obj_target)
    source_bb = get_axis_align_bounding_box(obj_source)

    # Convert normalized coordinates to world coordinates
    target_pos = np.array([
        target_bb[0] + target_point[0] * (target_bb[3] - target_bb[0]),
        target_bb[1] + target_point[1] * (target_bb[4] - target_bb[1]),
        target_bb[2] + target_point[2] * (target_bb[5] - target_bb[2])
    ])

    source_pos = np.array([
        source_bb[0] + source_point[0] * (source_bb[3] - source_bb[0]),
        source_bb[1] + source_point[1] * (source_bb[4] - source_bb[1]),
        source_bb[2] + source_point[2] * (source_bb[5] - source_bb[2])
    ])

    # Apply random rotation variance
    import random
    rot_euler = np.array(rotation) + np.array([
        random.uniform(-rotation_variance[0], rotation_variance[0]),
        random.uniform(-rotation_variance[1], rotation_variance[1]),
        random.uniform(-rotation_variance[2], rotation_variance[2])
    ])
    rot_euler_rad = np.radians(rot_euler)

    # Rotate source object
    grid_src = obj_source.grid
    res = grid_src["res"]
    voxel_size = (grid_src["max_bound"] - grid_src["min_bound"]) / res

    from scipy.spatial.transform import Rotation as R
    rotation_matrix = R.from_euler('xyz', rot_euler_rad).as_matrix()

    occupied = np.argwhere(grid_src["data"])
    transformed_contact = None
    if len(occupied) > 0:
        new_data = np.zeros_like(grid_src["data"])

        for idx in occupied:
            world_pos = grid_src["min_bound"] + (idx + 0.5) * voxel_size - source_pos
            rotated_pos = rotation_matrix @ world_pos
            final_pos = rotated_pos + target_pos
            new_idx = ((final_pos - grid_src["min_bound"]) / voxel_size).astype(int)

            if np.all(new_idx >= 0) and np.all(new_idx < res):
                new_data[tuple(new_idx)] = True

        # Don't transform contact point for subtract - the source geometry is being removed
        # Contact point should only come from union operations

        rotated_source = VoxelObject({
            "res": res,
            "data": new_data,
            "min_bound": grid_src["min_bound"],
            "max_bound": grid_src["max_bound"]
        })
    else:
        rotated_source = obj_source

    return subtract_mesh(obj_target, rotated_source)


_HEAD_AREA = None
_BASE_CENTER = None

def get_head_area():
    """
    Get the bounding box of all primitives marked with is_head=True.

    Returns:
        list: [[min_x, min_y, min_z], [max_x, max_y, max_z]] or None if no head primitives
    """
    global _HEAD_AREA
    return _HEAD_AREA.tolist() if _HEAD_AREA is not None else None

def record_base_center(point):
    """
    Record the center position of the base primitive.
    MUST be called exactly once in the assemble function after positioning the base primitive.

    Args:
        point (tuple or list): The 3D coordinates [x, y, z] of the base primitive center.
    """
    global _BASE_CENTER
    if _BASE_CENTER is not None:
        raise RuntimeError("record_base_center can only be called once")
    _BASE_CENTER = list(point)
    print(f"  [record_base_center] Recorded base center: {_BASE_CENTER}")

def empty_grid():
    """
    Create an empty 256x256x256 boolean occupancy grid from -0.5 to +0.5 in each axis.

    Returns:
        dict: A dictionary containing:
            - 'data': np.ndarray of shape (256, 256, 256), dtype=bool (all False initially).
            - 'res':  integer (256).
            - 'min_bound': np.array([-0.5, -0.5, -0.5]).
            - 'max_bound': np.array([0.5, 0.5, 0.5]).
    """
    grid = {}
    grid["res"] = 256
    grid["data"] = np.zeros((256, 256, 256), dtype=bool)  # All empty at first
    grid["min_bound"] = np.array([-0.2, -0.2, -0.2])
    grid["max_bound"] = np.array([ 0.2,  0.2,  0.2])
    return grid


def add_mesh(grid, mesh, voxel_value=1):
    """
    Convert 'mesh' into a volume of occupied voxels using an SDF (signed-distance) test,
    then mark those voxels with voxel_value in 'grid'.

    Args:
        grid (dict): The grid dictionary from empty_grid().
        mesh (trimesh.Trimesh): A triangular mesh (assumed to fit in [-0.5, 0.5]^3).
        voxel_value (int): Value to set for occupied voxels (1=non-head, 2=head).

    Returns:
        dict: The updated grid, same reference as input.
    """
    # Unpack grid data
    res = grid["res"]
    data = grid["data"]
    vmin = grid["min_bound"]
    vmax = grid["max_bound"]
    
    # Prepare query points: center of each voxel in (x,y,z)
    # shape: (res^3, 3)
    xs = np.linspace(vmin[0], vmax[0], res, endpoint=False) + (vmax[0]-vmin[0])/(2*res)
    ys = np.linspace(vmin[1], vmax[1], res, endpoint=False) + (vmax[1]-vmin[1])/(2*res)
    zs = np.linspace(vmin[2], vmax[2], res, endpoint=False) + (vmax[2]-vmin[2])/(2*res)

    # Create a full 3D grid of points
    # XX.shape = (res, res, res), etc.
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    points = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    
    # Convert the mesh into arrays for libigl
    # libigl signed_distance expects:
    #   points:    (#P,3) array
    #   V:         (#V,3) array of mesh vertices
    #   F:         (#F,3) array of mesh faces (integers)
    V = mesh.vertices
    F = mesh.faces
    
    # Compute signed distance with libigl
    # sdf_values < 0  => inside
    result = igl.signed_distance(points, V, F)
    if isinstance(result, tuple):
        sdf_values = result[0]
    else:
        sdf_values = result
    
    # Reshape back to (res, res, res)
    sdf_3d = sdf_values.reshape((res, res, res))
    
    # Occupied if sdf < 0
    inside = (sdf_3d < 0)

    # Set voxel value where inside (use maximum to preserve head voxels)
    data[inside] = np.maximum(data[inside], voxel_value)

    return grid


def sub_mesh(grid, mesh):
    """
    Convert 'mesh' into a volume using an SDF, then set those voxels to False
    (subtract from the grid).
    """
    res = grid["res"]
    data = grid["data"]
    vmin = grid["min_bound"]
    vmax = grid["max_bound"]
    
    xs = np.linspace(vmin[0], vmax[0], res, endpoint=False) + (vmax[0]-vmin[0])/(2*res)
    ys = np.linspace(vmin[1], vmax[1], res, endpoint=False) + (vmax[1]-vmin[1])/(2*res)
    zs = np.linspace(vmin[2], vmax[2], res, endpoint=False) + (vmax[2]-vmin[2])/(2*res)

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    points = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    V = mesh.vertices
    F = mesh.faces

    result = igl.signed_distance(points, V, F)
    if isinstance(result, tuple):
        sdf_values = result[0]
    else:
        sdf_values = result
    sdf_3d = sdf_values.reshape((res, res, res))
    inside = (sdf_3d < 0)
    
    # Subtraction => any inside voxel becomes False
    data[inside] = False
    
    return grid

def cut_grid(grid):
    """
    Return two new grids of the same resolution (256x256x256):
      - grid_bottom: occupies only z < 0
      - grid_up: occupies only z >= 0
    by zeroing out the complementary region in each grid.

    Each returned grid has:
      - 'data': a (256,256,256) boolean array
      - same 'min_bound' and 'max_bound' as the original
      - same 'res' as the original

    Args:
        grid (dict): Must have keys 'data', 'res', 'min_bound', 'max_bound'.
            'data' is a 3D boolean array: (256,256,256).

    Returns:
        (dict, dict): (grid_up, grid_bottom)
    """
    # Unpack the original grid
    res = grid["res"]
    data = grid["data"]
    vmin = grid["min_bound"]
    vmax = grid["max_bound"]

    # Create full copies for up & bottom
    data_up = data.copy()
    data_bottom = data.copy()

    # z_cut_idx is the voxel index corresponding to z=0 in our [-0.5, 0.5] range
    # If the range is exactly 1.0 in z, then the midpoint is 0.5 * res => 128
    z_cut_idx = res // 2

    # Clear out the 'bottom half' in data_up => everything below z=0
    # i.e. for indices 0..(z_cut_idx-1), set to False
    data_up[:, :, :z_cut_idx] = False

    # Clear out the 'upper half' in data_bottom => everything above z=0
    # i.e. for indices z_cut_idx..(res-1), set to False
    data_bottom[:, :, z_cut_idx:] = False

    # Build the two new grid dicts
    grid_up = {
        "res": res,
        "data": data_up,
        "min_bound": vmin.copy(),
        "max_bound": vmax.copy()
    }
    grid_bottom = {
        "res": res,
        "data": data_bottom,
        "min_bound": vmin.copy(),
        "max_bound": vmax.copy()
    }

    return grid_up, grid_bottom

def add_surface_variation(grid, noise_level=0.1):
    """Add realistic surface variation by perturbing surface voxels."""
    import random
    data = grid["data"].copy()
    res = grid["res"]

    # Find surface voxels (occupied with at least one empty neighbor)
    surface_mask = np.zeros_like(data, dtype=bool)
    for i in range(1, res-1):
        for j in range(1, res-1):
            for k in range(1, res-1):
                if data[i,j,k]:
                    # Check 6-connected neighbors
                    if not (data[i-1,j,k] and data[i+1,j,k] and
                           data[i,j-1,k] and data[i,j+1,k] and
                           data[i,j,k-1] and data[i,j,k+1]):
                        surface_mask[i,j,k] = True

    # Randomly flip surface voxels
    surface_indices = np.argwhere(surface_mask)
    for idx in surface_indices:
        if random.random() < noise_level:
            i, j, k = idx
            data[i,j,k] = not data[i,j,k]

    return {"res": res, "data": data, "min_bound": grid["min_bound"], "max_bound": grid["max_bound"]}

def grid_to_mesh(grid, do_simplify=True, target_num_faces=3000):
    """
    Convert a 3D occupancy grid into a surface mesh using Marching Cubes.
    Optionally simplify the mesh using Open3D's quadric decimation.

    Args:
        grid (dict): A dictionary with keys:
            - 'data': (256,256,256) boolean array (True = occupied).
            - 'res': int, resolution (e.g. 256).
            - 'min_bound': np.array([x_min, y_min, z_min]).
            - 'max_bound': np.array([x_max, y_max, z_max]).
        do_simplify (bool): Whether to perform mesh simplification (default True).
        target_num_faces (int): If simplifying, the target number of faces.

    Returns:
        trimesh.Trimesh: The extracted (and optionally simplified) mesh. 
            If the grid is empty or no surface is found, faces might be empty.
    """
    data = grid["data"]    # uint8 volume, shape = (256,256,256), values 0/1/2
    res = grid["res"]      # Typically 256
    min_b = grid["min_bound"]
    max_b = grid["max_bound"]

    # 1) Convert to binary: any non-zero value is occupied
    volume = (data > 0).astype(np.float32)

    # Check if grid is empty or uniform
    if not np.any(volume) or np.all(volume):
        return trimesh.Trimesh(vertices=[], faces=[])

    # 2) Extract an isosurface at 0.5
    #    verts_voxel -> Nx3 array in "voxel space" [0..res-1]
    #    faces       -> Mx3 indices
    #    normals     -> Nx3 normal vectors
    #    values      -> Nx1 (unused here)
    verts_voxel, faces, normals, _ = measure.marching_cubes(volume, level=0.5)

    if len(faces) == 0:
        # Empty or uniform grid => no surface
        return trimesh.Trimesh(vertices=[], faces=[])

    # 3) Map voxel coordinates to real-world coordinates
    box_size = max_b - min_b  # e.g. [1.0, 1.0, 1.0] if bounding box is [-0.5..0.5]^3
    scale = box_size / float(res) 
    verts_world = verts_voxel * scale + min_b  # Shift + scale

    # 4) Build a Trimesh
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces[:, ::-1], vertex_normals=normals)

    if do_simplify and len(mesh.faces) > 0:
        # 5) Convert Trimesh -> Open3D, simplify, then convert back
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

        # Optional: compute vertex normals so Open3D knows how to handle them
        o3d_mesh.compute_vertex_normals()

        # Perform quadric decimation
        simplified_o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_num_faces)

        # Convert back to Trimesh
        simplified_vertices = np.asarray(simplified_o3d_mesh.vertices)
        simplified_faces = np.asarray(simplified_o3d_mesh.triangles)

        mesh = trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces)

    return mesh

import numpy as np

# -------------
# 1) Assume these API functions are already implemented somewhere:
#    - generate_3d(name, scale)  -> Trimesh
#    - rotate_to_align(mesh)     -> Trimesh
#    - move(mesh, offset)        -> Trimesh
#    - rescale(mesh, ratio)      -> Trimesh
#    - get_volume(mesh)          -> float
#    - get_axis_align_bounding_box(mesh) -> (minx, miny, minz, maxx, maxy, maxz)
#    - empty_grid()              -> dict with 'data', 'res', 'min_bound', 'max_bound'
#    - add_mesh(grid, mesh)      -> dict
#    - sub_mesh(grid, mesh)      -> dict
#    - cut_grid(grid)            -> (grid_up, grid_bottom)  # each 256x256x256
#    - grid_to_mesh(grid, ...)   -> Trimesh
#
# 2) We’ll just define a skeleton call for generate_3d here, 
#    assuming it returns a dummy Trimesh. In practice, you'd have 
#    a text-to-3D pipeline that returns a gummy bear shape.

# from my_api import (
#     generate_3d, rotate_to_align, move, rescale, get_volume,
#     get_axis_align_bounding_box, empty_grid, add_mesh, sub_mesh,
#     cut_grid, grid_to_mesh
# )

import trimesh
import os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
meshy_api_key = ''
try:
    with open(os.path.join(project_path, 'meshy_api_key.txt')) as fi:
        meshy_api_key = fi.readlines()[0]
except FileNotFoundError:
    pass  # Optional for manual mode

def generate_3d(name):
    import io
    import json
    import requests
    import subprocess

    cmd1 =   "curl https://api.meshy.ai/openapi/v2/text-to-3d " + \
            "  -H \'Authorization: Bearer {}\' ".format(meshy_api_key) + \
            "  -H \'Content-Type: application/json\' " + \
            "  -d \'{\n" + \
            "  \"mode\": \"preview\",\n" + \
            "  \"prompt\": \"{}\",\n".format(name) + \
            "  \"art_style\": \"realistic\",\n" + \
            "  \"should_remesh\": true\n" + \
            "}\'\n" 
    result = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
    task_id = json.loads(result.stdout)['result']
    # task_id = "0195c83d-6d69-7826-ac1f-e97aa7ba7541"
    
    cmd2 = "curl https://api.meshy.ai/openapi/v2/text-to-3d/{} ".format(task_id) + \
            "-H \"Authorization: Bearer {}\" ".format(meshy_api_key)
    result = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
    output = json.loads(result.stdout)
    while not output['status'] == 'SUCCEEDED':
        print("Waiting for meshy to finish...")
        result = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
        output = json.loads(result.stdout)
    mesh_file = output['model_urls']['obj']
    
    response = requests.get(mesh_file)
    mesh = trimesh.load(io.BytesIO(response.content), file_type='obj')
    
    return mesh