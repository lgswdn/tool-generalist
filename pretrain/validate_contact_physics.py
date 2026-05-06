#!/usr/bin/env python3
"""validate_contact_physics.py — Physics-based contact stability filter.

For each config in a .pt file, spawns the tool (kinematic, gravity=off)
and the object (dynamic, gravity=on) at the recorded contact pose in Isaac Sim,
runs physics for --settle-steps, then checks whether the tool and object are
still in contact.  Surviving configs (with the new settled object pose) are
written to --output.

USD path convention (auto-derived from the OBJ paths stored in the .pt file):
  tool  OBJ → .../normalized_models/<name>.obj
           → .../objects_usd/<name>/<name>.usd
  object OBJ → .../coacd_normalized/<stem>.obj
           → .../coacd_usd/<stem>/<stem>.usd

Usage (single file):
    python pretrain/validate_contact_physics.py \\
        --input  pretrain/new_pretrain/tmp_data/fork/mug_pose0.pt \\
        --output pretrain/new_pretrain/validated/ \\
        --num-envs 64 --settle-steps 200 --threshold 0.008 --headless

Usage (with video recording):
    python pretrain/validate_contact_physics.py \\
        --input  file.pt --output validated/ \\
        --num-envs 1 --max-configs 5 \\
        --record-video out.mp4 --headless --enable_cameras

Note: one Isaac Sim instance is created per .pt file (different tool/object
pairs require different USD assets in the scene).
"""

# ── CLI + AppLauncher bootstrap (must be before any omni/isaacsim imports) ──
import argparse
from isaaclab.app import AppLauncher

cli_parser = argparse.ArgumentParser(description="Physics-based contact stability filter")
cli_parser.add_argument("--input",  nargs="+", required=True, help=".pt file(s) to validate")
cli_parser.add_argument("--output", required=True, help="Output directory")
cli_parser.add_argument("--num-envs",      type=int,   default=2048,    help="Parallel envs per batch")
cli_parser.add_argument("--settle-steps",  type=int,   default=200,   help="Physics steps to settle")
cli_parser.add_argument("--threshold",     type=float, default=0.002, help="Contact distance threshold (m)")
cli_parser.add_argument("--record-video",  default=None, metavar="PATH",
                   help="If given, record a video of settling to this .mp4 path")
cli_parser.add_argument("--capture-every", type=int,   default=4,
                   help="Capture one frame every N physics steps (default: 4)")
cli_parser.add_argument("--max-configs",  type=int,   default=None,
                   help="Limit configs processed per file (for debugging)")
cli_parser.add_argument("--test-canonical", action="store_true",
                   help="Load tool+obj at origin with identity pose, record video, and exit")
AppLauncher.add_app_launcher_args(cli_parser)
args = cli_parser.parse_args()

# Launch with headless + offscreen_render when recording video
app_launcher = AppLauncher(args)
_app = app_launcher.app
# ────────────────────────────────────────────────────────────────────────────

from pathlib import Path

import imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyR

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

# ── Constants ────────────────────────────────────────────────────────────────

ENV_SPACING = 1.0          # metres between env origins; objects are small (< 20 cm)
SIM_DT      = 1.0 / 60.0  # physics time step
GRAVITY     = (0.0, 0.0, -9.81)

# ── Helpers ──────────────────────────────────────────────────────────────────

def mat3_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation matrix → quaternion (w, x, y, z) as used by IsaacLab."""
    q_xyzw = ScipyR.from_matrix(R).as_quat()          # scipy: x, y, z, w
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def quat_wxyz_to_mat3(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) → (3,3) rotation matrix."""
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    return ScipyR.from_quat(q_xyzw).as_matrix()


def derive_usd_paths(tool_obj_path: str, object_obj_path: str):
    """Derive USD asset paths from the OBJ paths stored in the .pt file."""
    tool_p  = Path(tool_obj_path)
    obj_p   = Path(object_obj_path)

    tool_name    = tool_p.stem
    tool_usd_dir = tool_p.parent.parent / "objects_usd" / tool_name
    tool_usd     = tool_usd_dir / f"{tool_name}.usd"

    obj_stem     = obj_p.stem
    obj_usd_dir  = obj_p.parent.parent / "coacd_usd" / obj_stem
    obj_usd      = obj_usd_dir / f"{obj_stem}.usd"

    if not tool_usd.exists():
        raise FileNotFoundError(f"Tool USD not found: {tool_usd}")
    if not obj_usd.exists():
        raise FileNotFoundError(f"Object USD not found: {obj_usd}")

    return str(tool_usd), str(obj_usd)


# ── Scene builder ─────────────────────────────────────────────────────────────

def _compute_env_origins(num_envs: int, spacing: float) -> np.ndarray:
    """Compute grid-based env origins (matching Isaac Lab's layout)."""
    cols = int(np.ceil(np.sqrt(num_envs)))
    origins = np.zeros((num_envs, 3))
    for i in range(num_envs):
        row, col = divmod(i, cols)
        origins[i, 0] = col * spacing
        origins[i, 1] = row * spacing
    return origins


def build_scene(
    tool_usd: str,
    tool_scale: float,
    obj_usd: str,
    obj_scale: float,
    num_envs: int,
    record_video: str | None = None,
) -> tuple:
    """Create SimulationContext + scene with N envs using AddReference.

    Returns (sim_ctx, stage, env_origins_np, tool_scale, obj_scale, rep_annotator).
    """
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, gravity=GRAVITY)
    sim_ctx = SimulationContext(sim_cfg)
    sim_ctx.set_camera_view(eye=[3, 3, 3], target=[0, 0, 0])

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdLux, UsdShade
    stage = sim_utils.get_current_stage()

    # ── Ground plane (visible mesh + physics collision) ───────────────────
    ground = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
    e = 50.0
    ground.CreatePointsAttr().Set([
        Gf.Vec3f(-e, -e, 0), Gf.Vec3f(e, -e, 0),
        Gf.Vec3f(e, e, 0),   Gf.Vec3f(-e, e, 0),
    ])
    ground.CreateFaceVertexCountsAttr().Set([4])
    ground.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    ground.CreateSubdivisionSchemeAttr().Set("none")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    gnd_mat = UsdShade.Material.Define(stage, "/World/Materials/Ground")
    gnd_shd = UsdShade.Shader.Define(stage, "/World/Materials/Ground/Shader")
    gnd_shd.CreateIdAttr("UsdPreviewSurface")
    gnd_shd.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.78, 0.78, 0.74))
    gnd_shd.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    gnd_mat.CreateSurfaceOutput().ConnectToSource(gnd_shd.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(gnd_mat)

    # ── Lighting (dome + key light, matching view_tools.py) ───────────────
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(800.0)
    dome.CreateColorAttr((0.95, 0.96, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    sun.CreateIntensityAttr(2200.0)
    sun.CreateAngleAttr(0.45)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3f(-45.0, 0.0, 35.0))

    # ── Spawn envs with AddReference (like view_tools.py) ─────────────────
    env_origins = _compute_env_origins(num_envs, ENV_SPACING)
    UsdGeom.Xform.Define(stage, "/World/envs")
    ts, os_ = tool_scale, obj_scale

    for i in range(num_envs):
        env_path = f"/World/envs/env_{i}"
        UsdGeom.Xform.Define(stage, env_path)

        # Tool (kinematic rigid body)
        tool_prim = UsdGeom.Xform.Define(stage, f"{env_path}/Tool").GetPrim()
        tool_prim.GetReferences().AddReference(tool_usd)
        txf = UsdGeom.Xformable(tool_prim)
        txf.ClearXformOpOrder()
        txf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(float(env_origins[i, 0]), float(env_origins[i, 1]), 1.0))
        txf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(ts, ts, ts))
        UsdPhysics.RigidBodyAPI.Apply(tool_prim)
        tool_prim.GetAttribute("physics:kinematicEnabled").Set(True)
        for ch in tool_prim.GetAllChildren():
            if ch.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(ch)
                UsdPhysics.MeshCollisionAPI.Apply(ch)

        # Object (dynamic rigid body)
        obj_prim = UsdGeom.Xform.Define(stage, f"{env_path}/Object").GetPrim()
        obj_prim.GetReferences().AddReference(obj_usd)
        oxf = UsdGeom.Xformable(obj_prim)
        oxf.ClearXformOpOrder()
        oxf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(float(env_origins[i, 0]), float(env_origins[i, 1]), 0.1))
        oxf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(os_, os_, os_))
        UsdPhysics.RigidBodyAPI.Apply(obj_prim)
        for ch in obj_prim.GetAllChildren():
            if ch.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(ch)
                UsdPhysics.MeshCollisionAPI.Apply(ch)

    sim_ctx.reset()

    # ── Replicator camera (only when recording) ──────────────────────────
    rep_annotator = None
    if record_video:
        import omni.replicator.core as rep
        e0 = env_origins[0]
        cam_pos  = (float(e0[0]) + 0.4, float(e0[1]) - 0.4, 0.35)
        cam_look = (float(e0[0]), float(e0[1]), 0.05)
        rep_cam = rep.create.camera(
            position=cam_pos, look_at=cam_look,
            focal_length=24.0, clipping_range=(0.01, 1000.0))
        render_prod = rep.create.render_product(rep_cam, (1280, 720))
        rep_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rep_annotator.attach(render_prod)
        print(f"  [Video] Perspective camera at env 0")

    return sim_ctx, stage, env_origins, tool_scale, obj_scale, rep_annotator


def _set_prim_pose(stage, prim_path, pos, quat_wxyz, scale):
    """Set translate + orient + scale on a prim via USD xform ops."""
    from pxr import Gf, UsdGeom
    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(scale), float(scale), float(scale)))


def _get_prim_world_pose(stage, prim_path):
    """Read world position and quaternion (w,x,y,z) from a prim."""
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    world_tf = xf.ComputeLocalToWorldTransform(0)
    t = world_tf.ExtractTranslation()
    rot = world_tf.ExtractRotation()
    q = rot.GetQuat()
    imag = q.GetImaginary()
    pos = np.array([t[0], t[1], t[2]])
    quat = np.array([q.GetReal(), imag[0], imag[1], imag[2]])
    return pos, quat


def run_batch(
    sim_ctx,
    stage,
    num_envs: int,
    tool_scale: float,
    obj_scale: float,
    tool_pos:   np.ndarray,   # (num_envs, 3)
    tool_quat:  np.ndarray,   # (num_envs, 4) w,x,y,z
    obj_pos:    np.ndarray,   # (num_envs, 3)
    obj_quat:   np.ndarray,   # (num_envs, 4) w,x,y,z
    settle_steps: int,
    camera = None,
    capture_every: int = 4,
) -> tuple:
    """Reset env poses via USD xform ops, step physics, return final state."""

    for i in range(num_envs):
        _set_prim_pose(stage, f"/World/envs/env_{i}/Tool",
                       tool_pos[i], tool_quat[i], tool_scale)
        _set_prim_pose(stage, f"/World/envs/env_{i}/Object",
                       obj_pos[i], obj_quat[i], obj_scale)

    # Readback env 0 for debug
    t_rb, _ = _get_prim_world_pose(stage, "/World/envs/env_0/Tool")
    o_rb, _ = _get_prim_world_pose(stage, "/World/envs/env_0/Object")
    print(f"    [READBACK] tool_pos = {t_rb}  obj_pos = {o_rb}")

    frames = []
    # Capture t=0 frame
    if camera is not None:
        sim_ctx.render()
        rgba = camera.get_data()
        if rgba is not None and rgba.size > 0 and rgba.max() > 0:
            frames.append(rgba[:, :, :3])

    for step in range(settle_steps):
        sim_ctx.step()
        if camera is not None and step % capture_every == 0:
            sim_ctx.render()
            rgba = camera.get_data()
            if rgba is not None and rgba.size > 0 and rgba.max() > 0:
                frames.append(rgba[:, :, :3])

    # Read final object poses
    pos_final  = np.zeros((num_envs, 3))
    quat_final = np.zeros((num_envs, 4))
    for i in range(num_envs):
        pos_final[i], quat_final[i] = _get_prim_world_pose(
            stage, f"/World/envs/env_{i}/Object")
    return pos_final, quat_final, frames


# ── Per-file validation ───────────────────────────────────────────────────────

def validate_file(pt_path: Path, out_path: Path, args) -> None:
    """Filter one .pt file through physics simulation."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    # ── Load shared fields ────────────────────────────────────────────────────
    tool_translations = data["tool_translations"].numpy()   # (N, 3)
    tool_rotations    = data["tool_rotations"].numpy()      # (N, 3, 3)
    contact_pt_tool   = data["contact_pt_tool"].numpy()     # (N, 3) canonical
    contact_pt_obj    = data["contact_pt_obj"].numpy()      # (N, 3) world
    obj_centroid      = data["obj_centroid"].numpy()        # (3,)
    R_obj             = data["object_rotation"].numpy()     # (3, 3)
    tool_scale        = float(data["tool_scale"])
    obj_scale         = float(data["object_scale"])
    # tool_centroid_raw: mesh-frame centroid offset.  tool_translations stores
    # world-frame *centroid* positions, but Isaac Sim needs *prim origin*.
    # Prim origin = centroid - R @ centroid_raw
    tool_centroid_raw = data["tool_centroid_raw"].numpy()  # (3,)
    # obj_z_shift: in generation, mesh was grounded by `verts_z -= z_shift`.
    # In Isaac Sim the equivalent is setting prim z = -z_shift.
    obj_z_shift = float(data["obj_z_shift"])
    N = tool_translations.shape[0]

    # Contact point in object-local frame (used after settling to track moved obj)
    # contact_pt_obj is in world frame; local = R_obj.T @ (pt - obj_centroid)
    pt_obj_local = (R_obj.T @ (contact_pt_obj - obj_centroid).T).T   # (N, 3)

    # Tool contact point in world frame (tool is static → constant)
    # contact_pt_tool is centroid-subtracted canonical; world = R_tool @ pt + t_trans
    # (R_tool @ pt) + tool_translations  where pt is already in centroid frame
    pt_tool_world = np.einsum("nij,nj->ni", tool_rotations, contact_pt_tool) + tool_translations  # (N,3)

    # ── Build Isaac Sim scene ─────────────────────────────────────────────────
    print(f"\n[INFO] Building scene for {pt_path.name}  (N={N}, envs={args.num_envs})")
    tool_usd, obj_usd = derive_usd_paths(data["tool_mesh_path"], data["object_mesh_path"])
    print(f"  Tool USD  : {tool_usd}  (scale={tool_scale})")
    print(f"  Object USD: {obj_usd}  (scale={obj_scale})")

    sim_ctx, stage, scene_env_origins, ts, os_, cam = build_scene(
        tool_usd, tool_scale, obj_usd, obj_scale, args.num_envs,
        record_video=getattr(args, "record_video", None),
    )
    # scene_env_origins: (num_envs, 3) actual XY grid positions from Isaac Lab.
    # The validation loop uses these to correctly position each object/tool
    # relative to its environment, matching the 2D grid Isaac Lab uses.

    # Pre-compute object quaternion (same R_obj for all configs in this file)
    obj_quat_np = mat3_to_quat_wxyz(R_obj)   # (4,) w,x,y,z

    valid_indices   = []
    valid_obj_pos   = []
    valid_obj_quat  = []
    all_frames      = []      # collected only when --record-video is set

    # Optionally limit configs for debugging
    N_proc = min(N, args.max_configs) if args.max_configs else N

    # Process in chunks of num_envs
    for start in range(0, N_proc, args.num_envs):
        end  = min(start + args.num_envs, N)
        E    = end - start
        idxs = np.arange(start, end)

        # Lateral offset: shift generation frame so obj centroid XY aligns with env_origin.
        # Z uses -obj_z_shift to ground the mesh (generation does `verts_z -= z_shift`).
        xy_offset = np.zeros((args.num_envs, 3))
        xy_offset[:, 0] = scene_env_origins[:, 0] - obj_centroid[0]
        xy_offset[:, 1] = scene_env_origins[:, 1] - obj_centroid[1]
        # z column stays 0 — z grounding handled separately below

        obj_pos_np = np.zeros((args.num_envs, 3))
        for i, ci in enumerate(idxs):
            obj_pos_np[i] = xy_offset[i]
            obj_pos_np[i, 2] = -obj_z_shift   # ground the mesh (prim z = -z_shift)
        for i in range(E, args.num_envs):
            obj_pos_np[i] = obj_pos_np[0]

        obj_quat_batch = np.tile(obj_quat_np, (args.num_envs, 1))   # (num_envs, 4)

        tool_pos_np = np.zeros((args.num_envs, 3))
        for i, ci in enumerate(idxs):
            # tool_translations[ci] is the world-frame CENTROID position.
            # Isaac Sim expects PRIM ORIGIN: origin = centroid - R @ centroid_raw
            R_t = tool_rotations[ci]
            tool_prim_origin = tool_translations[ci] - R_t @ tool_centroid_raw
            tool_pos_np[i] = tool_prim_origin + xy_offset[i]
        for i in range(E, args.num_envs):
            tool_pos_np[i] = tool_pos_np[0]

        tool_quat_np = np.stack([mat3_to_quat_wxyz(tool_rotations[ci]) for ci in idxs])
        if E < args.num_envs:
            pad = np.tile(tool_quat_np[0:1], (args.num_envs - E, 1))
            tool_quat_np = np.concatenate([tool_quat_np, pad], axis=0)

        # ── Diagnostic: print initial separation for env 0 (first batch only) ─
        if start == 0:
            ci0 = idxs[0]
            o0  = obj_pos_np[0]
            t0  = tool_pos_np[0]
            dist0 = float(np.linalg.norm(t0 - o0))
            pt_tw = pt_tool_world[ci0]      # R_tool @ pt_canonical + t_tool (gen frame)
            pt_ow = contact_pt_obj[ci0]     # contact pt on obj (gen frame)
            cp_dist = float(np.linalg.norm(pt_tw - pt_ow))
            print(f"  [DEBUG batch 0 env 0]")
            print(f"    obj_centroid           = {obj_centroid}")
            print(f"    obj_z_shift            = {obj_z_shift:.6f}")
            print(f"    xy_offset[0]           = {xy_offset[0]}")
            print(f"    obj_pos[0]  (z=-zs)    = {o0}")
            print(f"    tool_pos[0] (prim)     = {t0}")
            print(f"    prim-to-prim dist      = {dist0:.4f} m")
            print(f"    pt_tool_world[0]       = {pt_tw}")
            print(f"    contact_pt_obj[0]      = {pt_ow}")
            print(f"    contact-pt distance    = {cp_dist:.6f} m "
                  f"(~0 means surfaces touching in gen data)")

        pos_f_np, quat_f_np, frames = run_batch(
            sim_ctx, stage, args.num_envs, ts, os_,
            tool_pos_np, tool_quat_np, obj_pos_np, obj_quat_batch,
            args.settle_steps,
            camera=cam, capture_every=getattr(args, "capture_every", 4),
        )
        all_frames.extend(frames)

        # ── Contact check ─────────────────────────────────────────────────────
        for i, ci in enumerate(idxs):
            R_new  = quat_wxyz_to_mat3(quat_f_np[i])
            p_new  = pos_f_np[i]
            # Object contact pt in new world frame
            pt_contact_new = R_new @ pt_obj_local[ci] + p_new      # (3,)
            # Tool contact pt: apply same XY-only offset as object pose.
            pt_tool_sim = pt_tool_world[ci] + xy_offset[i]   # (3,)
            dist = np.linalg.norm(pt_contact_new - pt_tool_sim)
            if dist < args.threshold:
                valid_indices.append(ci)
                valid_obj_pos.append(p_new)
                valid_obj_quat.append(quat_f_np[i])

        n_kept = sum(1 for ci in idxs if ci in set(valid_indices))
        print(f"  Batch [{start}:{end}]: {n_kept} / {E} kept  "
              f"(total so far: {len(valid_indices)}/{end})")

    print(f"  → {len(valid_indices)} / {N} configs passed  "
          f"({100*len(valid_indices)/max(N,1):.1f}%)")

    # ── Write video (always, even if no configs survived) ─────────────────
    rv = getattr(args, "record_video", None)
    if rv and all_frames:
        vid_path = Path(rv)
        vid_path.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, int(1.0 / SIM_DT / getattr(args, "capture_every", 4)))
        imageio.mimwrite(str(vid_path), all_frames, fps=fps, quality=8)
        print(f"  ✓ Video  → {vid_path}  ({len(all_frames)} frames @ {fps} fps)")
    elif rv:
        print(f"  ⚠ No frames captured — video not written")

    if not valid_indices:
        print("  ⚠  No valid configs — skipping output.")
        return

    # ── Build filtered output ─────────────────────────────────────────────────
    vi = valid_indices
    filtered = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == N:
            filtered[k] = v[vi]       # slice per-config tensors
        else:
            filtered[k] = v           # keep shared tensors / scalars as-is

    # Add settled object pose
    filtered["settled_obj_pos"]  = torch.tensor(np.array(valid_obj_pos),  dtype=torch.float32)
    filtered["settled_obj_quat"] = torch.tensor(np.array(valid_obj_quat), dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(filtered, out_path)
    print(f"  ✓ Saved  → {out_path}")


# ── Canonical test ────────────────────────────────────────────────────────────

def run_canonical_test(pt_path: Path, video_path: str):
    """Load tool+obj at origin with identity pose — pure visibility test."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    tool_scale = float(data["tool_scale"])
    obj_scale  = float(data["object_scale"])
    tool_usd, obj_usd = derive_usd_paths(data["tool_mesh_path"], data["object_mesh_path"])
    print(f"\n[CANONICAL TEST]")
    print(f"  Tool USD : {tool_usd}  (scale={tool_scale})")
    print(f"  Obj  USD : {obj_usd}  (scale={obj_scale})")

    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, gravity=GRAVITY)
    sim_ctx = SimulationContext(sim_cfg)
    sim_ctx.set_camera_view(eye=[0.4, -0.4, 0.35], target=[0, 0, 0.05])

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdLux, UsdShade
    stage = sim_utils.get_current_stage()

    # Ground
    ground = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
    e = 5.0
    ground.CreatePointsAttr().Set([
        Gf.Vec3f(-e, -e, 0), Gf.Vec3f(e, -e, 0),
        Gf.Vec3f(e, e, 0),   Gf.Vec3f(-e, e, 0)])
    ground.CreateFaceVertexCountsAttr().Set([4])
    ground.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    ground.CreateSubdivisionSchemeAttr().Set("none")
    gnd_mat = UsdShade.Material.Define(stage, "/World/Materials/Ground")
    gnd_shd = UsdShade.Shader.Define(stage, "/World/Materials/Ground/Shader")
    gnd_shd.CreateIdAttr("UsdPreviewSurface")
    gnd_shd.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.78, 0.78, 0.74))
    gnd_shd.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    gnd_mat.CreateSurfaceOutput().ConnectToSource(gnd_shd.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(gnd_mat)

    # Lighting
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(800.0)
    dome.CreateColorAttr((0.95, 0.96, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    sun.CreateIntensityAttr(2200.0)
    sun.CreateAngleAttr(0.45)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3f(-45.0, 0.0, 35.0))

    # Tool at origin — NO physics, just visual
    tool_prim = UsdGeom.Xform.Define(stage, "/World/CanonicalTool").GetPrim()
    tool_prim.GetReferences().AddReference(tool_usd)
    txf = UsdGeom.Xformable(tool_prim)
    txf.ClearXformOpOrder()
    txf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
    txf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(tool_scale, tool_scale, tool_scale))

    # Object slightly offset — NO physics, just visual
    obj_prim = UsdGeom.Xform.Define(stage, "/World/CanonicalObject").GetPrim()
    obj_prim.GetReferences().AddReference(obj_usd)
    oxf = UsdGeom.Xformable(obj_prim)
    oxf.ClearXformOpOrder()
    oxf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.15, 0, 0))
    oxf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(obj_scale, obj_scale, obj_scale))

    sim_ctx.reset()

    # Print prim tree
    for name, p in [("Tool", tool_prim), ("Object", obj_prim)]:
        print(f"\n  Prim tree ({name}):")
        def _walk(prim, depth=0):
            print(f"    {'  '*depth}{prim.GetPath()}  type={prim.GetTypeName()}")
            for c in prim.GetChildren():
                _walk(c, depth+1)
        _walk(p)

    # Camera
    import omni.replicator.core as rep
    rep_cam = rep.create.camera(
        position=(0.35, -0.35, 0.25),
        look_at=(0.05, 0.0, 0.05),
        focal_length=24.0,
        clipping_range=(0.01, 100.0))
    render_prod = rep.create.render_product(rep_cam, (1280, 720))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_prod)

    # Warm up + capture
    for _ in range(10):
        sim_ctx.render()

    frames = []
    for _ in range(60):
        sim_ctx.render()
        rgba = annotator.get_data()
        if rgba is not None and rgba.size > 0 and rgba.max() > 0:
            frames.append(rgba[:, :, :3])

    if frames:
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=30, quality=8)
        print(f"\n✓ Canonical test video → {video_path}  ({len(frames)} frames)")
    else:
        print("\n⚠ No frames captured!")

    import os; os._exit(0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    # args already parsed at module level (before AppLauncher bootstrap)

    # Canonical test mode — just load assets and render, no physics
    if args.test_canonical:
        pt_path = Path(args.input[0])
        vid = args.record_video or "canonical_test.mp4"
        run_canonical_test(pt_path, vid)
        return  # won't reach here (os._exit in run_canonical_test)

    out_root = Path(args.output)

    for pt_path_str in args.input:
        pt_path = Path(pt_path_str)
        if not pt_path.exists():
            print(f"[WARN] Not found: {pt_path}")
            continue
        out_path = out_root / pt_path.name
        try:
            validate_file(pt_path, out_path, args)
        except Exception as e:
            print(f"[ERROR] {pt_path.name}: {e}")
            import traceback; traceback.print_exc()

    # Cleanly stop replicator (with timeout) then hard-exit to avoid
    # Isaac Sim background threads hanging the process indefinitely.
    import threading, os
    def _stop_rep():
        try:
            import omni.replicator.core as rep
            rep.orchestrator.stop()
        except Exception:
            pass
    if getattr(args, "record_video", None):
        t = threading.Thread(target=_stop_rep, daemon=True)
        t.start()
        t.join(timeout=5.0)
    os._exit(0)


if __name__ == "__main__":
    main()
