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
        --num-envs 64 --settle-steps 200 --threshold 0.008

Usage (glob — shell expands):
    python pretrain/validate_contact_physics.py \\
        --input  pretrain/new_pretrain/tmp_data/**/*.pt \\
        --output pretrain/new_pretrain/validated/ \\
        --num-envs 128

Note: one Isaac Sim instance is created per .pt file (different tool/object
pairs require different USD assets in the scene).
"""

# ── Isaac Sim MUST be launched before any omni/isaacsim imports ─────────────
import sys
from isaacsim import SimulationApp
# Pre-parse --record-video so we can pass enable_cameras before full argparse
_record_video = next((sys.argv[i+1] for i, a in enumerate(sys.argv)
                      if a == "--record-video" and i+1 < len(sys.argv)), None)
_app_cfg = {"headless": True, "anti_aliasing": 0}
if _record_video:
    _app_cfg["enable_cameras"] = True
_app = SimulationApp(_app_cfg)
# ────────────────────────────────────────────────────────────────────────────

import argparse
import math
from pathlib import Path

import imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyR

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils.configclass import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg

# ── Patch: Isaac Sim installations sometimes lack rendering preset .kit files
#    (e.g. balanced.kit).  Since we run headless physics-only, skip silently.
_orig_render_cfg = SimulationContext._apply_render_settings_from_cfg
def _safe_render_cfg(self):
    try:
        _orig_render_cfg(self)
    except (FileNotFoundError, OSError):
        pass  # no rendering preset available; physics still works fine
SimulationContext._apply_render_settings_from_cfg = _safe_render_cfg

# ── Constants ────────────────────────────────────────────────────────────────

ENV_SPACING = 2.0          # metres between env origins (must exceed object size)
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

def build_scene(
    tool_usd: str,
    tool_scale: float,
    obj_usd: str,
    obj_scale: float,
    num_envs: int,
    record_video: str | None = None,
) -> tuple:
    """Create SimulationContext + scene with N envs.

    Returns (sim_ctx, scene, tool_obj, object_obj, camera_or_None).
    """
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, gravity=GRAVITY)
    sim_ctx = SimulationContext(sim_cfg)
    sim_ctx.set_camera_view(eye=[3, 3, 3], target=[0, 0, 0])

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7, dynamic_friction=0.7, restitution=0.0
            ),
        )
        # Kinematic tool — gravity disabled, does not move
        tool = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tool",
            spawn=sim_utils.UsdFileCfg(
                usd_path=tool_usd,
                scale=(tool_scale, tool_scale, tool_scale),
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 1)),
        )
        # Dynamic object — gravity on
        object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=obj_usd,
                scale=(obj_scale, obj_scale, obj_scale),
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=5.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0.1)),
        )
        # Optional camera watching env 0 (prim_path NOT in ENV_REGEX_NS)
        if record_video:
            camera = CameraCfg(
                prim_path="/World/RecordCam",
                update_period=0,       # update every physics step
                height=720,
                width=1280,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1000.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.6, -0.6, 0.6),
                    rot=(0.693, 0.430, -0.430, -0.360),  # look roughly toward (0,0,0)
                    convention="world",
                ),
            )

    scene_cfg = SceneCfg(num_envs=num_envs, env_spacing=ENV_SPACING)
    scene    = InteractiveScene(scene_cfg)

    sim_ctx.reset()
    scene.reset()

    tool_obj   = scene["tool"]
    object_obj = scene["object"]
    cam        = scene["camera"] if record_video else None
    return sim_ctx, scene, tool_obj, object_obj, cam


# ── Batch simulation ──────────────────────────────────────────────────────────

def run_batch(
    sim_ctx,
    scene,
    tool_obj:   RigidObject,
    object_obj: RigidObject,
    tool_pos:   torch.Tensor,   # (E, 3)
    tool_quat:  torch.Tensor,   # (E, 4) w,x,y,z
    obj_pos:    torch.Tensor,   # (E, 3)
    obj_quat:   torch.Tensor,   # (E, 4) w,x,y,z
    settle_steps: int,
    device: str,
    camera = None,              # Camera sensor or None
    capture_every: int = 4,     # capture one frame every N steps
) -> tuple:
    """Reset env poses, step physics, return (final_pos, final_quat, frames).

    frames is a list of (H,W,3) uint8 RGB arrays; empty when camera is None.
    """
    E = tool_pos.shape[0]
    zeros3 = torch.zeros(E, 3, device=device)

    tool_state = torch.cat([tool_pos, tool_quat, zeros3, zeros3], dim=-1)
    tool_obj.write_root_state_to_sim(tool_state)

    obj_state = torch.cat([obj_pos, obj_quat, zeros3, zeros3], dim=-1)
    object_obj.write_root_state_to_sim(obj_state)

    scene.write_data_to_sim()

    frames = []
    for step in range(settle_steps):
        sim_ctx.step()
        scene.update(SIM_DT)
        if camera is not None and step % capture_every == 0:
            camera.update(dt=SIM_DT)
            rgb = camera.data.output["rgb"]          # (1, H, W, 4) RGBA uint8
            frames.append(rgb[0, :, :, :3].cpu().numpy())  # (H, W, 3)

    pos_final  = object_obj.data.root_pos_w.clone()
    quat_final = object_obj.data.root_quat_w.clone()
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
    print(f"  Tool USD  : {tool_usd}")
    print(f"  Object USD: {obj_usd}")

    sim_ctx, scene, tool_obj, object_obj, cam = build_scene(
        tool_usd, tool_scale, obj_usd, obj_scale, args.num_envs,
        record_video=getattr(args, "record_video", None),
    )
    device = sim_ctx.device

    # Pre-compute object quaternion (same R_obj for all configs in this file)
    obj_quat_np = mat3_to_quat_wxyz(R_obj)   # (4,) w,x,y,z

    valid_indices   = []
    valid_obj_pos   = []
    valid_obj_quat  = []
    all_frames      = []      # collected only when --record-video is set

    # Process in chunks of num_envs
    for start in range(0, N, args.num_envs):
        end  = min(start + args.num_envs, N)
        E    = end - start
        idxs = np.arange(start, end)

        # Env origins for this chunk: (i * ENV_SPACING, 0, 0)
        env_origins = np.zeros((args.num_envs, 3))
        for i in range(args.num_envs):
            env_origins[i, 0] = i * ENV_SPACING

        # Object pose: env_origin + (0, 0, obj_centroid.z)
        # We translate the whole scene so obj_centroid_xy → env_origin_xy
        obj_pos_np = np.zeros((args.num_envs, 3))
        for i, ci in enumerate(idxs):
            obj_pos_np[i] = env_origins[i] + np.array([
                -obj_centroid[0], -obj_centroid[1], obj_centroid[2]
            ])
        # Pad last batch with first env pose if under-filled
        for i in range(E, args.num_envs):
            obj_pos_np[i] = obj_pos_np[0]

        obj_quat_batch = np.tile(obj_quat_np, (args.num_envs, 1))   # (E, 4)

        # Tool pose: original tool_translation shifted by same offset as object
        tool_pos_np = np.zeros((args.num_envs, 3))
        for i, ci in enumerate(idxs):
            offset = env_origins[i] + np.array([-obj_centroid[0], -obj_centroid[1], 0.0])
            tool_pos_np[i] = tool_translations[ci] + offset
        for i in range(E, args.num_envs):
            tool_pos_np[i] = tool_pos_np[0]

        tool_quat_np = np.stack([mat3_to_quat_wxyz(tool_rotations[ci]) for ci in idxs])
        # pad
        if E < args.num_envs:
            pad = np.tile(tool_quat_np[0:1], (args.num_envs - E, 1))
            tool_quat_np = np.concatenate([tool_quat_np, pad], axis=0)

        # To tensors
        tp = torch.tensor(tool_pos_np,  dtype=torch.float32, device=device)
        tq = torch.tensor(tool_quat_np, dtype=torch.float32, device=device)
        op = torch.tensor(obj_pos_np,   dtype=torch.float32, device=device)
        oq = torch.tensor(obj_quat_batch, dtype=torch.float32, device=device)

        pos_f, quat_f, frames = run_batch(
            sim_ctx, scene, tool_obj, object_obj,
            tp, tq, op, oq, args.settle_steps, device,
            camera=cam, capture_every=getattr(args, "capture_every", 4),
        )
        all_frames.extend(frames)

        pos_f_np  = pos_f.cpu().numpy()    # (num_envs, 3)
        quat_f_np = quat_f.cpu().numpy()   # (num_envs, 4)

        # ── Contact check ─────────────────────────────────────────────────────
        for i, ci in enumerate(idxs):
            R_new  = quat_wxyz_to_mat3(quat_f_np[i])
            p_new  = pos_f_np[i]
            # Object contact pt in new world frame
            pt_contact_new = R_new @ pt_obj_local[ci] + p_new      # (3,)
            # Tool contact pt (static, already in shifted sim frame)
            offset = env_origins[i] + np.array([-obj_centroid[0], -obj_centroid[1], 0.0])
            pt_tool_sim = pt_tool_world[ci] + offset                # (3,)
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

    if not valid_indices:
        print("  ⚠  No valid configs — skipping output.")
        return

    # ── Build filtered output ─────────────────────────────────────────────────
    vi = valid_indices
    filtered = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == N:
            filtered[k] = v[vi]       # slice per-config tensors
        else:
            filtered[k] = v           # keep shared tensors / scalars as-is

    # Add settled object pose
    filtered["settled_obj_pos"]  = torch.tensor(np.array(valid_obj_pos),  dtype=torch.float32)
    filtered["settled_obj_quat"] = torch.tensor(np.array(valid_obj_quat), dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(filtered, out_path)
    print(f"  ✓ Saved  → {out_path}")

    # ── Write video ───────────────────────────────────────────────────────────
    rv = getattr(args, "record_video", None)
    if rv and all_frames:
        vid_path = Path(rv)
        vid_path.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, int(1.0 / SIM_DT / getattr(args, "capture_every", 4)))
        imageio.mimwrite(str(vid_path), all_frames, fps=fps, quality=8)
        print(f"  ✓ Video  → {vid_path}  ({len(all_frames)} frames @ {fps} fps)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Physics-based contact stability filter")
    p.add_argument("--input",  nargs="+", required=True, help=".pt file(s) to validate")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--num-envs",      type=int,   default=2048,    help="Parallel envs per batch")
    p.add_argument("--settle-steps",  type=int,   default=200,   help="Physics steps to settle")
    p.add_argument("--threshold",     type=float, default=0.002, help="Contact distance threshold (m)")
    p.add_argument("--record-video",  default=None, metavar="PATH",
                   help="If given, record a video of env-0 settling to this .mp4 path")
    p.add_argument("--capture-every", type=int,   default=4,
                   help="Capture one frame every N physics steps (default: 4)")
    args = p.parse_args()

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

    _app.close()


if __name__ == "__main__":
    main()
