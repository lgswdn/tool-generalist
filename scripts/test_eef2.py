import argparse
import glob
import math
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test EEF environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video and save it.")
parser.add_argument("--video_length", type=int, default=300, help="Length of recorded video (in steps).")
parser.add_argument("--video_fps", type=int, default=30, help="FPS of the recorded video.")
parser.add_argument(
    "--robot_usd_dir",
    type=str,
    default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "robot_usd")),
    help="Directory containing robot USD files. All '*.usd' files in this directory will be used.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from IsaacLab_nonPrehensile.robots.franka import FRANKA_PANDA_TOOL_HIGH_PD_CFG

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path

# Local mirror path for Franka asset (optional). If missing, fallback to IsaacLab Nucleus URL.
# LOCAL_FRANKA_USD_PATH = "/mnt/afs/wangyuze/ToolGeneralist/static/franka/Robots/FrankaEmika/panda_instanceable.usd"
# REMOTE_FRANKA_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
# FRANKA_USD_PATH_TO_USE = LOCAL_FRANKA_USD_PATH if os.path.isfile(LOCAL_FRANKA_USD_PATH) else REMOTE_FRANKA_USD_PATH


def collect_robot_usd_paths(usd_dir: str) -> list[str]:
    """Collect and sort all USD files in a directory (non-recursive)."""
    usd_dir = os.path.abspath(usd_dir)
    if not os.path.isdir(usd_dir):
        raise FileNotFoundError(f"Robot USD directory does not exist: {usd_dir}")

    usd_paths = sorted(
        p
        for p in glob.glob(os.path.join(usd_dir, "*.usd"))
        if os.path.isfile(p) and os.path.basename(p) != "panda_instanceable.usd"
    )
    if len(usd_paths) == 0:
        raise FileNotFoundError(f"No USD files found in directory: {usd_dir}")
    return usd_paths


def build_multi_usd_robot_cfg(usd_paths: list[str]):
    """Create a robot cfg that cycles through USD files across envs in order."""
    robot_cfg = FRANKA_PANDA_TOOL_HIGH_PD_CFG.copy()
    base_spawn_cfg = robot_cfg.spawn

    robot_cfg.spawn = sim_utils.MultiUsdFileCfg(
        usd_path=usd_paths,
        random_choice=False,
        activate_contact_sensors=base_spawn_cfg.activate_contact_sensors,
        rigid_props=base_spawn_cfg.rigid_props,
        articulation_props=base_spawn_cfg.articulation_props,
        collision_props=base_spawn_cfg.collision_props,
        mass_props=base_spawn_cfg.mass_props,
        visual_material=base_spawn_cfg.visual_material,
        semantic_tags=base_spawn_cfg.semantic_tags,
    )
    return robot_cfg

def load_objects():
    usd_cfg = sim_utils.UsdFileCfg(
        usd_path='/mnt/afs/wangyuze/ToolGeneralist/static/000_robotic_fork_effector_var_000/000_robotic_fork_effector_var_000.usd',
        scale=(0.1, 0.1, 0.1),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    )
    return [usd_cfg]

@configclass
class NonPrehensileSceneCfg(InteractiveSceneCfg):
    replicate_physics: bool = False

    # Camera used for optional video recording
    record_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RecordCamera",
        update_period=0.0,
        height=1080,
        width=1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        # Diagonal view from above, looking toward the env origin.
        offset=CameraCfg.OffsetCfg(
            pos=(8.0, 8.0, 8.0),
            # yaw right by ~45 deg from previous diagonal view
            rot=(-0.3251, 0.6280, 0.6280, -0.3251),
            convention="ros",
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        debug_vis=True,
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    robot = FRANKA_PANDA_TOOL_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/robot",
    )

    # obj = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/obj",
    #     spawn=sim_utils.MultiAssetSpawnerCfg(
    #         assets_cfg=load_objects(),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.5, 0.0, 0.5),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     )
    # )

@configclass
class ActionsCfg:
    arm_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.1,
        use_zero_offset=True,
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg:
        # robot_state = ObsTerm(
        #     func=mdp.robot_state,
        # )
        pass

@configclass
class NonPrehensileEnvCfg(ManagerBasedEnvCfg):
    scene: NonPrehensileSceneCfg = NonPrehensileSceneCfg(num_envs=64, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        self.viewer.eye = (3.0, 3.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 8
        self.episode_length_s = 30

        self.sim.dt = 1 / 80
        self.sim.render_interval = self.decimation

def main():
    # print(f"[INFO] Franka USD path: {FRANKA_USD_PATH_TO_USE}")
    # franka_status = check_file_path(FRANKA_USD_PATH_TO_USE)
    # if franka_status == 1:
    #     print("[INFO] Franka USD source: local file")
    # elif franka_status == 2:
    #     print("[INFO] Franka USD source: remote (Nucleus/S3)")
    # else:
    #     raise FileNotFoundError(f"Franka USD not found: {FRANKA_USD_PATH_TO_USE}")

    usd_paths = collect_robot_usd_paths(args_cli.robot_usd_dir)
    print(f"[INFO] Loaded {len(usd_paths)} robot USD files from: {os.path.abspath(args_cli.robot_usd_dir)}")
    for i, usd_path in enumerate(usd_paths):
        print(f"       [{i:02d}] {usd_path}")

    env_cfg = NonPrehensileEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.robot = build_multi_usd_robot_cfg(usd_paths).replace(prim_path="{ENV_REGEX_NS}/robot")

    print(
        "[INFO] Record camera diagonal-view pose: "
        f"pos={env_cfg.scene.record_camera.offset.pos}, "
        f"rot={env_cfg.scene.record_camera.offset.rot}, "
        f"resolution={env_cfg.scene.record_camera.width}x{env_cfg.scene.record_camera.height}"
    )

    preview_envs = min(args_cli.num_envs, 16)
    print(f"[INFO] Deterministic env->USD assignment preview (first {preview_envs} envs):")
    for env_id in range(preview_envs):
        usd_path = usd_paths[env_id % len(usd_paths)]
        print(f"       env_{env_id:03d} -> {os.path.basename(usd_path)}")

    env = ManagerBasedEnv(cfg=env_cfg)

    # Reset once to initialize internal buffers and sensors.
    env.reset()

    # ── DEBUG: robot articulation info ──
    robot = env.scene["robot"]
    print(f"\n{'='*60}")
    print(f"[DEBUG] Robot type: {type(robot)}")
    print(f"[DEBUG] Num bodies : {robot.num_bodies}")
    print(f"[DEBUG] Body names : {robot.body_names}")
    print(f"[DEBUG] Num joints : {robot.num_joints}")
    print(f"[DEBUG] Joint names: {robot.joint_names}")
    print(f"[DEBUG] Action dim : {env.action_manager.total_action_dim}")
    print(f"[DEBUG] Init joint pos (cfg):")
    for jname, jval in FRANKA_PANDA_TOOL_HIGH_PD_CFG.init_state.joint_pos.items():
        print(f"         {jname}: {jval}")
    print(f"[DEBUG] Actual joint pos after reset (env 0):")
    jp = robot.data.joint_pos[0].cpu().numpy()
    for i, name in enumerate(robot.joint_names):
        print(f"         {name}: {jp[i]:.6f}")
    print(f"[DEBUG] Actual joint vel after reset (env 0):")
    jv = robot.data.joint_vel[0].cpu().numpy()
    for i, name in enumerate(robot.joint_names):
        print(f"         {name}: {jv[i]:.6f}")
    print(f"{'='*60}\n")

    print(ISAACLAB_NUCLEUS_DIR)

    video_writer = None
    frame_count = 0
    video_path = None
    frames_dir = None

    if args_cli.video:
        log_dir = os.path.join(os.getcwd(), "videos", "test_eef", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
        video_path = os.path.join(log_dir, "test_eef.mp4")
        frames_dir = os.path.join(log_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        height = env.scene["record_camera"].cfg.height
        width = env.scene["record_camera"].cfg.width
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, args_cli.video_fps, (width, height))
        print(f"[INFO] Recording video to: {video_path}")
        print(f"[INFO] Saving key frames (PNG) to: {frames_dir}")

    step_count = 0
    while simulation_app.is_running():
        # Sample random actions in [-1, 1] for each env and each action dimension.
        random_action = 2.0 * torch.rand(
            (env.num_envs, env.action_manager.total_action_dim), device=env.device
        ) - 1.0
        env.step(0.5 * random_action)
        step_count += 1

        # Print joint info every 50 steps for first 500 steps
        if step_count <= 500 and step_count % 50 == 0:
            jp = robot.data.joint_pos[0].cpu().numpy()
            jv = robot.data.joint_vel[0].cpu().numpy()
            max_vel = max(abs(jv))
            drift = ""
            print(f"[STEP {step_count:4d}] max|vel|={max_vel:.6f}  pos=[{', '.join(f'{p:.4f}' for p in jp)}]")

        if args_cli.video and video_writer is not None:
            rgb = env.scene["record_camera"].data.output["rgb"][0, ..., :3]
            frame = rgb.detach().cpu().numpy()
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            frame_count += 1

            # Save key frames as PNG (viewable in VSCode)
            if frames_dir is not None and (frame_count == 1 or frame_count % 50 == 0 or frame_count >= args_cli.video_length):
                png_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
                cv2.imwrite(png_path, frame_bgr)
                print(f"[INFO] Saved frame: {png_path}")

            if frame_count >= args_cli.video_length:
                break

    if video_writer is not None:
        video_writer.release()
        print(f"[INFO] Saved video: {video_path}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
