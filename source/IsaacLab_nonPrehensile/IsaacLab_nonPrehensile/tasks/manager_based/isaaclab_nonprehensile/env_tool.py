# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
import json
import yaml
import torch
from collections import deque
import time

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import CurriculumTermCfg as CurriculumTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG, FRANKA_PANDA_CFG
from IsaacLab_nonPrehensile.robots.franka import (
    FRANKA_PANDA_TOOL_HIGH_PD_CFG,
    build_multi_tool_robot_cfg,
)
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, RelativeJointPositionActionCfg, JointVelocityActionCfg, JointEffortActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp
from collections.abc import Sequence

from utils.assets import (
    ToolAssetContractError,
    load_selected_tool_ids,
    load_tool_adjusted_entry,
    load_tool_head_area,
    resolve_tool_mesh_path,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.cloud import Cloud

_CLOUD_CACHE = {}

# Load path configuration. By default this reads paths.yaml at the project
# root; set TOOL_GENERALIST_PATHS_YAML to run against an alternate asset config.
_DEFAULT_PATHS_CFG_FILE = os.path.join(os.path.dirname(__file__), "../" * 6, "paths.yaml")
_PATHS_CFG_FILE = os.environ.get("TOOL_GENERALIST_PATHS_YAML", _DEFAULT_PATHS_CFG_FILE)
_PATHS_CFG_FILE = os.path.abspath(os.path.normpath(_PATHS_CFG_FILE))
print(f"[INFO] Loading path config from {_PATHS_CFG_FILE}")
with open(_PATHS_CFG_FILE, "r") as _f:
    _PATHS = yaml.safe_load(_f)
from scipy.spatial.transform import Rotation as R
import numpy as np

from isaaclab.sensors import FrameTransformerCfg, CameraCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import FRAME_MARKER_SMALL_CFG
from utils.experiment.rl_runtime_spec import load_runtime_spec_from_env, runtime_spec_contract
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.asset_assignment import (
    OBJECT_ASSIGNMENT_SALT,
    TOOL_ASSIGNMENT_SALT,
    asset_indices_for_rank,
    sequential_spawn_indices_for_rank,
)

_RL_RUNTIME_SPEC = load_runtime_spec_from_env()
_RL_CONTRACT = runtime_spec_contract(_RL_RUNTIME_SPEC)
_PHYSICS_OBSERVATION_FIELDS = tuple(_RL_RUNTIME_SPEC["physics_observation_fields"])
_ACTION_DIM = int(_RL_RUNTIME_SPEC["action_dim"])
_OBSERVATION_DIM = int(_RL_RUNTIME_SPEC["observation_dim"])
_PHYSICS_DIM = int(_RL_RUNTIME_SPEC["physics_dim"])
_ROBOT_MODE = str(_RL_RUNTIME_SPEC.get("env_params", {}).get("robot_mode", "tool"))
if _ROBOT_MODE not in {"tool", "bare_franka"}:
    raise ValueError(f"Unsupported robot_mode: {_ROBOT_MODE!r}")
_USE_BARE_FRANKA = _ROBOT_MODE == "bare_franka"
_ASSET_ASSIGNMENT = _RL_RUNTIME_SPEC["asset_assignment_params"]
_ASSET_ASSIGNMENT_SEED = int(_ASSET_ASSIGNMENT["seed"])
_OBJECT_ASSIGNMENT_SEED = int(os.environ.get("TOOL_GENERALIST_OBJECT_ASSIGNMENT_SEED", _ASSET_ASSIGNMENT_SEED))
_RANDOMIZE_TOOL_ASSIGNMENT = bool(_ASSET_ASSIGNMENT["randomize_tool_assignment"])
_RANDOMIZE_OBJECT_ASSIGNMENT = bool(_ASSET_ASSIGNMENT["randomize_object_assignment"])
# Per-rank env count from the runtime spec; the helper derives global ids as
# global_rank * _NUM_ENVS_PER_RANK + local_env_id.
_NUM_ENVS_PER_RANK = int(_RL_RUNTIME_SPEC["num_envs"])
_GLOBAL_RANK = int(os.environ.get("TOOL_GENERALIST_GLOBAL_RANK", "0"))
_LOCAL_RANK = int(os.environ.get("TOOL_GENERALIST_LOCAL_RANK", "0"))
_WORLD_SIZE = int(os.environ.get("TOOL_GENERALIST_WORLD_SIZE", "1"))

if _NUM_ENVS_PER_RANK <= 0:
    raise ValueError("RL runtime spec num_envs must be > 0")
if _OBJECT_ASSIGNMENT_SEED < 0:
    raise ValueError("TOOL_GENERALIST_OBJECT_ASSIGNMENT_SEED must be >= 0")
if _GLOBAL_RANK < 0:
    raise ValueError("TOOL_GENERALIST_GLOBAL_RANK must be >= 0")
if _LOCAL_RANK < 0:
    raise ValueError("TOOL_GENERALIST_LOCAL_RANK must be >= 0")
if _WORLD_SIZE <= 0:
    raise ValueError("TOOL_GENERALIST_WORLD_SIZE must be > 0")


def _dr_event_enabled(term_cfg) -> bool:
    return bool(_RL_CONTRACT.domain_randomization.enabled and getattr(term_cfg, "enabled", False))


def _tool_dr_event_enabled(term_cfg) -> bool:
    return (not _USE_BARE_FRANKA) and _dr_event_enabled(term_cfg)


def _unsupported_event(enabled: bool, name: str):
    if enabled:
        raise NotImplementedError(f"{name} is not wired to an Isaac event yet")
    return None


def _table_top_z() -> float:
    return float(_RL_CONTRACT.table.pose_xyz[2]) + 0.5 * float(_RL_CONTRACT.table.size_xyz[2])


def _table_bounds_xy() -> tuple[tuple[float, float], tuple[float, float]]:
    center_x = float(_RL_CONTRACT.table.pose_xyz[0])
    center_y = float(_RL_CONTRACT.table.pose_xyz[1])
    half_x = 0.5 * float(_RL_CONTRACT.table.size_xyz[0])
    half_y = 0.5 * float(_RL_CONTRACT.table.size_xyz[1])
    return ((center_x - half_x, center_y - half_y), (center_x + half_x, center_y + half_y))


def load_object_candidates(
    source_path,
    usd_dir: str | None = None,
    obj_dir: str | None = None,
    uniform_scale=(1.0, 1.0, 1.0),
    *,
    use_scale_from_name: bool = False,
):
    """
    Load object candidates from a single JSON file.

    The JSON must be a list of strings in the fixed format "<name>-<scale>",
    for example: "core-bottle-xxxxxxxx-0.060".

    - usd_dir / obj_dir: directories used to build file paths as
      "<usd_dir>/<name>.usd" and "<obj_dir>/<name>.obj".
    - scale: parsed from the numeric suffix of each item and applied as
      uniform scaling (s, s, s).
    - The parameter `uniform_scale` is kept only for API compatibility and is not used.
    - If the JSON is not a list of strings or an entry does not match the
      expected format, a ValueError is raised.
    """
    assets: list[sim_utils.UsdFileCfg] = []
    assets_names = []

    # File mode: original behavior
    with open(source_path, "r") as f:
        data = json.load(f)
    # File mode: enforce fixed format list of strings like "<name>-<scale>"
    if not (isinstance(data, list) and all(isinstance(x, str) for x in data)):
        raise ValueError("Expected JSON to be a list of strings '<name>-<scale>'.")
    if usd_dir is None or obj_dir is None:
        raise ValueError("usd_dir and obj_dir must be provided.")

    for item in data:
        if '-' not in item:
            raise ValueError(f"Invalid item format (expected '<name>-<scale>'): {item}")
        base, scale_str = item.rsplit('-', 1)

        if base in assets_names:
            print(f"[WARNING] Asset {base} already exists, skipping...")
            continue
        assets_names.append(base)

        usd_path = os.path.join(usd_dir, f"{base}", f"{base}.usd")
        obj_path = os.path.join(obj_dir, f"{base}.obj")

        # Check if USD file exists, skip if not found
        if not os.path.exists(usd_path):
            print(f"[WARNING] USD file not found: {usd_path}, skipping...")
            continue

        usd_cfg = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(0.01, 0.01, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3)),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        )
        usd_cfg.obj_path = obj_path
        assets.append(usd_cfg)
    return assets


# Helper for point cloud caching, compatible with IsaacLab multi-env
def get_cached_cloud(obj_path):
    key = obj_path
    if key not in _CLOUD_CACHE:
        print(f"[cloud_cache] create key={key}", flush=True)
        _CLOUD_CACHE[key] = Cloud(obj_path)  # No scale parameter needed
    return _CLOUD_CACHE[key]


# ---------------------------------------------------------------------------
# Multi-tool data: each robot USD has a different welded tool. We keep the
# unique base lists, then expand them per local env using the runtime assignment
# contract and global rank metadata.
#
# Tool selection is driven by tools_selected.json.  Mesh/head-area resolution
# goes through utils.assets and the adjusted decomposed mesh contract.
# ---------------------------------------------------------------------------
TOOL_SCALE: float = float(_PATHS.get("tools", _PATHS.get("tool_mesh", {})).get("scale", 0.1))

TOOL_USD_PATHS: list[str] = []
TOOL_DATA: list[dict] = []
TOOL_ASSET_INDICES_BY_ENV: list[int] = []
TOOL_USD_PATHS_BY_ENV: list[str] = []
TOOL_SPAWN_ASSET_INDICES: list[int] = []
TOOL_USD_PATHS_FOR_SPAWN: list[str] = []

if _USE_BARE_FRANKA:
    print("[INFO] robot_mode=bare_franka: skipping tool USD/mesh loading")
else:
    _TOOLS_CFG = _PATHS["tools"]
    _TOOL_MESH_ROOT = _TOOLS_CFG.get("meshdata_adjusted_root")
    if not _TOOL_MESH_ROOT:
        raise ValueError("paths.yaml must define tools.meshdata_adjusted_root for adjusted tool meshes")
    _TOOLS_ADJUSTED_JSON = _TOOLS_CFG.get("tools_adjusted_json", _TOOLS_CFG.get("tools_json"))

    # Load the tool selection manifest: a list of tool names to include
    _selected_tool_names: list[str] = load_selected_tool_ids(_TOOLS_CFG["tools_selected_json"])

    # Build per-tool metadata + USD paths, filtered by the manifest
    for _tool_name in _selected_tool_names:
        _usd_path = os.path.join(_TOOLS_CFG["robots_usd_dir"], f"panda_instanceable_{_tool_name}.usd")
        _obj_path = str(resolve_tool_mesh_path(_TOOL_MESH_ROOT, _tool_name))

        if not os.path.isfile(_usd_path):
            print(f"[WARNING] Robot USD not found for tool '{_tool_name}': {_usd_path}, skipping")
            continue

        try:
            _adjusted_entry = load_tool_adjusted_entry(_TOOLS_ADJUSTED_JSON, _tool_name)
            _head_area = load_tool_head_area(_TOOLS_ADJUSTED_JSON, _obj_path, _tool_name)
        except ToolAssetContractError as exc:
            print(f"[WARNING] Tool '{_tool_name}' violates adjusted asset contract, skipping: {exc}")
            continue
        _base_center = _adjusted_entry.get("base_center")
        if _head_area is None:
            print(f"[WARNING] head_area not found for tool '{_tool_name}' in {_TOOLS_ADJUSTED_JSON}")
        if not os.path.isfile(_obj_path):
            print(f"[WARNING] adjusted decomposed mesh not found for tool '{_tool_name}': {_obj_path}")

        TOOL_USD_PATHS.append(_usd_path)
        TOOL_DATA.append({
            "name": _tool_name,
            "obj_path": _obj_path,
            "mesh_source": "adjusted_decomposed_mesh",
            "head_area": _head_area,
            "base_center": _base_center,
        })

    print(f"[INFO] Loaded {len(TOOL_DATA)} tool variants from {_TOOLS_CFG['robots_usd_dir']}")
    if not TOOL_DATA:
        raise RuntimeError(
            "No valid tool variants remain after filtering tools_selected.json. "
            "Check tools.tools_adjusted_json head_area entries and robot USD paths."
        )
    TOOL_ASSET_INDICES_BY_ENV = asset_indices_for_rank(
        _NUM_ENVS_PER_RANK,
        _GLOBAL_RANK,
        len(TOOL_DATA),
        randomize=_RANDOMIZE_TOOL_ASSIGNMENT,
        seed=_ASSET_ASSIGNMENT_SEED,
        salt=TOOL_ASSIGNMENT_SALT,
    )
    TOOL_USD_PATHS_BY_ENV = [TOOL_USD_PATHS[index] for index in TOOL_ASSET_INDICES_BY_ENV]
    TOOL_SPAWN_ASSET_INDICES = (
        TOOL_ASSET_INDICES_BY_ENV
        if _RANDOMIZE_TOOL_ASSIGNMENT
        else sequential_spawn_indices_for_rank(_NUM_ENVS_PER_RANK, _GLOBAL_RANK, len(TOOL_DATA))
    )
    TOOL_USD_PATHS_FOR_SPAWN = [TOOL_USD_PATHS[index] for index in TOOL_SPAWN_ASSET_INDICES]
    print(
        f"[INFO] Tool assignment rank={_GLOBAL_RANK}/{_WORLD_SIZE} "
        f"local_rank={_LOCAL_RANK} envs={_NUM_ENVS_PER_RANK} "
        f"randomize={_RANDOMIZE_TOOL_ASSIGNMENT}"
    )
    print(
        f"[INFO] Tool spawn prototypes envs={_NUM_ENVS_PER_RANK} "
        f"spawn_assets={len(TOOL_USD_PATHS_FOR_SPAWN)} "
        f"total_assets={len(TOOL_USD_PATHS)}"
    )

# Legacy single-tool aliases (index 0) for backward-compatible imports
TOOL_OBJ_PATH: str = TOOL_DATA[0]["obj_path"] if TOOL_DATA else ""
TOOL_HEAD_AREA_NORM = TOOL_DATA[0].get("head_area") if TOOL_DATA else None


def _assigned_index_for_env(env_id: int, assignment: list[int], label: str) -> int:
    if len(assignment) == 0:
        raise ValueError(f"No {label} assignment entries are available")
    env_id = int(env_id)
    if env_id < 0:
        raise ValueError("env_id must be >= 0")
    if env_id >= len(assignment):
        raise ValueError(
            f"env_id {env_id} is outside the precomputed {label} assignment "
            f"for {len(assignment)} local envs"
        )
    return assignment[env_id]


def get_tool_index_for_env(env_id: int) -> int:
    """Return the unique tool index assigned to a local env."""
    return _assigned_index_for_env(env_id, TOOL_ASSET_INDICES_BY_ENV, "tool")


def get_tool_data_for_env(env_id: int) -> dict:
    """Return per-tool metadata dict for the given env_id."""
    return TOOL_DATA[get_tool_index_for_env(env_id)]


OBJECT_ASSET_CFGS: list[sim_utils.UsdFileCfg] = load_object_candidates(
    _PATHS["dgn"]["candidates_json"],
    usd_dir=_PATHS["dgn"]["usd_dir"],
    obj_dir=_PATHS["dgn"]["obj_dir"],
)
OBJECT_ASSET_INDICES_BY_ENV: list[int] = asset_indices_for_rank(
    _NUM_ENVS_PER_RANK,
    _GLOBAL_RANK,
    len(OBJECT_ASSET_CFGS),
    randomize=_RANDOMIZE_OBJECT_ASSIGNMENT,
    seed=_OBJECT_ASSIGNMENT_SEED,
    salt=OBJECT_ASSIGNMENT_SALT,
)
OBJECT_ASSET_CFGS_BY_ENV: list[sim_utils.UsdFileCfg] = [
    OBJECT_ASSET_CFGS[index] for index in OBJECT_ASSET_INDICES_BY_ENV
]
OBJECT_SPAWN_ASSET_INDICES: list[int] = (
    OBJECT_ASSET_INDICES_BY_ENV
    if _RANDOMIZE_OBJECT_ASSIGNMENT
    else sequential_spawn_indices_for_rank(_NUM_ENVS_PER_RANK, _GLOBAL_RANK, len(OBJECT_ASSET_CFGS))
)
OBJECT_ASSET_CFGS_FOR_SPAWN: list[sim_utils.UsdFileCfg] = [
    OBJECT_ASSET_CFGS[index] for index in OBJECT_SPAWN_ASSET_INDICES
]
print(
    f"[INFO] Object assignment rank={_GLOBAL_RANK}/{_WORLD_SIZE} "
    f"local_rank={_LOCAL_RANK} envs={_NUM_ENVS_PER_RANK} "
    f"randomize={_RANDOMIZE_OBJECT_ASSIGNMENT} seed={_OBJECT_ASSIGNMENT_SEED}"
)
print(
    f"[INFO] Object spawn prototypes envs={_NUM_ENVS_PER_RANK} "
    f"spawn_assets={len(OBJECT_ASSET_CFGS_FOR_SPAWN)} "
    f"total_assets={len(OBJECT_ASSET_CFGS)}"
)


def get_object_index_for_env(env_id: int) -> int:
    """Return the unique object index assigned to a local env."""
    return _assigned_index_for_env(env_id, OBJECT_ASSET_INDICES_BY_ENV, "object")


def get_object_asset_cfg_for_env(env_id: int) -> sim_utils.UsdFileCfg:
    """Return the base object asset config assigned to a local env."""
    return OBJECT_ASSET_CFGS[get_object_index_for_env(env_id)]


default_joint_pos = FRANKA_PANDA_HIGH_PD_CFG.init_state.joint_pos.copy()
# User-defined joint workspace for Franka arm (7 DOF)
JOINT_BOX_MIN_BASE = [-0.3, -0.4636, -0.2, -2.7432, -0.3335, 1.5269, -1.5707963267948966]
JOINT_BOX_MAX_BASE = [0.3, 0.5432, 0.2, -1.5237, 0.3335, 2.5744, 1.5707963267948966]
JOINT_BOX_SHIFT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
JOINT_BOX_MIN = [mn + d for mn, d in zip(JOINT_BOX_MIN_BASE, JOINT_BOX_SHIFT)]
JOINT_BOX_MAX = [mx + d for mx, d in zip(JOINT_BOX_MAX_BASE, JOINT_BOX_SHIFT)]
# Choose initial pose as midpoint within the box range
_joint_init_mid = [(mn + mx) / 2.0 for mn, mx in zip(JOINT_BOX_MIN, JOINT_BOX_MAX)]
custom_joint_init = {
    "panda_joint1": _joint_init_mid[0],
    "panda_joint2": _joint_init_mid[1],
    "panda_joint3": _joint_init_mid[2],
    "panda_joint4": _joint_init_mid[3],
    "panda_joint5": _joint_init_mid[4],
    "panda_joint6": _joint_init_mid[5],
    "panda_joint7": _joint_init_mid[6],
}
bare_franka_path = os.path.abspath(_PATHS["robot"]["franka_usd"])
if _USE_BARE_FRANKA and not os.path.isfile(bare_franka_path):
    print(
        f"[WARNING] Bare Franka USD not found at {bare_franka_path}; "
        "using Isaac Lab's default FRANKA_PANDA_HIGH_PD_CFG USD"
    )
arm_only_actuators = {
    actuator_name: actuator_config 
    for actuator_name, actuator_config in FRANKA_PANDA_HIGH_PD_CFG.actuators.items() 
    if "hand" not in actuator_name and "finger" not in actuator_name
}

EE_TARGET_PRIM_PATH = (
    "{ENV_REGEX_NS}/Robot/panda_hand"
    if _USE_BARE_FRANKA
    else "{ENV_REGEX_NS}/Robot/tool_mount/link_coacd_convex_piece_0"
)
EE_TARGET_NAME = "ee_hand" if _USE_BARE_FRANKA else "ee_tool"


def make_robot_cfg() -> ArticulationCfg:
    if _USE_BARE_FRANKA:
        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
        if os.path.isfile(bare_franka_path):
            robot_cfg.spawn.usd_path = bare_franka_path
        robot_cfg.spawn.rigid_props.disable_gravity = True
    else:
        robot_cfg = build_multi_tool_robot_cfg(TOOL_USD_PATHS_FOR_SPAWN, random_choice=False)

    return robot_cfg.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=custom_joint_init),
    )


@configclass
class NonPrehensileSceneCfg(InteractiveSceneCfg):
    """Configuration for a non-prehensile scene."""
    
    # Disable physics replication to avoid conflicts with MultiAssetSpawnerCfg
    replicate_physics: bool = False
    if not _RL_CONTRACT.table.enabled:
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",  # optional: "plane", "usd", "generator"
            collision_group=-1,
            # Explicitly None to skip bind_physics_material in spawn_ground_plane().
            # Terrain friction is set at reset by the randomize_terrain_material event.
            physics_material=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            debug_vis=False,
        )
    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    if _RL_CONTRACT.table.enabled:
        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=tuple(_RL_CONTRACT.table.pose_xyz),
            ),
            spawn=sim_utils.CuboidCfg(
                size=tuple(_RL_CONTRACT.table.size_xyz),
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=_RL_CONTRACT.table.material.static_friction,
                    dynamic_friction=_RL_CONTRACT.table.material.dynamic_friction,
                    restitution=_RL_CONTRACT.table.material.restitution,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(_RL_CONTRACT.table.color_rgba[:3]),
                ),
            ),
        )
    # # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.6, 0, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(
    #         # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #         scale=(0.8, 0.6, 1.0),
    #     ),
    # )
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     # init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
    #         # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #         scale=(2.0, 3.0, 1.0),
    #     ),
    # )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=OBJECT_ASSET_CFGS_FOR_SPAWN,
            random_choice=False,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    robot = make_robot_cfg()

    # FrameTransformer anchored to the active end-effector body.
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=EE_TARGET_PRIM_PATH,
                name=EE_TARGET_NAME,
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),  # Offset is applied in get_head_area_pos_w via _head_area_offsets
                ),
            ),
        ],
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    target_object_pose = mdp.StablePoseCommandCfg(
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,  # Visualize target pose
        xy_offset_range=_RL_CONTRACT.object_pose_sampling.xy_offset_range,
        initial_position_range=_RL_CONTRACT.object_pose_sampling.initial_position_range,
    )

@configclass
class RelativeJointPositionActionsCfg:
    """Relative (delta) joint position action specifications for the MDP."""
    # Relative joint position control: q_target = q_current + scaled_action
    arm_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=_RL_CONTRACT.action.scale,
        use_zero_offset=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # Object Cloud (512*3=1536D: point cloud xyz in env frame)
        object_cloud = ObsTerm(
            func=mdp.get_object_pointcloud_in_env_frame,
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        ) if _RL_CONTRACT.observation.include_object_cloud else None

        # Tool Cloud (512*3=1536D: tool point cloud xyz in env frame)
        tool_cloud = ObsTerm(
            func=mdp.get_tool_pointcloud_in_env_frame,
            noise=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        ) if (_RL_CONTRACT.observation.include_tool_cloud and not _USE_BARE_FRANKA) else None

        # Object bbox center (3D): MUST come AFTER object_cloud so the cache is populated.
        object_bbox_center = ObsTerm(
            func=mdp.get_obj_bbox_center
        ) if _RL_CONTRACT.observation.include_bbox_centers else None

        # Tool bbox center (3D): MUST come AFTER tool_cloud so the cache is populated.
        tool_bbox_center = ObsTerm(
            func=mdp.get_tool_bbox_center
        ) if (
            _RL_CONTRACT.observation.include_bbox_centers
            and _RL_CONTRACT.observation.include_tool_cloud
            and not _USE_BARE_FRANKA
        ) else None

        # Hand State (9D: hand position[3] + rotation_matrix[6])
        hand_state = ObsTerm(
            func=mdp.hand_state, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )
        
        # Robot State (14D: joint_positions[7] + joint_velocities[7])
        robot_state = ObsTerm(
            func=mdp.robot_state,
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )
        
        # Previous Action (Variable D: depends on action type) - using IsaacLab's built-in function
        previous_action = ObsTerm(func=mdp.last_action)
        
        # Relative Pose Goal (9D: goal relative to current object pose)
        relative_goal_pose = ObsTerm(
            func=mdp.rel_pose_goal, params={"command_name": "target_object_pose"},
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )

        # abs_goal = ObsTerm(func=mdp.abs_pose_goal, params={"command_name": "target_object_pose"})
        # cur_pose = ObsTerm(func=mdp.object_pose_9d_in_env_frame)
        
        # Physical parameters: field order comes from rl_runtime_spec.json.
        phys_params = ObsTerm(
            func=mdp.phys_params,
            params={"field_names": _PHYSICS_OBSERVATION_FIELDS},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_initial_object_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    randomize_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "scale_range": _RL_CONTRACT.domain_randomization.object.scale.range,
            "asset_cfg": SceneEntityCfg("object"),
        },
    ) if _dr_event_enabled(_RL_CONTRACT.domain_randomization.object.scale) else None

    # Tool mass randomization: randomize the mass of the tool body (link_coacd_convex_piece_0)
    randomize_tool_mass = EventTerm(
        func=mdp.randomize_tool_mass,
        mode="reset",
        params={
            "mass_range": _RL_CONTRACT.domain_randomization.tool.mass.range,
        },
    ) if _tool_dr_event_enabled(_RL_CONTRACT.domain_randomization.tool.mass) else None

    # Tool friction randomization: randomize friction of the tool body's collision shapes
    randomize_tool_friction = EventTerm(
        func=mdp.randomize_tool_friction,
        mode="reset",
        params={
            "static_friction_range": _RL_CONTRACT.domain_randomization.tool.material.static_friction_range,
            "dynamic_friction_range": _RL_CONTRACT.domain_randomization.tool.material.dynamic_friction_range,
            "restitution_range": _RL_CONTRACT.domain_randomization.tool.material.restitution_range,
        },
    ) if _tool_dr_event_enabled(_RL_CONTRACT.domain_randomization.tool.material) else None

    # Physical parameter randomization events
    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": _RL_CONTRACT.domain_randomization.object.mass.range,
            "operation": "abs",  # Absolute value operation
            "distribution": "uniform",
            "recompute_inertia": _RL_CONTRACT.domain_randomization.object.mass.recompute_inertia,
        },
    ) if _dr_event_enabled(_RL_CONTRACT.domain_randomization.object.mass) else None

    # object material randomization
    randomize_object_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": _RL_CONTRACT.domain_randomization.object.material.static_friction_range,
            "dynamic_friction_range": _RL_CONTRACT.domain_randomization.object.material.dynamic_friction_range,
            "restitution_range": _RL_CONTRACT.domain_randomization.object.material.restitution_range,
            "num_buckets": _RL_CONTRACT.domain_randomization.object.material.num_buckets,
            "make_consistent": _RL_CONTRACT.domain_randomization.object.material.make_consistent,
        },
    ) if _dr_event_enabled(_RL_CONTRACT.domain_randomization.object.material) else None

    # Terrain friction randomization - using custom function to randomize terrain material
    randomize_terrain_material = EventTerm(
        func=mdp.randomize_terrain_material,
        mode="reset",
        params={
            "static_friction_range": _RL_CONTRACT.domain_randomization.ground.material.static_friction_range,
            "dynamic_friction_range": _RL_CONTRACT.domain_randomization.ground.material.dynamic_friction_range,
            "restitution_range": _RL_CONTRACT.domain_randomization.ground.material.restitution_range,
            "num_buckets": _RL_CONTRACT.domain_randomization.ground.material.num_buckets,
        },
    ) if (
        not _RL_CONTRACT.table.enabled
        and _dr_event_enabled(_RL_CONTRACT.domain_randomization.ground.material)
    ) else None

    randomize_table_material = _unsupported_event(
        _RL_CONTRACT.table.enabled
        and _dr_event_enabled(_RL_CONTRACT.domain_randomization.table_surface.material),
        "table_surface material randomization",
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    task_success = RewTerm(
        func=mdp.task_success_reward,
        params={
            "command_name": "target_object_pose", 
            "threshold": _RL_CONTRACT.reward.success_threshold,
            "rotation_threshold": _RL_CONTRACT.reward.rotation_threshold,
            "planar": False,
            "base_reward": 1.0,
        },
        weight=_RL_CONTRACT.reward.task_success_term_weight
    )

    contact_reward = RewTerm(
        func=mdp.object_ee_distance_tanh,
        params={
            "std": _RL_CONTRACT.reward.contact_std,
        },
        weight=_RL_CONTRACT.reward.contact_term_weight,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance_tanh,
        params={
            "std": _RL_CONTRACT.reward.object_goal_std,
            "command_name": "target_object_pose",
            "obj_ee_distance_threshold": _RL_CONTRACT.reward.contact_std,
            "rotation_distance_divisor": getattr(_RL_CONTRACT.reward, "rotation_distance_divisor", 5.0),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=_RL_CONTRACT.reward.object_goal_tracking_term_weight,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance_tanh,
        params={
            "std": _RL_CONTRACT.reward.object_goal_fine_std,
            "command_name": "target_object_pose",
            "obj_ee_distance_threshold": _RL_CONTRACT.reward.contact_std,
            "rotation_distance_divisor": getattr(_RL_CONTRACT.reward, "rotation_distance_divisor", 5.0),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=_RL_CONTRACT.reward.object_goal_tracking_fine_term_weight,
    )
    
    # Energy penalty: c_energy = k_e * Σ(τ_i * q̇_i)
    energy_penalty = RewTerm(
        func=mdp.joint_power_penalty,
        params={"k_e": 0.0001},
        weight=_RL_CONTRACT.reward.energy_penalty_weight,
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reached = DoneTerm(
        func=mdp.object_reached_goal,
        params={
            "command_name": "target_object_pose",
            "threshold": _RL_CONTRACT.reward.success_threshold,
            "rotation_threshold": _RL_CONTRACT.reward.rotation_threshold,
            "planar": False,
        },
    )
    object_dropped = DoneTerm(
        func=mdp.object_dropped_off_table,
        params={"minimum_height": _table_top_z() - 0.15}
    )

@configclass
class NonPrehensileEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: NonPrehensileSceneCfg = NonPrehensileSceneCfg(
        num_envs=_RL_CONTRACT.env.num_envs,
        env_spacing=_RL_CONTRACT.env.env_spacing,
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: RelativeJointPositionActionsCfg = RelativeJointPositionActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # Observation normalization
    normalize_observations: bool = True  # Whether to normalize observations to [-1,1] range, except hand_state and pointcloud 
    object_cloud_centering: str = _RL_CONTRACT.observation.object_cloud_centering
    tool_cloud_centering: str = _RL_CONTRACT.observation.tool_cloud_centering
    mesh_centering: str = _RL_CONTRACT.observation.mesh_centering
    action_dim: int = _ACTION_DIM
    observation_dim: int = _OBSERVATION_DIM
    physics_dim: int = _PHYSICS_DIM
    robot_mode: str = _ROBOT_MODE
    physics_observation_fields: tuple[str, ...] = _PHYSICS_OBSERVATION_FIELDS
    table_enabled: bool = _RL_CONTRACT.table.enabled
    table_size_xyz: tuple[float, float, float] = tuple(_RL_CONTRACT.table.size_xyz)
    table_pose_xyz: tuple[float, float, float] = tuple(_RL_CONTRACT.table.pose_xyz)
    table_bounds_xy: tuple[tuple[float, float], tuple[float, float]] = _table_bounds_xy()
    table_placement_margin_xy: float = _RL_CONTRACT.table.placement_margin_xy
    table_placement_max_attempts: int = _RL_CONTRACT.table.placement_max_attempts
    table_material = _RL_CONTRACT.table.material
    # Visualization settings
    visualize_current_object_pose: bool = True  # Enable current object pose visualization
    visualize_object_pointcloud: bool = False  # Enable object point cloud visualization for debug in first env
    visualize_tool_pointcloud: bool = False if _USE_BARE_FRANKA else True
    visualize_eef_position: bool = False  # Enable eef tool position visualization
    visualize_object_velocity_mass: bool = False  # Enable 7D object velocity & mass visualization
    visualize_tool_velocity_mass: bool = False  # Enable 7D tool velocity & mass visualization

    # Performance settings
    use_torch_compile: bool = True  # Enable torch.compile on hot paths

    # Whether to enforce (apply) the robot soft joint limits configured in this env
    # Set to False to skip updating soft joint limits at environment creation time.
    enforce_joint_limits: bool = False

    # Disable observation noise across all policy observation terms during env creation
    disable_obs_noise: bool = False

    def __post_init__(self) -> None:
        # Optionally disable observation noise for evaluation or ablations
        if self.disable_obs_noise:
            obs_cfg = self.observations
            policy_cfg = getattr(obs_cfg, "policy", None)
            if policy_cfg is not None:
                for attr_name in dir(policy_cfg):
                    if attr_name.startswith("_"):
                        continue
                    term = getattr(policy_cfg, attr_name, None)
                    if term is None:
                        continue
                    noise = getattr(term, "noise", None)
                    if noise is None:
                        continue
                    if hasattr(noise, "mean"):
                        noise.mean = 0.0
                    if hasattr(noise, "std"):
                        noise.std = 0.0
        
        # General settings - match reference config
        self.decimation = _RL_CONTRACT.env.decimation
        self.episode_length_s = _RL_CONTRACT.env.episode_length_s
        
        # Viewer settings
        self.viewer.eye = (2.5, 0.5, 0.8)
        # self.viewer.eye = (6, 0, 6)
        
        # Simulation settings - match reference config dt
        self.sim.dt = _RL_CONTRACT.env.sim_dt
        self.sim.render_interval = self.decimation
        
        # Physics settings - match reference config
        self.sim.physx.solver_position_iteration_count = _RL_CONTRACT.env.solver_position_iteration_count
        self.sim.physx.solver_velocity_iteration_count = _RL_CONTRACT.env.solver_velocity_iteration_count


class NonPrehensileEnv(ManagerBasedRLEnv):
    """Custom environment wrapper for non-prehensile manipulation.
    
    This class extends ManagerBasedRLEnv and relies on IsaacLab's built-in
    action tracking via action_manager.action for previous action observations.
    """
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        # Initialize the base environment
        super().__init__(cfg, render_mode, **kwargs)
        # Override Franka arm soft joint limits with user-defined box and clamp current state
        robot = self.scene["robot"]
        entity = SceneEntityCfg("robot", joint_names=["panda_joint.*"])  # 7 arm joints
        entity.resolve(self.scene)
        joint_ids = entity.joint_ids
        mins = torch.tensor(JOINT_BOX_MIN, device=self.device, dtype=torch.float32).view(1, -1).repeat(self.num_envs, 1)
        maxs = torch.tensor(JOINT_BOX_MAX, device=self.device, dtype=torch.float32).view(1, -1).repeat(self.num_envs, 1)
        # update soft limits in-place
        if self.cfg.enforce_joint_limits:
            limits = robot.data.soft_joint_pos_limits
            limits[:, joint_ids, 0] = mins
            limits[:, joint_ids, 1] = maxs
            robot.data.soft_joint_pos_limits[:] = limits

        # _head_area_offsets and _object_scales are computed in post_reset() below.
        self._head_area_offsets = torch.zeros(self.num_envs, 3, device=self.device)

        # Initialize success tracking buffers
        self.episode_success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.total_episodes = 0
        self.total_successes = 0

        # Sliding window for recent success rate (last 100 episodes)
        self.recent_success_window = deque(maxlen=100)
        self.recent_success_rate = 0.0

        # Global step counter for periodic debug prints
        self._global_step = 0

        # Run post-init setup: physics settings, scale caching, head area offsets
        self.post_reset()
    
    def step(self, action):
        """Override step to track success rates."""
        # Call parent step method
        obs, reward, terminated, truncated, info = super().step(action)

        # Increment global step and periodically print timers
        # self._global_step += 1
        # if self._global_step % 1 == 0:
        #     from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import print_obs_timers
        #     from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.commands import print_cmd_timers
        #     print_obs_timers(self)
        #     print_cmd_timers(self)

        success_mask = self.termination_manager.get_term("reached")
        # Update episode success buffer
        self.episode_success_buf = self.episode_success_buf | success_mask
        
        # Check for episode endings (terminated or truncated)
        episode_ended = terminated | truncated
        
        # Update success statistics when episodes end
        if torch.any(episode_ended):
            ended_env_ids = torch.where(episode_ended)[0]
            for env_id in ended_env_ids:
                self.total_episodes += 1
                episode_success = self.episode_success_buf[env_id].item()
                if episode_success:
                    self.total_successes += 1
                
                # Add to sliding window for recent success rate
                self.recent_success_window.append(episode_success)
            
            # Store success status before reset for external access
            self._episode_success_before_reset = self.episode_success_buf.clone()
            
            # Reset success buffer for ended episodes
            self.episode_success_buf[episode_ended] = False
            
            # Calculate and log success rate
            if self.total_episodes > 0:
                success_rate = self.total_successes / self.total_episodes
                # Add to extras["log"] for tensorboard logging
                if "log" not in self.extras:
                    self.extras["log"] = dict()
                self.extras["log"]["success_rate"] = success_rate
                self.extras["log"]["total_episodes"] = self.total_episodes
                self.extras["log"]["total_successes"] = self.total_successes
                
                # Calculate recent success rate (sliding window of last 100 episodes)
                if len(self.recent_success_window) > 0:
                    self.recent_success_rate = sum(self.recent_success_window) / len(self.recent_success_window)
                    self.extras["log"]["recent_success_rate"] = self.recent_success_rate

        return obs, reward, terminated, truncated, info

    def post_reset(self):
        # NOTE: _object_scales and _tool_scales removed — scales are now baked into
        # per-env sampled point clouds at init time (in get_object_pointcloud).

        # Compute per-env head area offsets from the fixed fork OBJ + head_area_norm.
        # Each offset is in the tool's local frame (relative to link_coacd_convex_piece_0 origin).
        if _USE_BARE_FRANKA:
            self._head_area_offsets = torch.zeros(self.num_envs, 3, device=self.device)
        else:
            self._head_area_offsets = mdp.compute_head_area_offsets_from_usd(self)
