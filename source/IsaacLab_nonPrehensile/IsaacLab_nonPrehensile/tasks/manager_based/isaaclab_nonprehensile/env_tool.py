# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
import json
import yaml
import torch
import xml.etree.ElementTree as ET
from collections import deque
import time
from pathlib import Path

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
from collections.abc import Mapping, Sequence

from utils.assets import (
    GeneratedGripperAsset,
    OneDofGripperAsset,
    ToolAssetContractError,
    load_generated_gripper_manifest,
    load_one_dof_gripper_manifest,
    load_selected_tool_ids,
    load_tool_adjusted_entry,
    load_tool_head_area,
    resolve_tool_mesh_path,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.cloud import Cloud

_CLOUD_CACHE = {}

# Load path configuration. By default this reads configs/paths/default.yaml;
# set TOOL_GENERALIST_PATHS_YAML to run against an alternate asset config.
_DEFAULT_PATHS_CFG_FILE = os.path.join(
    os.path.dirname(__file__), "../" * 6, "configs/paths/default.yaml"
)
_PATHS_CFG_FILE = os.environ.get("TOOL_GENERALIST_PATHS_YAML", _DEFAULT_PATHS_CFG_FILE)
_PATHS_CFG_FILE = os.path.abspath(os.path.normpath(_PATHS_CFG_FILE))
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
    GENERATED_GRIPPER_ASSIGNMENT_SALT,
    OBJECT_ASSIGNMENT_SALT,
    ONE_DOF_GRIPPER_ASSIGNMENT_SALT,
    TOOL_ASSIGNMENT_SALT,
    asset_indices_for_rank,
    cross_embodiment_mode_for_rank,
    sequential_spawn_indices_for_rank,
)

_RL_RUNTIME_SPEC = load_runtime_spec_from_env()
_RL_CONTRACT = runtime_spec_contract(_RL_RUNTIME_SPEC)
_PHYSICS_OBSERVATION_FIELDS = tuple(_RL_RUNTIME_SPEC["physics_observation_fields"])
_ACTION_DIM = int(_RL_RUNTIME_SPEC["action_dim"])
_OBSERVATION_DIM = int(_RL_RUNTIME_SPEC["observation_dim"])
_PHYSICS_DIM = int(_RL_RUNTIME_SPEC["physics_dim"])
_NUM_ENVS_PER_RANK = int(_RL_RUNTIME_SPEC["num_envs"])
_GLOBAL_RANK = int(os.environ.get("TOOL_GENERALIST_GLOBAL_RANK", "0"))
_LOCAL_RANK = int(os.environ.get("TOOL_GENERALIST_LOCAL_RANK", "0"))
_WORLD_SIZE = int(os.environ.get("TOOL_GENERALIST_WORLD_SIZE", "1"))
_REQUESTED_ROBOT_MODE = str(_RL_RUNTIME_SPEC.get("env_params", {}).get("robot_mode", "tool"))
_SUPPORTED_ROBOT_MODES = {
    "tool",
    "bare_franka",
    "official_panda_gripper",
    "generated_gripper",
    "one_dof_gripper",
    "cross_embodiment_gripper",
}
if _REQUESTED_ROBOT_MODE not in _SUPPORTED_ROBOT_MODES:
    raise ValueError(f"Unsupported robot_mode: {_REQUESTED_ROBOT_MODE!r}")
_USE_CROSS_EMBODIMENT_GRIPPER = _REQUESTED_ROBOT_MODE == "cross_embodiment_gripper"
if _USE_CROSS_EMBODIMENT_GRIPPER:
    _ROBOT_MODE = cross_embodiment_mode_for_rank(_GLOBAL_RANK, _WORLD_SIZE)
    print(
        "[INFO] cross_embodiment_gripper "
        f"rank={_GLOBAL_RANK}/{_WORLD_SIZE} effective_robot_mode={_ROBOT_MODE}"
    )
else:
    _ROBOT_MODE = _REQUESTED_ROBOT_MODE
_USE_BARE_FRANKA = _ROBOT_MODE == "bare_franka"
_USE_OFFICIAL_PANDA_GRIPPER = _ROBOT_MODE == "official_panda_gripper"
_USE_GENERATED_GRIPPER = _ROBOT_MODE == "generated_gripper"
_USE_ONE_DOF_GRIPPER = _ROBOT_MODE == "one_dof_gripper"
_USE_WELDED_TOOL = _ROBOT_MODE == "tool"
# Franka specifies 50 mm/s travel speed per finger. Experiments that reproduce
# the historical asset-limit behavior override this explicitly in RLEnvCfg.
_GENERATED_PARALLEL_FINGER_VELOCITY_LIMIT_M_S = float(
    _RL_RUNTIME_SPEC["env_params"][
        "generated_parallel_finger_velocity_limit_m_s"
    ]
)
_ASSET_ASSIGNMENT = _RL_RUNTIME_SPEC["asset_assignment_params"]
_ASSET_ASSIGNMENT_SEED = int(_ASSET_ASSIGNMENT["seed"])
_OBJECT_ASSIGNMENT_SEED = int(os.environ.get("TOOL_GENERALIST_OBJECT_ASSIGNMENT_SEED", _ASSET_ASSIGNMENT_SEED))
_RANDOMIZE_TOOL_ASSIGNMENT = bool(_ASSET_ASSIGNMENT["randomize_tool_assignment"])
_RANDOMIZE_OBJECT_ASSIGNMENT = bool(_ASSET_ASSIGNMENT["randomize_object_assignment"])
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
    return _USE_WELDED_TOOL and _dr_event_enabled(term_cfg)


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

    Supported schemas are either a list of strings in the fixed format
    "<name>-<scale>" or a list of dictionaries with ``object`` and ``scale``
    fields. Dictionary entries intentionally ignore all pose-related fields.

    - usd_dir / obj_dir: directories used to build file paths as
      "<usd_dir>/<name>.usd" and "<obj_dir>/<name>.obj".
    - Dictionary-entry scales are always applied as uniform scaling (s, s, s).
    - String-entry scales are applied only when ``use_scale_from_name`` is true;
      otherwise their historical spawn scale of 0.01 is preserved.
    - The parameter `uniform_scale` is kept only for API compatibility and is not used.
    - Mixed schemas, invalid names, non-positive scales, and duplicate objects
      with conflicting scales are rejected.
    """
    assets: list[sim_utils.UsdFileCfg] = []
    asset_scales_by_name: dict[str, float] = {}

    with open(source_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty JSON list of object candidates.")
    if usd_dir is None or obj_dir is None:
        raise ValueError("usd_dir and obj_dir must be provided.")

    string_schema = all(isinstance(item, str) for item in data)
    dict_schema = all(isinstance(item, Mapping) for item in data)
    if not string_schema and not dict_schema:
        raise ValueError(
            "Object candidates must be uniformly '<name>-<scale>' strings or "
            "dictionaries containing 'object' and 'scale'."
        )

    for index, item in enumerate(data):
        if string_schema:
            if "-" not in item:
                raise ValueError(f"Invalid object candidate at index {index}: {item!r}")
            base, scale_text = item.rsplit("-", 1)
            try:
                manifest_scale = float(scale_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid object scale at index {index}: {scale_text!r}"
                ) from exc
            spawn_scale = manifest_scale if use_scale_from_name else 0.01
        else:
            base = item.get("object")
            manifest_scale = item.get("scale")
            if not isinstance(base, str) or not base:
                raise ValueError(
                    f"Object candidate at index {index} requires a non-empty string 'object'."
                )
            if isinstance(manifest_scale, bool) or not isinstance(manifest_scale, (int, float)):
                raise ValueError(
                    f"Object candidate at index {index} requires a numeric 'scale'."
                )
            manifest_scale = float(manifest_scale)
            spawn_scale = manifest_scale

        if not math.isfinite(manifest_scale) or manifest_scale <= 0.0:
            raise ValueError(
                f"Object candidate at index {index} has invalid scale {manifest_scale!r}."
            )

        previous_scale = asset_scales_by_name.get(base)
        if previous_scale is not None:
            if not math.isclose(previous_scale, manifest_scale, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"Object {base!r} has conflicting scales: "
                    f"{previous_scale} and {manifest_scale}."
                )
            print(f"[WARNING] Asset {base} already exists at the same scale, skipping...")
            continue
        asset_scales_by_name[base] = manifest_scale

        usd_path = os.path.join(usd_dir, f"{base}", f"{base}.usd")
        obj_path = os.path.join(obj_dir, f"{base}.obj")

        if not os.path.exists(usd_path):
            if dict_schema:
                raise FileNotFoundError(
                    f"Object candidate {base!r} is missing its required USD: {usd_path}"
                )
            print(f"[WARNING] USD file not found: {usd_path}, skipping...")
            continue
        if dict_schema and not os.path.exists(obj_path):
            raise FileNotFoundError(
                f"Object candidate {base!r} is missing its required OBJ: {obj_path}"
            )

        usd_cfg = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(spawn_scale, spawn_scale, spawn_scale),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3)),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=_RL_CONTRACT.env.object_solver_position_iteration_count,
                solver_velocity_iteration_count=_RL_CONTRACT.env.object_solver_velocity_iteration_count,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=_RL_CONTRACT.env.max_depenetration_velocity,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=_RL_CONTRACT.env.contact_offset,
                rest_offset=_RL_CONTRACT.env.rest_offset,
            ),
        )
        usd_cfg.obj_path = obj_path
        assets.append(usd_cfg)
    return assets


# Helper for geometry/point-cloud caching, compatible with IsaacLab multi-env.
# Sampled points inside Cloud remain lazy; requesting vertices or stable poses
# does not load either point-cloud source.
def get_cached_cloud(
    obj_path,
    *,
    pointcloud_source: str = "mesh_sampled",
    preprocessed_pointcloud_path: str | Path | None = None,
    target_num_points: int = 512,
):
    key = (
        str(obj_path),
        str(pointcloud_source),
        str(preprocessed_pointcloud_path) if preprocessed_pointcloud_path is not None else None,
        int(target_num_points),
    )
    if key not in _CLOUD_CACHE:
        _CLOUD_CACHE[key] = Cloud(
            obj_path,
            target_num_points=target_num_points,
            pointcloud_source=pointcloud_source,
            preprocessed_pointcloud_path=preprocessed_pointcloud_path,
        )
    return _CLOUD_CACHE[key]


def get_cached_object_cloud(obj_path):
    """Return an object Cloud whose sampled points follow the RL observation config."""
    source = str(_RL_CONTRACT.observation.object_cloud_source)
    preprocessed_path = None
    if source == "preprocessed":
        pointcloud_dir = Path(
            _RL_CONTRACT.observation.object_cloud_preprocessed_dir
        ).expanduser()
        preprocessed_path = pointcloud_dir / (
            f"{Path(obj_path).stem}_first_hit_fps_"
            f"{int(_RL_CONTRACT.observation.num_points)}.npy"
        )
    return get_cached_cloud(
        obj_path,
        pointcloud_source=source,
        preprocessed_pointcloud_path=preprocessed_path,
        target_num_points=int(_RL_CONTRACT.observation.num_points),
    )


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

if _USE_WELDED_TOOL:
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

# Legacy single-tool aliases (index 0) for backward-compatible imports
TOOL_OBJ_PATH: str = TOOL_DATA[0]["obj_path"] if TOOL_DATA else ""
TOOL_HEAD_AREA_NORM = TOOL_DATA[0].get("head_area") if TOOL_DATA else None


def _path_relative_to_paths_yaml(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(_PATHS_CFG_FILE).parent / path).resolve()


def _generated_gripper_manifest_path() -> Path:
    section = _PATHS.get("generated_grippers")
    if not isinstance(section, Mapping):
        raise RuntimeError(
            "robot_mode=generated_gripper requires paths.yaml key "
            "generated_grippers.manifest pointing to an explicit generated-gripper manifest"
        )
    value = section.get("manifest")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            "robot_mode=generated_gripper requires paths.yaml key "
            "generated_grippers.manifest pointing to an explicit generated-gripper manifest"
        )
    return _path_relative_to_paths_yaml(value)


def _generated_gripper_root_path() -> Path:
    section = _PATHS.get("generated_grippers")
    if not isinstance(section, Mapping):
        raise RuntimeError(
            "robot_mode=generated_gripper requires paths.yaml key generated_grippers.root"
        )
    value = section.get("root")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            "robot_mode=generated_gripper requires paths.yaml key generated_grippers.root"
        )
    return _path_relative_to_paths_yaml(value)


def get_generated_gripper_cloud_cache_dir() -> Path:
    section = _PATHS.get("generated_grippers")
    if not isinstance(section, Mapping):
        raise RuntimeError(
            "robot_mode=generated_gripper requires paths.yaml section generated_grippers"
        )
    value = section.get("cloud_cache_dir")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            "robot_mode=generated_gripper requires "
            "generated_grippers.cloud_cache_dir"
        )
    path = _path_relative_to_paths_yaml(value)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Generated-gripper cloud cache directory does not exist: {path}"
        )
    return path


def _one_dof_gripper_path(key: str) -> Path:
    section = _PATHS.get("one_dof_grippers")
    if not isinstance(section, Mapping):
        raise RuntimeError("robot_mode=one_dof_gripper requires paths.yaml section one_dof_grippers")
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"robot_mode=one_dof_gripper requires one_dof_grippers.{key}")
    return _path_relative_to_paths_yaml(value)


def _require_uniform_generated_value(label: str, values: list):
    if not values:
        raise RuntimeError(f"No generated gripper metadata available for {label}")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise RuntimeError(
            f"generated_gripper requires a uniform {label} across manifest entries; "
            f"got {values!r}"
        )
    return first


def _generated_gripper_articulation_signature(asset: GeneratedGripperAsset):
    urdf_path = asset.root_dir / "isaac.urdf"
    if not urdf_path.is_file():
        raise RuntimeError(
            f"generated_gripper asset {asset.gripper_id!r} is missing required URDF: {urdf_path}"
        )
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(
            f"generated_gripper asset {asset.gripper_id!r} has invalid URDF: {urdf_path}"
        ) from exc

    links = tuple(link.get("name") for link in root.findall("link"))
    joints = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        joints.append(
            (
                joint.get("name"),
                joint.get("type"),
                None if parent is None else parent.get("link"),
                None if child is None else child.get("link"),
            )
        )
    return links, tuple(joints)


def _require_generated_usd_current(asset: GeneratedGripperAsset) -> None:
    urdf_path = asset.root_dir / "isaac.urdf"
    if not urdf_path.is_file():
        raise RuntimeError(
            f"generated_gripper asset {asset.gripper_id!r} is missing required URDF: {urdf_path}"
        )
    if not asset.usd_path.is_file():
        raise RuntimeError(
            f"generated_gripper asset {asset.gripper_id!r} is missing required USD: {asset.usd_path}"
        )
    if asset.usd_path.stat().st_mtime < urdf_path.stat().st_mtime:
        raise RuntimeError(
            f"generated_gripper asset {asset.gripper_id!r} has a stale USD converted before "
            f"its URDF was updated: usd={asset.usd_path}, urdf={urdf_path}. "
            "Rerun gripper/convert_urdf.py for generated_gripper before launching RL."
        )


def _require_uniform_generated_articulation_topology(
    assets: list[GeneratedGripperAsset],
    *,
    context: str,
) -> None:
    if len(assets) <= 1:
        return

    groups: dict[tuple, list[str]] = {}
    for asset in assets:
        _require_generated_usd_current(asset)
        signature = _generated_gripper_articulation_signature(asset)
        groups.setdefault(signature, []).append(asset.gripper_id)

    if len(groups) == 1:
        return

    previews = []
    for signature, ids in groups.items():
        links, joints = signature
        preview_ids = ", ".join(ids[:5])
        if len(ids) > 5:
            preview_ids += f", ... ({len(ids)} total)"
        previews.append(
            f"ids=[{preview_ids}] links={len(links)} joints={len(joints)}"
        )
    raise RuntimeError(
        "generated_gripper cannot spawn mixed articulation topologies in one IsaacLab "
        f"Articulation view ({context}). IsaacLab requires a shared PhysX articulation "
        "metatype; mixing grippers with different link/joint graphs causes "
        "root_physx_view.shared_metatype to be None. Regenerate the grippers with a "
        "uniform topology, or use a manifest containing one topology. Groups: "
        + "; ".join(previews)
    )


GENERATED_GRIPPER_DATA: list[GeneratedGripperAsset] = []
GENERATED_GRIPPER_USD_PATHS: list[str] = []
GENERATED_GRIPPER_ASSET_INDICES_BY_ENV: list[int] = []
GENERATED_GRIPPER_USD_PATHS_BY_ENV: list[str] = []
GENERATED_GRIPPER_SPAWN_ASSET_INDICES: list[int] = []
GENERATED_GRIPPER_USD_PATHS_FOR_SPAWN: list[str] = []
GENERATED_GRIPPER_FINGER_JOINT_NAMES: tuple[str, str] = ("", "")
GENERATED_GRIPPER_OPEN_JOINT_POS: float = 0.0
GENERATED_GRIPPER_EE_BODY_NAME: str = ""
ONE_DOF_GRIPPER_DATA: list[OneDofGripperAsset] = []
ONE_DOF_GRIPPER_ASSET_INDICES_BY_ENV: list[int] = []
ONE_DOF_GRIPPER_SPAWN_ASSET_INDICES: list[int] = []
ONE_DOF_GRIPPER_USD_PATHS_FOR_SPAWN: list[str] = []
ONE_DOF_GRIPPER_ACTUATED_JOINT_NAMES: tuple[str, ...] = ()
ONE_DOF_GRIPPER_OPEN_JOINT_POSITIONS: tuple[float, ...] = ()
ONE_DOF_GRIPPER_CLOSED_JOINT_POSITIONS: tuple[float, ...] = ()
ONE_DOF_GRIPPER_EE_BODY_NAME: str = ""

if _USE_GENERATED_GRIPPER:
    _manifest_path = _generated_gripper_manifest_path()
    GENERATED_GRIPPER_DATA = load_generated_gripper_manifest(
        _manifest_path,
        expected_root=_generated_gripper_root_path(),
    )
    GENERATED_GRIPPER_USD_PATHS = [str(entry.usd_path) for entry in GENERATED_GRIPPER_DATA]
    GENERATED_GRIPPER_FINGER_JOINT_NAMES = _require_uniform_generated_value(
        "finger_joint_names",
        [entry.finger_joint_names for entry in GENERATED_GRIPPER_DATA],
    )
    GENERATED_GRIPPER_OPEN_JOINT_POS = float(
        _require_uniform_generated_value(
            "open_joint_pos",
            [entry.open_joint_pos for entry in GENERATED_GRIPPER_DATA],
        )
    )
    GENERATED_GRIPPER_EE_BODY_NAME = str(
        _require_uniform_generated_value(
            "ee_body_name",
            [entry.ee_body_name for entry in GENERATED_GRIPPER_DATA],
        )
    )
    GENERATED_GRIPPER_ASSET_INDICES_BY_ENV = asset_indices_for_rank(
        _NUM_ENVS_PER_RANK,
        _GLOBAL_RANK,
        len(GENERATED_GRIPPER_DATA),
        randomize=_RANDOMIZE_TOOL_ASSIGNMENT,
        seed=_ASSET_ASSIGNMENT_SEED,
        salt=GENERATED_GRIPPER_ASSIGNMENT_SALT,
    )
    GENERATED_GRIPPER_USD_PATHS_BY_ENV = [
        GENERATED_GRIPPER_USD_PATHS[index] for index in GENERATED_GRIPPER_ASSET_INDICES_BY_ENV
    ]
    GENERATED_GRIPPER_SPAWN_ASSET_INDICES = (
        GENERATED_GRIPPER_ASSET_INDICES_BY_ENV
        if _RANDOMIZE_TOOL_ASSIGNMENT
        else sequential_spawn_indices_for_rank(
            _NUM_ENVS_PER_RANK,
            _GLOBAL_RANK,
            len(GENERATED_GRIPPER_DATA),
        )
    )
    GENERATED_GRIPPER_USD_PATHS_FOR_SPAWN = [
        GENERATED_GRIPPER_USD_PATHS[index] for index in GENERATED_GRIPPER_SPAWN_ASSET_INDICES
    ]
    _require_uniform_generated_articulation_topology(
        [GENERATED_GRIPPER_DATA[index] for index in sorted(set(GENERATED_GRIPPER_SPAWN_ASSET_INDICES))],
        context=f"rank={_GLOBAL_RANK} envs={_NUM_ENVS_PER_RANK}",
    )

if _USE_ONE_DOF_GRIPPER:
    ONE_DOF_GRIPPER_DATA = load_one_dof_gripper_manifest(
        _one_dof_gripper_path("manifest"),
        expected_root=_one_dof_gripper_path("root"),
        require_usd=True,
    )
    families = {asset.topology_family for asset in ONE_DOF_GRIPPER_DATA}
    signatures = {asset.topology_signature for asset in ONE_DOF_GRIPPER_DATA}
    if len(families) != 1 or len(signatures) != 1:
        raise RuntimeError(
            "one_dof_gripper requires one homogeneous topology family per Isaac process; "
            f"families={sorted(families)} topology_count={len(signatures)}"
        )
    ONE_DOF_GRIPPER_ACTUATED_JOINT_NAMES = _require_uniform_generated_value(
        "one_dof_gripper.actuated_joint_names",
        [asset.actuated_joint_names for asset in ONE_DOF_GRIPPER_DATA],
    )
    ONE_DOF_GRIPPER_OPEN_JOINT_POSITIONS = _require_uniform_generated_value(
        "one_dof_gripper.open_joint_positions",
        [asset.open_joint_positions for asset in ONE_DOF_GRIPPER_DATA],
    )
    ONE_DOF_GRIPPER_CLOSED_JOINT_POSITIONS = _require_uniform_generated_value(
        "one_dof_gripper.closed_joint_positions",
        [asset.closed_joint_positions for asset in ONE_DOF_GRIPPER_DATA],
    )
    ONE_DOF_GRIPPER_EE_BODY_NAME = _require_uniform_generated_value(
        "one_dof_gripper.ee_body_name",
        [asset.ee_body_name for asset in ONE_DOF_GRIPPER_DATA],
    )
    ONE_DOF_GRIPPER_ASSET_INDICES_BY_ENV = asset_indices_for_rank(
        _NUM_ENVS_PER_RANK,
        _GLOBAL_RANK,
        len(ONE_DOF_GRIPPER_DATA),
        randomize=_RANDOMIZE_TOOL_ASSIGNMENT,
        seed=_ASSET_ASSIGNMENT_SEED,
        salt=ONE_DOF_GRIPPER_ASSIGNMENT_SALT,
    )
    ONE_DOF_GRIPPER_SPAWN_ASSET_INDICES = (
        ONE_DOF_GRIPPER_ASSET_INDICES_BY_ENV
        if _RANDOMIZE_TOOL_ASSIGNMENT
        else sequential_spawn_indices_for_rank(
            _NUM_ENVS_PER_RANK,
            _GLOBAL_RANK,
            len(ONE_DOF_GRIPPER_DATA),
        )
    )
    ONE_DOF_GRIPPER_USD_PATHS_FOR_SPAWN = [
        str(ONE_DOF_GRIPPER_DATA[index].usd_path)
        for index in ONE_DOF_GRIPPER_SPAWN_ASSET_INDICES
    ]


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


def get_generated_gripper_index_for_env(env_id: int) -> int:
    """Return the generated gripper index assigned to a local env."""
    return _assigned_index_for_env(env_id, GENERATED_GRIPPER_ASSET_INDICES_BY_ENV, "generated gripper")


def get_generated_gripper_data_for_env(env_id: int) -> GeneratedGripperAsset:
    """Return per-gripper metadata for the given env_id."""
    return GENERATED_GRIPPER_DATA[get_generated_gripper_index_for_env(env_id)]


def get_one_dof_gripper_index_for_env(env_id: int) -> int:
    return _assigned_index_for_env(env_id, ONE_DOF_GRIPPER_ASSET_INDICES_BY_ENV, "one-DoF gripper")


def get_one_dof_gripper_data_for_env(env_id: int) -> OneDofGripperAsset:
    return ONE_DOF_GRIPPER_DATA[get_one_dof_gripper_index_for_env(env_id)]


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
official_panda_gripper_joint_init = custom_joint_init.copy()
official_panda_gripper_joint_init.update(
    {
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }
)
generated_gripper_joint_init = custom_joint_init.copy()
if _USE_GENERATED_GRIPPER:
    generated_gripper_joint_init.update(
        {
            GENERATED_GRIPPER_FINGER_JOINT_NAMES[0]: GENERATED_GRIPPER_OPEN_JOINT_POS,
            GENERATED_GRIPPER_FINGER_JOINT_NAMES[1]: GENERATED_GRIPPER_OPEN_JOINT_POS,
        }
    )
one_dof_gripper_joint_init = custom_joint_init.copy()
if _USE_ONE_DOF_GRIPPER:
    one_dof_gripper_joint_init.update(
        dict(zip(ONE_DOF_GRIPPER_ACTUATED_JOINT_NAMES, ONE_DOF_GRIPPER_OPEN_JOINT_POSITIONS))
    )
_bare_franka_usd = _PATHS.get("robot", {}).get("franka_usd", "")
bare_franka_path = os.path.abspath(_bare_franka_usd) if _bare_franka_usd else ""
if _USE_BARE_FRANKA and not bare_franka_path:
    raise ValueError("paths.yaml must define robot.franka_usd when robot_mode=bare_franka")
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
if _USE_OFFICIAL_PANDA_GRIPPER:
    _official_tools_cfg = _PATHS.get("tools")
    if not isinstance(_official_tools_cfg, Mapping) or not _official_tools_cfg.get("robots_usd_dir"):
        raise ValueError(
            "paths.yaml must define tools.robots_usd_dir when robot_mode=official_panda_gripper"
        )
    OFFICIAL_PANDA_GRIPPER_PROPS_DIR = os.path.abspath(
        os.path.join(_official_tools_cfg["robots_usd_dir"], "Props")
    )
else:
    OFFICIAL_PANDA_GRIPPER_PROPS_DIR = ""

if _USE_WELDED_TOOL:
    EE_TARGET_PRIM_PATH = "{ENV_REGEX_NS}/Robot/tool_mount/link_coacd_convex_piece_0"
    EE_TARGET_NAME = "ee_tool"
elif _USE_GENERATED_GRIPPER:
    if not GENERATED_GRIPPER_EE_BODY_NAME:
        raise RuntimeError("generated_gripper metadata must provide a non-empty ee_body_name")
    EE_TARGET_PRIM_PATH = f"{{ENV_REGEX_NS}}/Robot/{GENERATED_GRIPPER_EE_BODY_NAME}"
    EE_TARGET_NAME = "ee_generated_gripper"
elif _USE_ONE_DOF_GRIPPER:
    EE_TARGET_PRIM_PATH = f"{{ENV_REGEX_NS}}/Robot/{ONE_DOF_GRIPPER_EE_BODY_NAME}"
    EE_TARGET_NAME = "ee_one_dof_gripper"
else:
    EE_TARGET_PRIM_PATH = "{ENV_REGEX_NS}/Robot/panda_hand"
    EE_TARGET_NAME = "ee_hand"


def _build_gripper_robot_cfg(
    usd_paths: list[str], joint_names: Sequence[str], *, mode_name: str
) -> ArticulationCfg:
    """Build generated and one-DoF grippers from exactly the same runtime baseline."""
    if not usd_paths:
        raise RuntimeError(f"{mode_name} requires at least one robot USD path")
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
    base_spawn = robot_cfg.spawn
    robot_cfg.spawn = sim_utils.MultiUsdFileCfg(
        usd_path=usd_paths,
        random_choice=False,
        activate_contact_sensors=base_spawn.activate_contact_sensors,
        rigid_props=base_spawn.rigid_props,
        articulation_props=base_spawn.articulation_props,
        collision_props=base_spawn.collision_props,
        mass_props=base_spawn.mass_props,
        visual_material=base_spawn.visual_material,
        semantic_tags=base_spawn.semantic_tags,
    )
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot_cfg.actuators["panda_hand"].joint_names_expr = list(joint_names)
    return robot_cfg


def build_generated_gripper_robot_cfg(usd_paths: list[str]) -> ArticulationCfg:
    robot_cfg = _build_gripper_robot_cfg(
        usd_paths,
        GENERATED_GRIPPER_FINGER_JOINT_NAMES,
        mode_name="generated_gripper",
    )
    robot_cfg.actuators["panda_hand"].velocity_limit_sim = (
        _GENERATED_PARALLEL_FINGER_VELOCITY_LIMIT_M_S
    )
    return robot_cfg


def build_one_dof_gripper_robot_cfg(usd_paths: list[str]) -> ArticulationCfg:
    robot_cfg = _build_gripper_robot_cfg(
        usd_paths,
        ONE_DOF_GRIPPER_ACTUATED_JOINT_NAMES,
        mode_name="one_dof_gripper",
    )
    # Keep the arm identical to generated_gripper, while allowing each official
    # mechanism to carry conservative drive tuning in its reviewed manifest.
    actuator_specs = {asset.actuator for asset in ONE_DOF_GRIPPER_DATA}
    if len(actuator_specs) != 1:
        raise RuntimeError(
            "one_dof_gripper requires one actuator specification per Isaac process; "
            f"found {len(actuator_specs)}"
        )
    actuator_spec = next(iter(actuator_specs))
    hand_actuator = robot_cfg.actuators["panda_hand"]
    hand_actuator.effort_limit_sim = actuator_spec.effort_limit
    hand_actuator.stiffness = actuator_spec.stiffness
    hand_actuator.damping = actuator_spec.damping
    hand_actuator.armature = actuator_spec.armature
    hand_actuator.velocity_limit_sim = actuator_spec.velocity_limit
    return robot_cfg


def make_robot_cfg() -> ArticulationCfg:
    if _USE_ONE_DOF_GRIPPER:
        robot_cfg = build_one_dof_gripper_robot_cfg(ONE_DOF_GRIPPER_USD_PATHS_FOR_SPAWN)
    elif _USE_GENERATED_GRIPPER:
        robot_cfg = build_generated_gripper_robot_cfg(GENERATED_GRIPPER_USD_PATHS_FOR_SPAWN)
    elif _USE_BARE_FRANKA or _USE_OFFICIAL_PANDA_GRIPPER:
        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
        if _USE_BARE_FRANKA and os.path.isfile(bare_franka_path):
            robot_cfg.spawn.usd_path = bare_franka_path
        robot_cfg.spawn.rigid_props.disable_gravity = True
    else:
        robot_cfg = build_multi_tool_robot_cfg(TOOL_USD_PATHS_FOR_SPAWN, random_choice=False)

    robot_cfg.spawn.rigid_props.max_depenetration_velocity = _RL_CONTRACT.env.max_depenetration_velocity
    robot_cfg.spawn.articulation_props.solver_position_iteration_count = (
        _RL_CONTRACT.env.articulation_solver_position_iteration_count
    )
    robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = (
        _RL_CONTRACT.env.articulation_solver_velocity_iteration_count
    )
    robot_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
        contact_offset=_RL_CONTRACT.env.contact_offset,
        rest_offset=_RL_CONTRACT.env.rest_offset,
    )
    return robot_cfg.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=official_panda_gripper_joint_init
            if _USE_OFFICIAL_PANDA_GRIPPER
            else generated_gripper_joint_init
            if _USE_GENERATED_GRIPPER
            else one_dof_gripper_joint_init
            if _USE_ONE_DOF_GRIPPER
            else custom_joint_init
        ),
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
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=_RL_CONTRACT.env.contact_offset,
                    rest_offset=_RL_CONTRACT.env.rest_offset,
                ),
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
                solver_position_iteration_count=_RL_CONTRACT.env.object_solver_position_iteration_count,
                solver_velocity_iteration_count=_RL_CONTRACT.env.object_solver_velocity_iteration_count,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=_RL_CONTRACT.env.max_depenetration_velocity,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=_RL_CONTRACT.env.contact_offset,
                rest_offset=_RL_CONTRACT.env.rest_offset,
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
        # Video/interactive preference; the training launcher overrides it off.
        debug_vis=True,
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
class OfficialPandaGripperActionsCfg:
    """7D arm delta action plus 1D symmetric Panda gripper openness."""

    arm_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=_RL_CONTRACT.action.scale,
        use_zero_offset=True,
    )
    gripper_action = mdp.SymmetricPandaGripperActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        closed_joint_pos=0.0,
        open_joint_pos=0.04,
        clip=_RL_CONTRACT.action.clip,
    )


@configclass
class GeneratedGripperActionsCfg:
    """7D arm delta action plus 1D symmetric generated-gripper openness."""

    arm_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=_RL_CONTRACT.action.scale,
        use_zero_offset=True,
    )
    gripper_action = mdp.SymmetricGeneratedGripperActionCfg(
        asset_name="robot",
        joint_names=list(GENERATED_GRIPPER_FINGER_JOINT_NAMES),
        closed_joint_pos=0.0,
        open_joint_pos=GENERATED_GRIPPER_OPEN_JOINT_POS,
        clip=_RL_CONTRACT.action.clip,
        semantic_closure=_USE_CROSS_EMBODIMENT_GRIPPER,
    )


@configclass
class OneDofGripperActionsCfg:
    """7D Panda arm delta action plus one embodiment-independent closure command."""

    arm_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=_RL_CONTRACT.action.scale,
        use_zero_offset=True,
    )
    gripper_action = mdp.SemanticOneDofGripperActionCfg(
        asset_name="robot",
        joint_names=list(ONE_DOF_GRIPPER_ACTUATED_JOINT_NAMES),
        open_joint_positions=list(ONE_DOF_GRIPPER_OPEN_JOINT_POSITIONS),
        closed_joint_positions=list(ONE_DOF_GRIPPER_CLOSED_JOINT_POSITIONS),
        clip=_RL_CONTRACT.action.clip,
    )


ActionsCfg = (
    OfficialPandaGripperActionsCfg
    if _USE_OFFICIAL_PANDA_GRIPPER
    else GeneratedGripperActionsCfg
    if _USE_GENERATED_GRIPPER
    else OneDofGripperActionsCfg
    if _USE_ONE_DOF_GRIPPER
    else RelativeJointPositionActionsCfg
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
            noise=(
                GaussianNoiseCfg(mean=0.0, std=0.005, operation="add")
                if _RL_CONTRACT.observation.point_cloud_noise_enabled
                else None
            ),
        ) if _RL_CONTRACT.observation.include_object_cloud else None

        # Tool Cloud (512*3=1536D: tool point cloud xyz in env frame)
        tool_cloud = ObsTerm(
            func=mdp.get_tool_pointcloud_in_env_frame,
            noise=(
                GaussianNoiseCfg(mean=0.0, std=0.002, operation="add")
                if _RL_CONTRACT.observation.point_cloud_noise_enabled
                else None
            ),
        ) if (_RL_CONTRACT.observation.include_tool_cloud and not _USE_BARE_FRANKA) else None

        kinematic_gripper_clouds = ObsTerm(
            func=mdp.get_generated_gripper_kinematic_state_clouds,
        ) if _RL_CONTRACT.observation.include_kinematic_gripper_clouds else None

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
        
        # Robot State: 14D arm-only or 18D arm+gripper, per robot_mode.
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

        task_embedding = ObsTerm(
            func=mdp.target_pose_task_embedding,
            params={"command_name": "target_object_pose"},
        ) if _RL_CONTRACT.observation.task_embedding_dim > 0 else None

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

    preload_object_pointclouds = EventTerm(
        func=mdp.preload_object_pointclouds,
        mode="prestartup",
        params={
            "object_cloud_source": _RL_CONTRACT.observation.object_cloud_source,
            "preprocessed_dir": _RL_CONTRACT.observation.object_cloud_preprocessed_dir,
            "num_points": _RL_CONTRACT.observation.num_points,
            "asset_cfg": SceneEntityCfg("object"),
        },
    ) if _RL_CONTRACT.observation.include_object_cloud else None

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
    actions: ActionsCfg = ActionsCfg()
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
    num_points: int = _RL_CONTRACT.observation.num_points
    object_cloud_source: str = _RL_CONTRACT.observation.object_cloud_source
    object_cloud_preprocessed_dir: str = (
        _RL_CONTRACT.observation.object_cloud_preprocessed_dir
    )
    robot_mode: str = _ROBOT_MODE
    requested_robot_mode: str = _REQUESTED_ROBOT_MODE
    physics_observation_fields: tuple[str, ...] = _PHYSICS_OBSERVATION_FIELDS
    table_enabled: bool = _RL_CONTRACT.table.enabled
    table_size_xyz: tuple[float, float, float] = tuple(_RL_CONTRACT.table.size_xyz)
    table_pose_xyz: tuple[float, float, float] = tuple(_RL_CONTRACT.table.pose_xyz)
    table_bounds_xy: tuple[tuple[float, float], tuple[float, float]] = _table_bounds_xy()
    table_placement_margin_xy: float = _RL_CONTRACT.table.placement_margin_xy
    table_placement_max_attempts: int = _RL_CONTRACT.table.placement_max_attempts
    table_material = _RL_CONTRACT.table.material
    # Video/interactive visualization preferences. RL training overrides every
    # ``visualize_*`` field (and command ``debug_vis``) off before gym.make.
    visualize_current_object_pose: bool = True
    visualize_object_pointcloud: bool = False  # Enable object point cloud visualization for debug in first env
    visualize_tool_pointcloud: bool = (
        bool(_RL_CONTRACT.env.visualize_tool_pointcloud) and not _USE_BARE_FRANKA
    )
    visualize_head_area_center: bool = True
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
        self.viewer.eye = (2.0, 0.4, 0.65)
        # self.viewer.eye = (6, 0, 6)
        
        # Simulation settings - match reference config dt
        self.sim.dt = _RL_CONTRACT.env.sim_dt
        self.sim.render_interval = self.decimation
        
        # Physics settings - match reference config
        self.sim.physx.solver_position_iteration_count = _RL_CONTRACT.env.solver_position_iteration_count
        self.sim.physx.solver_velocity_iteration_count = _RL_CONTRACT.env.solver_velocity_iteration_count
        self.sim.physx.enable_ccd = _RL_CONTRACT.env.enable_ccd


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
        self.task_recent_success_windows = {}
        self.task_total_episodes = {}
        self.task_total_successes = {}

        # Global step counter for periodic debug prints
        self._global_step = 0

        self._fine_grained_timing_enabled = bool(
            getattr(_RL_CONTRACT.launch, "print_fine_grained_timing", False)
        )
        self._fine_grained_timing_seconds = {
            "total": 0.0,
            "action": 0.0,
            "recorder": 0.0,
            "physics": 0.0,
            "termination": 0.0,
            "reward": 0.0,
            "reset": 0.0,
            "command_events": 0.0,
            "observation": 0.0,
            "success_tracking": 0.0,
        }

        # Run post-init setup: physics settings, scale caching, head area offsets
        self.post_reset()

    def reset_fine_grained_timing(self):
        for key in self._fine_grained_timing_seconds:
            self._fine_grained_timing_seconds[key] = 0.0

    def _reset_idx(self, env_ids):
        """Reset environments and refresh only their cached physical parameters."""

        super()._reset_idx(env_ids)
        mdp.refresh_phys_params_cache(self, env_ids=env_ids)

    def _fine_grained_timing_sync(self):
        device = torch.device(self.device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _fine_grained_timing_start(self) -> float:
        self._fine_grained_timing_sync()
        return time.perf_counter()

    def _fine_grained_timing_stop(self, key: str, start: float):
        self._fine_grained_timing_sync()
        self._fine_grained_timing_seconds[key] += time.perf_counter() - start

    def fine_grained_timing_summary(self, iteration: int) -> str:
        total = self._fine_grained_timing_seconds["total"]
        keys = (
            "action",
            "recorder",
            "physics",
            "termination",
            "reward",
            "reset",
            "command_events",
            "observation",
            "success_tracking",
        )
        accounted = sum(self._fine_grained_timing_seconds[key] for key in keys)
        unaccounted = max(total - accounted, 0.0)
        details = " ".join(
            f"{key}={self._fine_grained_timing_seconds[key]:.3f}s"
            for key in keys
        )
        return (
            f"[EnvStepTiming][rank {_GLOBAL_RANK}/{_WORLD_SIZE}] iter={iteration} "
            f"total={total:.3f}s {details} unaccounted={unaccounted:.3f}s"
        )

    def _timed_manager_step(self, action):
        timing_start = self._fine_grained_timing_start()
        self.action_manager.process_action(action.to(self.device))
        self._fine_grained_timing_stop("action", timing_start)

        timing_start = self._fine_grained_timing_start()
        self.recorder_manager.record_pre_step()
        self._fine_grained_timing_stop("recorder", timing_start)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        timing_start = self._fine_grained_timing_start()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
        self._fine_grained_timing_stop("physics", timing_start)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        timing_start = self._fine_grained_timing_start()
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self._fine_grained_timing_stop("termination", timing_start)

        timing_start = self._fine_grained_timing_start()
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self._fine_grained_timing_stop("reward", timing_start)

        if len(self.recorder_manager.active_terms) > 0:
            timing_start = self._fine_grained_timing_start()
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
            self._fine_grained_timing_stop("recorder", timing_start)

        timing_start = self._fine_grained_timing_start()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)
        self._fine_grained_timing_stop("reset", timing_start)

        timing_start = self._fine_grained_timing_start()
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self._fine_grained_timing_stop("command_events", timing_start)

        timing_start = self._fine_grained_timing_start()
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self._fine_grained_timing_stop("observation", timing_start)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    def step(self, action):
        """Override step to track success rates."""
        total_timing_start = (
            self._fine_grained_timing_start()
            if self._fine_grained_timing_enabled
            else None
        )
        success_timing_start = (
            self._fine_grained_timing_start()
            if self._fine_grained_timing_enabled
            else None
        )
        command_term = self.command_manager.get_term("target_object_pose")
        task_index_before_step = getattr(command_term, "target_pose_task_index", None)
        task_names = tuple(getattr(command_term, "target_pose_task_names", ("stable_pose", "secondary_task")))
        if task_index_before_step is not None:
            task_index_before_step = task_index_before_step.clone()
        if success_timing_start is not None:
            self._fine_grained_timing_stop("success_tracking", success_timing_start)

        # Call parent step method
        if self._fine_grained_timing_enabled:
            obs, reward, terminated, truncated, info = self._timed_manager_step(action)
        else:
            obs, reward, terminated, truncated, info = super().step(action)

        success_timing_start = (
            self._fine_grained_timing_start()
            if self._fine_grained_timing_enabled
            else None
        )
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
                if task_index_before_step is not None:
                    task_idx = int(task_index_before_step[env_id].item())
                    task_name = task_names[task_idx] if 0 <= task_idx < len(task_names) else f"task_{task_idx}"
                    window = self.task_recent_success_windows.setdefault(task_name, deque(maxlen=100))
                    window.append(episode_success)
                    self.task_total_episodes[task_name] = self.task_total_episodes.get(task_name, 0) + 1
                    if episode_success:
                        self.task_total_successes[task_name] = self.task_total_successes.get(task_name, 0) + 1
            
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
                for task_name, window in self.task_recent_success_windows.items():
                    task_episodes = self.task_total_episodes.get(task_name, 0)
                    if task_episodes <= 0 or len(window) <= 0:
                        continue
                    task_successes = self.task_total_successes.get(task_name, 0)
                    self.extras["log"][f"success_rate/{task_name}"] = task_successes / task_episodes
                    self.extras["log"][f"recent_success_rate/{task_name}"] = sum(window) / len(window)
                    self.extras["log"][f"total_episodes/{task_name}"] = task_episodes
                    self.extras["log"][f"total_successes/{task_name}"] = task_successes

        if success_timing_start is not None:
            self._fine_grained_timing_stop("success_tracking", success_timing_start)
        if total_timing_start is not None:
            self._fine_grained_timing_stop("total", total_timing_start)

        return obs, reward, terminated, truncated, info

    def post_reset(self):
        # NOTE: _object_scales and _tool_scales removed — scales are now baked into
        # per-env sampled point clouds at init time (in get_object_pointcloud).

        # Welded tools use per-tool head-area offsets in the tool body frame.
        # Franka-only modes use their robot bodies directly for end-effector centers.
        if _USE_WELDED_TOOL:
            self._head_area_offsets = mdp.compute_head_area_offsets_from_usd(self)
        else:
            self._head_area_offsets = torch.zeros(self.num_envs, 3, device=self.device)
