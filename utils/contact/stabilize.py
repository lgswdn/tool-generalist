"""Stabilize and physics-runner contract for contact generation.

The real Isaac path is loaded only from ``IsaacPhysicsRunner.run``.  Importing
this module does not import torch, Isaac, Omniverse, Kaolin, or trimesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Protocol, Sequence


def _torch():
    import torch

    return torch


def _log(message: str) -> None:
    print(f"[contact_generation.physics] {message}", flush=True)


@dataclass(frozen=True)
class PhysicsRunConfig:
    t_stabilize: int = 120
    t_postcontact: int = 120
    run_postcontact: bool = True
    require_stabilized_contact: bool = True
    runner: str = "isaac"
    unsigned_distance_accept_eps: float = 0.005
    post_delta_seed: int = 0
    post_delta_translation_min: Sequence[float] = (-0.02, -0.02, 0.0)
    post_delta_translation_max: Sequence[float] = (0.02, 0.02, 0.04)
    post_delta_rotation_max_rad: float = 0.25
    post_tool_reach_translation_eps: float = 2e-3
    post_tool_reach_rotation_eps_rad: float = 5e-2
    post_object_table_z_min: float = 0.0
    object_mesh_path: str | None = None
    tool_mesh_path: str | None = None
    object_scale: float | Sequence[float] = 1.0
    tool_scale_xyz: Sequence[float] = (1.0, 1.0, 1.0)
    env_spacing: float = 2.0
    visualization_enabled: bool = False
    visualization_stabilization_picture: bool = False
    visualization_stabilization_picture_num: int = 8
    visualization_postcontact_video: bool = False
    visualization_postcontact_video_num: int = 8
    visualization_video_dir: str | None = None
    visualization_picture_dir: str | None = None
    visualization_video_width: int = 640
    visualization_video_height: int = 480
    visualization_video_fps: int = 30
    visualization_camera_pos: Sequence[float] = (0.24, 0.14, 0.18)
    visualization_camera_target: Sequence[float] = (0.0, 0.0, 0.05)
    visualization_max_candidates: int = 1
    debug_dir: str | None = None
    headless: bool = True
    close_after_run: bool = False


@dataclass
class PhysicsBatchResult:
    success_mask: Any
    stabilized_in_contact: Any
    stabilized_contact_count: Any
    stabilized_contact_impulse_norm: Any
    stabilized_unsigned_distance_min: Any
    stabilize_steps: Any
    post_tool_delta_pose9d_E: Any
    post_tool_achieved_delta_pose9d_E: Any
    post_object_delta_pose9d_E: Any
    postcontact_steps: Any
    status: str
    runner: str
    is_real_physics: bool = False
    stabilized_candidates: Mapping[str, Any] | None = None
    failure_reasons: Sequence[str] | None = None
    stage_usd_path: str | None = None
    debug_json_path: str | None = None
    video_paths: Sequence[str] = field(default_factory=list)
    video_metadata: Sequence[Mapping[str, Any]] = field(default_factory=list)
    visualization_timeline: Sequence[Mapping[str, Any]] = field(default_factory=list)
    snapshot_paths: Sequence[str] = field(default_factory=list)
    debug_paths: Mapping[str, str] = field(default_factory=dict)


class PhysicsRunner:
    name = "base"
    is_real_physics = False

    def run(
        self,
        candidates: Mapping[str, Any],
        physical_props: Mapping[str, Any],
        cfg: PhysicsRunConfig,
    ) -> PhysicsBatchResult:
        raise NotImplementedError


@dataclass
class IsaacCandidateResult:
    success: bool
    status: str
    stabilize_steps: int = 0
    stabilized_in_contact: bool = False
    stabilized_contact_count: int = 0
    stabilized_contact_impulse_norm: float = 0.0
    stabilized_unsigned_distance_min: float | None = None
    post_tool_delta_pose9d_E: Sequence[float] | Any | None = None
    post_tool_achieved_delta_pose9d_E: Sequence[float] | Any | None = None
    post_object_delta_pose9d_E: Sequence[float] | Any | None = None
    postcontact_steps: int = 0
    stabilized: Mapping[str, Sequence[float] | Any] | None = None
    stage_usd_path: str | None = None
    debug_json_path: str | None = None
    video_path: str | None = None
    snapshot_paths: Sequence[str] = field(default_factory=list)
    debug_paths: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)


class IsaacContactAdapter(Protocol):
    is_real_physics: bool

    def run_batch(
        self,
        *,
        candidates: Mapping[str, Any],
        physical_props: Mapping[str, Any],
        cfg: PhysicsRunConfig,
        commanded_tool_delta_pose9d_E: Any,
    ) -> Sequence[IsaacCandidateResult]:
        ...

    def close(self) -> None:
        ...


REQUIRED_CANDIDATE_FIELDS = (
    "object_rotation_E",
    "object_bbox_center_E",
    "tool_translation_E",
    "tool_rotation_E",
    "contact_point_E",
)

REQUIRED_PHYSICAL_FIELDS = (
    "object_mass",
    "tool_mass",
    "object_friction",
    "tool_friction",
    "ground_friction",
)


def _empty_result(n: int, cfg: PhysicsRunConfig, status: str, runner: str) -> PhysicsBatchResult:
    torch = _torch()
    return PhysicsBatchResult(
        success_mask=torch.zeros(n, dtype=torch.bool),
        stabilized_in_contact=torch.zeros(n, dtype=torch.bool),
        stabilized_contact_count=torch.zeros(n, dtype=torch.int64),
        stabilized_contact_impulse_norm=torch.zeros(n, dtype=torch.float32),
        stabilized_unsigned_distance_min=torch.full((n,), float("nan"), dtype=torch.float32),
        stabilize_steps=torch.zeros(n, dtype=torch.int64),
        post_tool_delta_pose9d_E=torch.zeros(n, 9, dtype=torch.float32),
        post_tool_achieved_delta_pose9d_E=torch.zeros(n, 9, dtype=torch.float32),
        post_object_delta_pose9d_E=torch.zeros(n, 9, dtype=torch.float32),
        postcontact_steps=torch.zeros(n, dtype=torch.int64),
        status=status,
        runner=runner,
        is_real_physics=False,
    )


def _as_float_tensor(value: Sequence[float] | Any, shape: tuple[int, ...], key: str):
    torch = _torch()
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{key} must have shape {shape}, got {tuple(tensor.shape)}")
    return tensor


def _rotation_z(angle):
    torch = _torch()
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)
    return torch.stack(
        (
            torch.stack((cos, -sin, zeros), dim=-1),
            torch.stack((sin, cos, zeros), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ),
        dim=-2,
    )


def pose9d(translation, rotation):
    torch = _torch()
    rot6 = rotation[..., :, :2].reshape(*rotation.shape[:-2], 6)
    return torch.cat((translation, rot6), dim=-1).to(dtype=torch.float32)


def sample_commanded_tool_delta_pose9d(n: int, cfg: PhysicsRunConfig):
    torch = _torch()
    if n <= 0:
        return torch.zeros(0, 9, dtype=torch.float32)
    lo = _as_float_tensor(cfg.post_delta_translation_min, (3,), "post_delta_translation_min")
    hi = _as_float_tensor(cfg.post_delta_translation_max, (3,), "post_delta_translation_max")
    if bool((hi < lo).any()):
        raise ValueError("post_delta_translation_max must be >= post_delta_translation_min")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(cfg.post_delta_seed))
    translation = lo + (hi - lo) * torch.rand((n, 3), generator=rng, dtype=torch.float32)
    angle_max = float(cfg.post_delta_rotation_max_rad)
    angles = torch.empty(n, dtype=torch.float32).uniform_(-angle_max, angle_max, generator=rng)
    return pose9d(translation, _rotation_z(angles))


def validate_batch_inputs(candidates: Mapping[str, Any], physical_props: Mapping[str, Any]) -> int:
    torch = _torch()
    missing = [key for key in REQUIRED_CANDIDATE_FIELDS if key not in candidates]
    if missing:
        raise KeyError(f"Missing candidate fields: {missing}")
    missing_props = [key for key in REQUIRED_PHYSICAL_FIELDS if key not in physical_props]
    if missing_props:
        raise KeyError(f"Missing physical property fields: {missing_props}")

    n = int(candidates["tool_translation_E"].shape[0])
    for key in REQUIRED_CANDIDATE_FIELDS:
        value = candidates[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")
        if int(value.shape[0]) != n:
            raise ValueError(f"{key} first dimension must be {n}, got {value.shape[0]}")
    for key in REQUIRED_PHYSICAL_FIELDS:
        value = physical_props[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")
        if int(value.shape[0]) != n:
            raise ValueError(f"{key} first dimension must be {n}, got {value.shape[0]}")
    return n


def _candidate_at(candidates: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        key: candidates[key][index].detach().cpu().to(dtype=_torch().float32)
        for key in REQUIRED_CANDIDATE_FIELDS
    }


def _props_at(physical_props: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        key: physical_props[key][index].detach().cpu().to(dtype=_torch().float32)
        for key in REQUIRED_PHYSICAL_FIELDS
    }


def _candidate_is_finite(candidate: Mapping[str, Any], props: Mapping[str, Any]) -> bool:
    torch = _torch()
    tensors = list(candidate.values()) + list(props.values())
    return all(bool(torch.isfinite(t.float()).all()) for t in tensors)


class IsaacPhysicsRunner(PhysicsRunner):
    """Delegate stabilize/post-contact simulation to a lazy Isaac adapter."""

    name = "isaac"

    def __init__(self, adapter: IsaacContactAdapter | None = None):
        self._adapter = adapter

    def _load_adapter(self, cfg: PhysicsRunConfig) -> IsaacContactAdapter:
        if self._adapter is None:
            from .isaac import IsaacSimAdapter

            self._adapter = IsaacSimAdapter(
                headless=cfg.headless,
                debug_log=lambda message: _log(f"[ISAAC] {message}"),
            )
        return self._adapter

    def close(self) -> None:
        adapter = self._adapter
        if adapter is not None and hasattr(adapter, "close"):
            adapter.close()
        self._adapter = None

    def run(
        self,
        candidates: Mapping[str, Any],
        physical_props: Mapping[str, Any],
        cfg: PhysicsRunConfig,
    ) -> PhysicsBatchResult:
        torch = _torch()
        n = validate_batch_inputs(candidates, physical_props)
        if n == 0:
            return _empty_result(0, cfg, "physics_no_candidates", self.name)

        adapter = self._load_adapter(cfg)
        commanded_deltas = sample_commanded_tool_delta_pose9d(n, cfg)

        success_mask = torch.zeros(n, dtype=torch.bool)
        stabilized_in_contact = torch.zeros(n, dtype=torch.bool)
        stabilized_contact_count = torch.zeros(n, dtype=torch.int64)
        stabilized_contact_impulse_norm = torch.zeros(n, dtype=torch.float32)
        stabilized_unsigned_distance_min = torch.full((n,), float("nan"), dtype=torch.float32)
        stabilize_steps = torch.zeros(n, dtype=torch.int64)
        post_tool_delta_pose9d_E = torch.zeros(n, 9, dtype=torch.float32)
        post_tool_achieved_delta_pose9d_E = torch.zeros(n, 9, dtype=torch.float32)
        post_object_delta_pose9d_E = torch.zeros(n, 9, dtype=torch.float32)
        postcontact_steps = torch.zeros(n, dtype=torch.int64)
        stabilized = {
            key: candidates[key].detach().cpu().clone().to(dtype=torch.float32)
            for key in REQUIRED_CANDIDATE_FIELDS
        }
        failure_reasons = ["not_run"] * n
        stage_usd_path: str | None = None
        debug_json_path: str | None = None
        snapshot_paths: list[str] = []
        video_metadata: list[Mapping[str, Any]] = []
        visualization_timeline: list[Mapping[str, Any]] = []
        debug_paths: dict[str, str] = {}

        run_batch = getattr(adapter, "run_batch", None)
        if not callable(run_batch):
            raise TypeError("Isaac contact physics requires an adapter.run_batch implementation")

        _log(
            "[PHYSICS-BATCH-START] "
            f"runner={self.name} candidates={n} run_postcontact={cfg.run_postcontact} "
            f"t_stabilize={cfg.t_stabilize} t_postcontact={cfg.t_postcontact}"
        )
        batch_start = time.monotonic()
        results = list(
            run_batch(
                candidates=candidates,
                physical_props=physical_props,
                cfg=cfg,
                commanded_tool_delta_pose9d_E=commanded_deltas,
            )
        )
        if len(results) != n:
            raise ValueError(f"adapter.run_batch returned {len(results)} results for {n} candidates")

        batch_elapsed = time.monotonic() - batch_start
        for index, result in enumerate(results):
            stabilized_in_contact[index] = bool(result.stabilized_in_contact)
            stabilized_contact_count[index] = int(result.stabilized_contact_count)
            stabilized_contact_impulse_norm[index] = float(result.stabilized_contact_impulse_norm)
            if result.stabilized_unsigned_distance_min is not None:
                stabilized_unsigned_distance_min[index] = float(result.stabilized_unsigned_distance_min)
            stabilize_steps[index] = int(result.stabilize_steps)
            postcontact_steps[index] = int(result.postcontact_steps)
            post_tool_delta_pose9d_E[index] = (
                _as_float_tensor(result.post_tool_delta_pose9d_E, (9,), "post_tool_delta_pose9d_E")
                if result.post_tool_delta_pose9d_E is not None
                else commanded_deltas[index]
            )
            if result.post_tool_achieved_delta_pose9d_E is not None:
                post_tool_achieved_delta_pose9d_E[index] = _as_float_tensor(
                    result.post_tool_achieved_delta_pose9d_E,
                    (9,),
                    "post_tool_achieved_delta_pose9d_E",
                )
            else:
                post_tool_achieved_delta_pose9d_E[index] = post_tool_delta_pose9d_E[index]
            if result.post_object_delta_pose9d_E is not None:
                post_object_delta_pose9d_E[index] = _as_float_tensor(
                    result.post_object_delta_pose9d_E, (9,), "post_object_delta_pose9d_E"
                )
            if result.stage_usd_path:
                stage_usd_path = str(result.stage_usd_path)
            if result.debug_json_path:
                debug_json_path = str(result.debug_json_path)
            if result.video_path:
                snapshot_paths.append(str(result.video_path))
            result_metrics = {} if result.metrics is None else dict(result.metrics)
            video_info = result_metrics.get("video")
            if isinstance(video_info, Mapping):
                video_metadata.append({str(key): value for key, value in video_info.items()})
            timeline_info = result_metrics.get("visualization_timeline")
            if isinstance(timeline_info, Sequence) and not isinstance(timeline_info, (str, bytes)):
                visualization_timeline.extend(
                    dict(item) for item in timeline_info if isinstance(item, Mapping)
                )
            snapshot_paths.extend(str(path) for path in result.snapshot_paths)
            debug_paths.update({str(key): str(value) for key, value in result.debug_paths.items()})

            if result.stabilized:
                for key in REQUIRED_CANDIDATE_FIELDS:
                    if key in result.stabilized:
                        stabilized[key][index] = _as_float_tensor(
                            result.stabilized[key],
                            tuple(stabilized[key][index].shape),
                            f"stabilized.{key}",
                        )

            finite_outputs = all(
                bool(torch.isfinite(tensor).all())
                for tensor in (
                    post_tool_delta_pose9d_E[index],
                    post_tool_achieved_delta_pose9d_E[index],
                    post_object_delta_pose9d_E[index],
                )
            )
            stable_outputs = all(bool(torch.isfinite(stabilized[key][index]).all()) for key in stabilized)
            if not bool(result.success):
                passes = False
                reason = str(result.status or "physics_rejected")
            elif not finite_outputs:
                passes = False
                reason = "non_finite_physics_output"
            elif not stable_outputs:
                passes = False
                reason = "non_finite_stabilized_pose"
            elif bool(cfg.require_stabilized_contact) and not bool(stabilized_in_contact[index]):
                passes = False
                reason = str(result.status or "stabilize_unsigned_distance_exceeded")
            elif bool(cfg.require_stabilized_contact) and int(stabilize_steps[index]) <= 0:
                passes = False
                reason = str(result.status or "stabilize_zero_steps")
            else:
                passes = True
                reason = "success"
            success_mask[index] = passes
            failure_reasons[index] = reason

        if cfg.close_after_run and hasattr(adapter, "close"):
            adapter.close()

        any_success = bool(success_mask.any().item())
        is_real = bool(getattr(adapter, "is_real_physics", False))
        status = "complete" if any_success and is_real else "physics_failed"
        _log(
            "[PHYSICS-BATCH-DONE] "
            f"runner={self.name} status={status} successes={int(success_mask.sum().item())}/{n} "
            f"sim_and_result_elapsed_s={time.monotonic() - batch_start:.3f} "
            f"sim_batch_elapsed_s={batch_elapsed:.3f}"
        )
        return PhysicsBatchResult(
            success_mask=success_mask,
            stabilized_in_contact=stabilized_in_contact,
            stabilized_contact_count=stabilized_contact_count,
            stabilized_contact_impulse_norm=stabilized_contact_impulse_norm,
            stabilized_unsigned_distance_min=stabilized_unsigned_distance_min,
            stabilize_steps=stabilize_steps,
            post_tool_delta_pose9d_E=post_tool_delta_pose9d_E,
            post_tool_achieved_delta_pose9d_E=post_tool_achieved_delta_pose9d_E,
            post_object_delta_pose9d_E=post_object_delta_pose9d_E,
            postcontact_steps=postcontact_steps,
            status=status,
            runner=self.name,
            is_real_physics=is_real,
            stabilized_candidates=stabilized,
            failure_reasons=failure_reasons,
            stage_usd_path=stage_usd_path,
            debug_json_path=debug_json_path,
            video_paths=[path for path in snapshot_paths if str(path).endswith(".mp4")],
            video_metadata=video_metadata,
            visualization_timeline=visualization_timeline,
            snapshot_paths=snapshot_paths,
            debug_paths=debug_paths,
        )


def get_physics_runner(name: str) -> PhysicsRunner:
    normalized = (name or "isaac").lower()
    if normalized == "isaac":
        return IsaacPhysicsRunner()
    raise ValueError(f"Unknown physics runner '{name}'")


def _uniform(rng: Any, n: int, lo: float, hi: float):
    torch = _torch()
    return torch.empty(n, dtype=torch.float32).uniform_(float(lo), float(hi), generator=rng)


def sample_physical_properties(
    n: int,
    seed: int,
    object_mass_range: Sequence[float] = (0.05, 0.5),
    tool_mass_range: Sequence[float] = (0.05, 0.5),
    object_friction_range: Sequence[float] = (0.4, 1.2),
    tool_friction_range: Sequence[float] = (0.4, 1.2),
    ground_friction_range: Sequence[float] = (0.4, 1.2),
) -> dict[str, Any]:
    torch = _torch()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    return {
        "object_mass": _uniform(rng, n, *object_mass_range),
        "tool_mass": _uniform(rng, n, *tool_mass_range),
        "object_friction": _uniform(rng, n, *object_friction_range),
        "tool_friction": _uniform(rng, n, *tool_friction_range),
        "ground_friction": _uniform(rng, n, *ground_friction_range),
    }
