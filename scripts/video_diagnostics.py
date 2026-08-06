"""Helpers for recording policy debug diagnostics into videos."""

from __future__ import annotations

from typing import Any

import torch


WRIST_DISTANCE_BODY_NAMES = ("panda_link5", "panda_link6", "panda_link7")
WRIST_SURFACE_BODY_NAMES = ("panda_link6", "panda_link7")


def _body_ids_for_names(base_env: Any, robot_name: str, body_names: tuple[str, ...]) -> tuple[list[int], list[str]]:
    cache_key = "_".join(body_names)
    cache_name = f"_recording_diag_{robot_name}_{cache_key}_body_ids"
    cached = getattr(base_env, cache_name, None)
    if cached is not None:
        return cached

    robot = base_env.scene[robot_name]
    available_names = [str(name) for name in robot.data.body_names]
    ids: list[int] = []
    resolved_names: list[str] = []
    for requested_name in body_names:
        if requested_name in available_names:
            ids.append(available_names.index(requested_name))
            resolved_names.append(requested_name)

    resolved = (ids, resolved_names)
    setattr(base_env, cache_name, resolved)
    return resolved


def _bimanual_wrist_min_distance(base_env: Any, env_i: int) -> tuple[float | None, str | None]:
    try:
        robot1 = base_env.scene["robot_1"]
        robot2 = base_env.scene["robot_2"]
    except (KeyError, AttributeError):
        return None, None

    ids1, names1 = _body_ids_for_names(base_env, "robot_1", WRIST_DISTANCE_BODY_NAMES)
    ids2, names2 = _body_ids_for_names(base_env, "robot_2", WRIST_DISTANCE_BODY_NAMES)
    if not ids1 or not ids2:
        return None, None

    pos1 = robot1.data.body_pos_w[env_i, ids1, :]
    pos2 = robot2.data.body_pos_w[env_i, ids2, :]
    distances = torch.cdist(pos1.unsqueeze(0), pos2.unsqueeze(0))[0]
    flat_index = int(torch.argmin(distances).detach().cpu())
    index2 = flat_index % len(ids2)
    index1 = flat_index // len(ids2)
    pair_name = f"{names1[index1]}-{names2[index2]}"
    return float(distances[index1, index2].detach().cpu()), pair_name


def _surface_z(base_env: Any) -> float:
    if bool(getattr(base_env.cfg, "table_enabled", False)):
        table_pose = getattr(base_env.cfg, "table_pose_xyz", (0.0, 0.0, -0.02))
        table_size = getattr(base_env.cfg, "table_size_xyz", (1.5, 1.5, 0.04))
        return float(table_pose[2]) + 0.5 * float(table_size[2])
    return 0.0


def _bimanual_wrist_surface_clearance(
    base_env: Any,
    env_i: int,
) -> tuple[float | None, str | None, float | None]:
    try:
        robot_names = ("robot_1", "robot_2")
        robots = [base_env.scene[name] for name in robot_names]
    except (KeyError, AttributeError):
        return None, None, None

    surface_z = _surface_z(base_env)
    best_clearance: float | None = None
    best_name: str | None = None
    for robot_name, robot in zip(robot_names, robots, strict=True):
        ids, names = _body_ids_for_names(base_env, robot_name, WRIST_SURFACE_BODY_NAMES)
        if not ids:
            continue
        clearances = robot.data.body_pos_w[env_i, ids, 2] - surface_z
        local_index = int(torch.argmin(clearances).detach().cpu())
        clearance = float(clearances[local_index].detach().cpu())
        if best_clearance is None or clearance < best_clearance:
            best_clearance = clearance
            best_name = f"{robot_name}:{names[local_index]}"
    return best_clearance, best_name, surface_z


def recording_debug_metrics(
    base_env: Any,
    env_id: int,
    reward_params: dict[str, Any] | None = None,
    *,
    command_name: str = "target_object_pose",
) -> dict[str, Any]:
    reward_params = reward_params or {}
    object_asset = base_env.scene["object"]
    env_i = int(env_id)

    command = base_env.command_manager.get_command(command_name)
    target_pos = command[env_i, :3]
    target_quat = command[env_i, 3:7]

    object_pos = object_asset.data.root_pos_w[env_i, :3] - base_env.scene.env_origins[env_i]
    object_quat = object_asset.data.root_quat_w[env_i]
    pos_distance = torch.norm(target_pos - object_pos)

    dot_product = torch.sum(object_quat * target_quat)
    dot_product = torch.clamp(torch.abs(dot_product), max=1.0)
    rot_distance = torch.clamp(2.0 * torch.acos(dot_product), max=torch.pi)

    pos_threshold = float(reward_params.get("success_threshold", 0.05))
    rot_threshold = float(reward_params.get("rotation_threshold", 0.1))
    wrist_warning_distance = float(reward_params.get("bimanual_arm_proximity_warning_distance", 0.20))
    wrist_failure_distance = float(reward_params.get("bimanual_arm_proximity_failure_distance", 0.15))
    wrist_surface_warning_height = float(reward_params.get("bimanual_wrist_surface_warning_height", 0.12))
    wrist_surface_contact_height = float(reward_params.get("bimanual_wrist_surface_contact_height", 0.06))
    dwell_steps = max(1, int(reward_params.get("stable_success_dwell_steps", 10)))
    dwell_counter = getattr(base_env, "_goal_pose_success_count", None)
    dwell_count = 0
    if dwell_counter is not None and int(dwell_counter.shape[0]) > env_i:
        dwell_count = int(dwell_counter[env_i].detach().cpu())

    pos_value = float(pos_distance.detach().cpu())
    rot_value = float(rot_distance.detach().cpu())
    pose_ok = pos_value < pos_threshold and rot_value < rot_threshold
    wrist_min_distance, wrist_min_pair = _bimanual_wrist_min_distance(base_env, env_i)
    wrist_surface_clearance, wrist_surface_link, surface_z = _bimanual_wrist_surface_clearance(base_env, env_i)

    return {
        "env_id": env_i,
        "pos_distance": pos_value,
        "rot_distance": rot_value,
        "pos_threshold": pos_threshold,
        "rot_threshold": rot_threshold,
        "dwell_count": dwell_count,
        "dwell_steps": dwell_steps,
        "pose_ok": pose_ok,
        "success_now": pose_ok and dwell_count >= dwell_steps,
        "wrist_min_distance": wrist_min_distance,
        "wrist_min_pair": wrist_min_pair,
        "wrist_warning_distance": wrist_warning_distance,
        "wrist_failure_distance": wrist_failure_distance,
        "wrist_surface_clearance": wrist_surface_clearance,
        "wrist_surface_link": wrist_surface_link,
        "wrist_surface_z": surface_z,
        "wrist_surface_warning_height": wrist_surface_warning_height,
        "wrist_surface_contact_height": wrist_surface_contact_height,
    }


def format_recording_diagnostics(metrics: dict[str, Any], *, step: int | None = None) -> str:
    prefix = f"step={step} " if step is not None else ""
    text = (
        f"{prefix}env={metrics['env_id']} "
        f"pos={metrics['pos_distance']:.4f}/{metrics['pos_threshold']:.4f} "
        f"rot={metrics['rot_distance']:.4f}/{metrics['rot_threshold']:.4f} "
        f"dwell={metrics['dwell_count']}/{metrics['dwell_steps']} "
        f"pose={'Y' if metrics['pose_ok'] else 'N'} "
        f"success={'Y' if metrics['success_now'] else 'N'}"
    )
    wrist_min_distance = metrics.get("wrist_min_distance")
    if wrist_min_distance is not None:
        text += (
            f" wrist_min={wrist_min_distance:.4f}/"
            f"{metrics.get('wrist_warning_distance', 0.20):.4f}/"
            f"{metrics.get('wrist_failure_distance', 0.15):.4f}"
        )
    wrist_surface_clearance = metrics.get("wrist_surface_clearance")
    if wrist_surface_clearance is not None:
        text += (
            f" wrist_surface={wrist_surface_clearance:.4f}/"
            f"{metrics.get('wrist_surface_warning_height', 0.12):.4f}/"
            f"{metrics.get('wrist_surface_contact_height', 0.06):.4f}"
        )
    output_gate_expert_a = metrics.get("output_gate_expert_a_weight")
    if output_gate_expert_a is not None:
        text += (
            f" gate={metrics.get('output_gate_selected_expert', 'unknown')}"
            f"(a={float(output_gate_expert_a):.3f},"
            f"b={float(metrics.get('output_gate_expert_b_weight', 1.0 - float(output_gate_expert_a))):.3f})"
        )
    return text


def overlay_recording_diagnostics(frame: Any, metrics: dict[str, Any], *, step: int | None = None) -> Any:
    try:
        import cv2
    except ImportError:
        return frame

    lines = [
        f"env {metrics['env_id']} step {step if step is not None else '-'}",
        (
            f"goal pos {metrics['pos_distance']:.3f}/{metrics['pos_threshold']:.3f} m  "
            f"rot {metrics['rot_distance']:.3f}/{metrics['rot_threshold']:.3f} rad"
        ),
        f"dwell {metrics['dwell_count']}/{metrics['dwell_steps']} steps in threshold",
    ]
    wrist_min_distance = metrics.get("wrist_min_distance")
    if wrist_min_distance is not None:
        wrist_min_pair = metrics.get("wrist_min_pair") or "unknown"
        wrist_warning_distance = float(metrics.get("wrist_warning_distance", 0.20))
        wrist_failure_distance = float(metrics.get("wrist_failure_distance", 0.15))
        lines.append(
            f"wrist/forearm min {wrist_min_distance:.3f} m  "
            f"warn {wrist_warning_distance:.3f} fail {wrist_failure_distance:.3f}  pair {wrist_min_pair}"
        )
    wrist_surface_clearance = metrics.get("wrist_surface_clearance")
    if wrist_surface_clearance is not None:
        wrist_surface_link = metrics.get("wrist_surface_link") or "unknown"
        wrist_surface_z = float(metrics.get("wrist_surface_z", 0.0))
        wrist_surface_warning_height = float(metrics.get("wrist_surface_warning_height", 0.12))
        wrist_surface_contact_height = float(metrics.get("wrist_surface_contact_height", 0.06))
        lines.append(
            f"wrist surface {wrist_surface_clearance:.3f} m above z={wrist_surface_z:.3f}  "
            f"warn {wrist_surface_warning_height:.3f} contact {wrist_surface_contact_height:.3f}  "
            f"link {wrist_surface_link}"
        )
    output_gate_expert_a = metrics.get("output_gate_expert_a_weight")
    if output_gate_expert_a is not None:
        expert_a = float(output_gate_expert_a)
        expert_b = float(metrics.get("output_gate_expert_b_weight", 1.0 - expert_a))
        gate_range = ""
        gate_min = metrics.get("output_gate_expert_a_min")
        gate_max = metrics.get("output_gate_expert_a_max")
        if gate_min is not None and gate_max is not None and abs(float(gate_max) - float(gate_min)) > 1.0e-6:
            gate_range = f"  range {float(gate_min):.2f}-{float(gate_max):.2f}"
        lines.append(
            f"output gate {metrics.get('output_gate_selected_expert', 'unknown')}  "
            f"model_a {expert_a:.3f}  model_b {expert_b:.3f}{gate_range}"
        )
    lines.append(
        (
            f"pose {'OK' if metrics['pose_ok'] else 'NO'}  "
            f"success {'OK' if metrics['success_now'] else 'NO'}"
        )
    )
    height, width = frame.shape[:2]
    font_scale = max(0.42, min(width, height) / 900.0)
    thickness = 1 if min(width, height) < 700 else 2
    line_height = int(22 * font_scale) + 8
    box_height = 12 + line_height * len(lines)
    cv2.rectangle(frame, (6, 6), (width - 6, box_height), (0, 0, 0), thickness=-1)
    for index, line in enumerate(lines):
        y = 26 + index * line_height
        cv2.putText(
            frame,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return frame
