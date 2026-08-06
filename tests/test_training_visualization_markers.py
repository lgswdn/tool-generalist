from __future__ import annotations

from types import SimpleNamespace

from utils.experiment.isaac_rl_launcher import _disable_training_visualization_markers


def test_training_disables_markers_without_requiring_video_preferences_to_be_false():
    target_command = SimpleNamespace(debug_vis=True)
    already_hidden_command = SimpleNamespace(debug_vis=False)
    env_cfg = SimpleNamespace(
        visualize_current_object_pose=True,
        visualize_tool_pointcloud=True,
        visualize_head_area_center=True,
        visualize_object_pointcloud=False,
        commands=SimpleNamespace(
            target_object_pose=target_command,
            already_hidden=already_hidden_command,
        ),
        unrelated_flag=True,
    )

    disabled = _disable_training_visualization_markers(env_cfg)

    assert env_cfg.visualize_current_object_pose is False
    assert env_cfg.visualize_tool_pointcloud is False
    assert env_cfg.visualize_head_area_center is False
    assert env_cfg.visualize_object_pointcloud is False
    assert target_command.debug_vis is False
    assert already_hidden_command.debug_vis is False
    assert env_cfg.unrelated_flag is True
    assert disabled == [
        "visualize_current_object_pose",
        "visualize_head_area_center",
        "visualize_tool_pointcloud",
        "commands.target_object_pose.debug_vis",
    ]
