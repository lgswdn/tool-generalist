from __future__ import annotations

from configs.config_contact_gen import ContactGenCfg
from contact_generation import batch_generate


class _FakeGeneratorConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_batch_config_physics_options_are_passed_to_contact_generator(tmp_path):
    cfg = ContactGenCfg()
    cfg.B = 2
    cfg.M = 3
    cfg.chunk_B = 4
    cfg.physics.t_stabilize = 31
    cfg.physics.t_postcontact = 37
    cfg.physics.object_mass_range = (1.0, 1.1)
    cfg.physics.tool_mass_range = (2.0, 2.1)
    cfg.physics.object_friction_range = (0.21, 0.22)
    cfg.physics.tool_friction_range = (0.31, 0.32)
    cfg.physics.ground_friction_range = (0.41, 0.42)
    cfg.physics.runner = "isaac"
    cfg.physics.stabilize_linear_velocity_eps = 0.004
    cfg.physics.stabilize_angular_velocity_eps = 0.005
    cfg.physics.post_delta_seed = 123
    cfg.physics.post_delta_translation_min = (-0.01, -0.02, 0.0)
    cfg.physics.post_delta_translation_max = (0.03, 0.04, 0.05)
    cfg.physics.post_delta_rotation_max_rad = 0.4
    cfg.physics.post_tool_reach_translation_eps = 0.006
    cfg.physics.post_tool_reach_rotation_eps_rad = 0.07
    cfg.physics.post_object_table_z_min = -0.001
    cfg.physics.post_linear_velocity_eps = 0.08
    cfg.physics.post_angular_velocity_eps = 0.09
    physics_options = batch_generate.physics_options_from_config(cfg)
    physics_options.update(
        {
            "debug_dir": str(tmp_path / "debug"),
            "headless": False,
            "close_after_run": True,
        }
    )

    captured = []

    def fake_optimize(config):
        captured.append(config)

    tools_meta = tmp_path / "tools_adjusted.json"
    tools_meta.write_text("[]", encoding="utf-8")

    assert batch_generate.run_pair(
        str(tmp_path / "meshdata_adjusted" / "tool_a" / "coacd" / "decomposed.obj"),
        str(tmp_path / "objects" / "object_a.obj"),
        "tool_a",
        "object_a",
        str(tmp_path / "out"),
        str(tools_meta),
        [0.1, 0.1, 0.1],
        0,
        cfg,
        pose_idx=5,
        num_poses=10,
        seed=77,
        physics_options=physics_options,
        generator_api=(_FakeGeneratorConfig, fake_optimize),
    )

    generated = captured[0]
    assert generated.physics_runner == "isaac"
    assert generated.t_stabilize == 31
    assert generated.t_postcontact == 37
    assert generated.object_mass_range == (1.0, 1.1)
    assert generated.tool_mass_range == (2.0, 2.1)
    assert generated.object_friction_range == (0.21, 0.22)
    assert generated.tool_friction_range == (0.31, 0.32)
    assert generated.ground_friction_range == (0.41, 0.42)
    assert generated.stabilize_linear_velocity_eps == 0.004
    assert generated.stabilize_angular_velocity_eps == 0.005
    assert generated.post_delta_seed == 123
    assert generated.post_delta_translation_min == (-0.01, -0.02, 0.0)
    assert generated.post_delta_translation_max == (0.03, 0.04, 0.05)
    assert generated.post_delta_rotation_max_rad == 0.4
    assert generated.post_tool_reach_translation_eps == 0.006
    assert generated.post_tool_reach_rotation_eps_rad == 0.07
    assert generated.post_object_table_z_min == -0.001
    assert generated.post_linear_velocity_eps == 0.08
    assert generated.post_angular_velocity_eps == 0.09
    assert generated.debug_dir.endswith("debug/tool_a/object_a_pose5")
    assert generated.headless is False
    assert generated.close_after_run is True
