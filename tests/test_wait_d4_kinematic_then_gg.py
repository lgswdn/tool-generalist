from __future__ import annotations

import runpy


def test_waiter_maps_all_d4_kinematic_parents_to_their_gg_children():
    namespace = runpy.run_path("scripts/wait_unicorn_full_yes_then_gg.py")
    variants = namespace["VARIANTS"]
    expected = {
        "prl_d4_kinematic_concavity": (
            "ce_prl_unicorn_d4_full_nonpenetrating_contact_concavity_biased_kinematic_dgn_5k",
            "ce_prl_unicorn_d4_full_nonpenetrating_contact_concavity_biased_kinematic_gg_15k",
        ),
        "prl_d4_kinematic_paper": (
            "ce_prl_unicorn_d4_full_paper_contact_kinematic_dgn_5k",
            "ce_prl_unicorn_d4_full_paper_contact_kinematic_gg_15k",
        ),
        "prl_d4_kinematic_paper_head": (
            "ce_prl_unicorn_d4_full_paper_head_kinematic_dgn_5k",
            "ce_prl_unicorn_d4_full_paper_head_kinematic_gg_15k",
        ),
    }
    for variant, pair in expected.items():
        assert variants[variant][:2] == pair
        assert variants[variant][4] == 200


def test_waiter_maps_frozen_ggbest_pointnet_parent_to_gg_child():
    namespace = runpy.run_path("scripts/wait_unicorn_full_yes_then_gg.py")
    assert namespace["VARIANTS"]["general_oracle_pointnet_ggbest_frozen"][:2] == (
        "ce_general_oracle_pointcloud_pointnet_ggbest_frozen_dgn_5k",
        "ce_general_oracle_pointcloud_pointnet_ggbest_frozen_gg_15k",
    )
