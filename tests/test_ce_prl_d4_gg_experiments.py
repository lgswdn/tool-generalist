from __future__ import annotations

from pathlib import Path

from configs.panda_comparison_common import GG_UNGRASPABLE_MANIFEST
from utils.config.loader import load_exp_cfg


def test_every_d4_dgn_parent_has_a_strict_gg_child():
    experiment_dir = Path("configs/experiments")
    parents = sorted(
        path.stem
        for path in experiment_dir.glob("ce_prl_unicorn_d4_full*_dgn_5k.py")
    )
    children = sorted(
        path.stem
        for path in experiment_dir.glob("ce_prl_unicorn_d4_full*_gg_15k.py")
    )
    assert len(parents) == 9
    assert children == [
        f"{parent.removesuffix('_dgn_5k')}_gg_15k" for parent in parents
    ]

    for child in children:
        parent = f"{child.removesuffix('_gg_15k')}_dgn_5k"
        cfg = load_exp_cfg(experiment_dir / f"{child}.py")
        assert cfg.pretrain_reuse == f"{parent}.py"
        assert cfg.contact_gen.enabled is False
        assert cfg.contact_gen.regenerate is False
        assert cfg.pretrain.retrain is False
        assert cfg.rl.resume_checkpoint is None
        assert cfg.rl.init_checkpoint.endswith("/model_last.pt")
        assert cfg.rl.ppo.max_iterations == 15_000
        assert cfg.general.rl_objects_manifest == GG_UNGRASPABLE_MANIFEST
        assert cfg.rl.domain_randomization.object.scale.enabled is False
        assert cfg.num_gpus == 8
        assert cfg.model.tce.vit_depth == 4
