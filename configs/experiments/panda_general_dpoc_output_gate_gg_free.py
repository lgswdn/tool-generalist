"""GG output gate with trainable expert fusion/MLPs and frozen encoders."""

from configs.experiments.panda_general_dpoc_output_gate_gg import (
    CURR_EXPERT_CHECKPOINT,
    DPOC_ENCODER_CHECKPOINT,
    GRASPGEN_FAILURE_OBJECTS_MANIFEST,
    STABLE_EXPERT_CHECKPOINT,
)
from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_dpoc_output_gate_gg_free")
EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = DPOC_ENCODER_CHECKPOINT
EXP_CFG.general.rl_objects_manifest = GRASPGEN_FAILURE_OBJECTS_MANIFEST
EXP_CFG.rl.actor_critic_class = "ActorCriticTGOutputGate"
EXP_CFG.rl.output_gate_expert_a_checkpoint = STABLE_EXPERT_CHECKPOINT
EXP_CFG.rl.output_gate_expert_b_checkpoint = CURR_EXPERT_CHECKPOINT
EXP_CFG.rl.output_gate_freeze_experts = False
EXP_CFG.rl.freeze_encoder = True
EXP_CFG.rl.output_gate_hidden_dims = (256, 64)
EXP_CFG.rl.output_gate_initial_expert_a_weight = 0.8
EXP_CFG.rl.output_gate_per_action = False
EXP_CFG.rl.init_checkpoint = None
EXP_CFG.rl.resume_checkpoint = None
EXP_CFG.rl.domain_randomization.object.scale.enabled = False
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0
