"""Direct-grasp evaluation config for the official Panda gripper on unstable poses."""

from copy import deepcopy

from configs.experiments.panda_gripper_diff_post import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)

# Keep evaluation artifacts and any diagnostics separate from the training run.
EXP_CFG.name = "single_unstable_diff_post_grasp_eval"
EXP_CFG.general.name = "single_unstable_diff_post_grasp_eval"
EXP_CFG.rl.name = "single_unstable_diff_post_grasp_eval"
EXP_CFG.rl.launch.run_name = "single_unstable_diff_post_grasp_eval"

# This evaluation targets unstable object poses only.  Setting both endpoints is
# intentional: consumers may read the endpoint probabilities even when the
# curriculum itself is disabled.
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 0.0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0.0

# Explicitly apply fixed object friction at reset.
EXP_CFG.rl.domain_randomization.object.material.enabled = True
EXP_CFG.rl.domain_randomization.object.material.static_friction_range = (1.5, 1.5)
EXP_CFG.rl.domain_randomization.object.material.dynamic_friction_range = (1.2, 1.2)

# Replay candidates already carry the exact object scale used to generate each
# grasp.  The replay manifest applies that scale before PhysX starts, so no
# additional prestartup scale randomization may run for this evaluation.
EXP_CFG.rl.domain_randomization.object.scale.enabled = False
