# Bimanual Unstable Tool Environment Plan

Goal: add a new Isaac Lab bimanual task using the unstable/random object pose goal, without changing the behavior or config hashes of existing single-arm tasks.

## Scope

- Add a new bimanual unstable environment/task.
- Spawn two Franka-tool arms per env, each with independently assigned random tool USDs. Same-tool pairs are allowed.
- Keep the unstable target-pose command, object velocity observation, still-at-goal reward, and dropped-object termination semantics.
- Feed three point clouds to the policy encoder: tool 1, tool 2, object.
- Keep each cloud in env-frame observations, then center each one by its own bbox center inside the model before ViT encoding.
- Additive only: new files/configs/classes where practical, preserving old default config hashes.

## Environment

- New env module: `env_tool_bimanual_unstable.py`.
- New scene assets:
  - `robot_1`
  - `robot_2`
  - `ee_frame_1`
  - `ee_frame_2`
  - existing object/table/light command setup reused from current tool env patterns.
- New action group: two relative joint position action terms, each 7D, total 14D.
- New observations:
  - `object_cloud`
  - `tool1_cloud`
  - `tool2_cloud`
  - `object_bbox_center`
  - `tool1_bbox_center`
  - `tool2_bbox_center`
  - `hand1_state`
  - `hand2_state`
  - `robot1_state`
  - `robot2_state`
  - `previous_action`
  - `relative_goal_pose`
  - `object_velocity`
  - `physics`
- New rewards wrap the unstable single-arm reward logic for nearest/both EEs:
  - contact reward uses nearest EE/tool head distance.
  - object-goal tracking gates on nearest EE distance.
  - stillness and success remain object-based.
  - energy penalty sums both robots.

## Model

- New actor class: `ActorCriticTGBimanual`.
- New observation layout helper: bimanual-specific slices, leaving `ActorCriticTG` layout untouched.
- New 3-stream TCE encoder wrapper:
  - patchifies tool 1, tool 2, and object separately.
  - concatenates `3P` tokens before the existing ViT blocks.
  - loads existing 2-stream TCE checkpoints by copying tool type embedding into both tool slots and object embedding into the object slot.
- Context includes:
  - `tool1_bbox_center - object_bbox_center`
  - `tool2_bbox_center - object_bbox_center`
  - `tool2_bbox_center - tool1_bbox_center`
  - `object_bbox_center`
  - bimanual hand/robot state
  - previous action
  - relative goal
  - object velocity
  - physics

## Config/Runtime

- Add the new actor class to runtime validation and RSL-RL module registry.
- Add bimanual observation layout support without changing defaults.
- Add a new experiment config that opts into the new task and actor.
- Keep existing experiment configs and default dataclass field values untouched where hashes depend on them.

## Verification

- Static tests for:
  - bimanual observation/action dimensions.
  - new actor class is registered.
  - existing config hashes remain unchanged.
  - checkpoint type-embedding expansion path.
- Isaac smoke test if the current environment has Isaac Lab import/runtime available.
