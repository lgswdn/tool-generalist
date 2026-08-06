# One-policy-DoF grippers

This path is for grippers whose policy interface is one normalized closure
command, even when the physical mechanism contains several coupled joints.
Dexterous hands are intentionally out of scope.

## Category contract

The manifest accepts at most one official model in each category:

- `robotiq_like`
- `rg_like`
- `three_finger`

Generated populations may contain many assets in either topology-stable
training category:

- `two_finger_revolute`
- `three_finger_high_dof`

The integrated official representatives are:

| Category | Model | One-command adapter |
| --- | --- | --- |
| `robotiq_like` | Robotiq 2F-140 | primary joint + hard mimics |
| `rg_like` | OnRobot RG2 | primary joint + hard mimics |
| `three_finger` | Robotiq 3-Finger Adaptive | fixed basic spread + nine-joint nominal closure synergy |

The policy convention is always `-1 = open`, `+1 = closed`. For the 3-Finger,
"one DoF" means one policy command: it does not expose independent fingers or
the scissor mode. Its fixed-spread synergy is a deterministic nominal closure,
not a claim that the simulator reproduces the hardware's underactuated contact
adaptation.

Isaac/PhysX articulation views require uniform topology. Different topology
families therefore run as homogeneous simulator processes; a shared policy can
train across those processes, but unlike grippers from the same topology family
they cannot be randomly mixed inside one articulation view.

## Generated GraspGenX-style families

Generate the two remaining non-dexterous families with:

```bash
python scripts/generate_graspgenx_grippers.py \
  --output-root gripper/generated_graspgenx \
  --num-revolute 100 \
  --num-three-finger 100 \
  --seed 0
```

The revolute family has five mechanism links and four revolute joints. Every
asset samples an outward open angle and therefore also a distinct inward closed
angle over a common 0.65-radian joint sweep. It implements both GraspGenX
closing modes: a `1:-1` mode whose top links remain globally parallel, and a
`1:1` pinch mode whose top links curl with the mid links. Both retain the same
homogeneous four-element closure target. Optional outer bars and optional
square/round fingertip blocks change geometry without changing topology; the
mid and top finger links themselves are always present. Its geometry ranges
match the fierce 200-asset Panda-general population (51--235 mm total finger
length, roughly 6--58 mm link thickness, and 6--35 mm tip width), including
slender chopstick-like samples. Hinge spacing is solved from the exact closed
surface envelopes so the opposing fingers have zero nominal geometric gap.

The three-finger family uses one fixed nine-joint topology (three joints per
finger). Finger dimensions, base dimensions, wrist ratio, and top-finger side
rotation vary, while all nine joints follow the same linearly interpolated
one-command closure trajectory. The six-joint alternative is intentionally not
mixed into this manifest because it would require another PhysX articulation
topology. Whole-finger lengths and independent link widths/depths use the same
aggressive Panda-general envelope (51--235 mm length and roughly 6--58 mm
cross-sections). Each finger's mounting radius is solved from its complete
three-joint closed kinematics, so all three centerlines terminate at the common
palm axis with zero nominal gap.

Both families use a nominal 0.8-second full closure. They copy the generated
parallel-gripper Franka arm/base exactly and record every sampled parameter in
both the manifest and each asset's `params.json`.

Convert them on the Isaac machine:

```bash
python scripts/convert_one_dof_gripper.py \
  --manifest gripper/generated_graspgenx/two_finger_revolute.json \
  --headless

python scripts/convert_one_dof_gripper.py \
  --manifest gripper/generated_graspgenx/three_finger_high_dof.json \
  --headless
```

## Asset workflow

1. Rebuild the two derived official URDFs after changing the pinned source or
   mount logic:

   ```bash
   python scripts/build_official_one_dof_urdfs.py
   ```

2. Add one manifest entry under `configs/grippers/` with explicit URDF, USD,
   control, grasp-frame, and per-link cloud geometry metadata.
3. Keep `command_dim` equal to one. Use `primary_joint_with_mimics` for a single
   driven URDF joint or `joint_synergy` for an explicit multi-joint closure map.
4. Ensure every movable cloud link is either the primary joint's child or is
   connected by a valid mimic joint. Open and closed targets must lie inside the
   URDF limits.
5. Convert on an Isaac Sim-compatible machine. Choose one manifest per process:

   ```bash
   python scripts/convert_one_dof_gripper.py \
     --manifest configs/grippers/robotiq_2f140.json \
     --headless
   ```

   The other manifests are `configs/grippers/onrobot_rg2.json` and
   `configs/grippers/robotiq_3f.json`.

   The converter validates that every URDF `<mimic>` follower receives a
   `PhysxMimicJointAPI` and overrides Isaac's compliant mimic defaults with a
   hard constraint. If an older USD lacks those constraints or still uses
   positive mimic compliance values, rerun the command with `--force`.

6. Select a paths YAML containing `one_dof_grippers.root` and
   `one_dof_grippers.manifest`, then use `robot_mode=one_dof_gripper`, task
   `one-dof-gripper-v0`, and cloud source `gripper_cloud_cache_v1`.

The runtime observation remains 18D: seven arm positions, seven arm velocities,
and four semantic gripper values (closure, closure velocity, tracking error,
and normalized effort). The tool cloud is always exactly 512 points and follows
the simulated pose of every declared gripper link.

## Visual validation

Render the complete generated revolute population without Isaac/Kit. Opening
fraction `1` is fully open and `0` is fully closed:

```bash
python scripts/render_generated_gripper_contact_sheet.py \
  --manifest gripper/generated_graspgenx/two_finger_revolute.json \
  --output videos/generated_two_finger_revolute/contact_sheets/open.png \
  --num 100 --per-page 20 --opening 1.0

python scripts/render_generated_gripper_contact_sheet.py \
  --manifest gripper/generated_graspgenx/two_finger_revolute.json \
  --output videos/generated_two_finger_revolute/contact_sheets/closed.png \
  --num 100 --per-page 20 --opening 0.0

python scripts/render_generated_gripper_contact_sheet.py \
  --manifest gripper/generated_graspgenx/three_finger_high_dof.json \
  --output videos/generated_three_finger_high_dof/contact_sheets/open.png \
  --num 100 --per-page 20 --opening 1.0

python scripts/render_generated_gripper_contact_sheet.py \
  --manifest gripper/generated_graspgenx/three_finger_high_dof.json \
  --output videos/generated_three_finger_high_dof/contact_sheets/closed.png \
  --num 100 --per-page 20 --opening 0.0
```

After conversion, run one environment with a stationary arm and slow closure
sweep:

```bash
python scripts/visualize_one_dof_gripper_random.py \
  --config configs/experiments/onrobot_rg2_diff_post.py \
  --num_envs 1 \
  --gripper_action_mode sweep
```

Use `configs/experiments/robotiq_3f_diff_post.py` for the three-finger synergy
or `configs/experiments/robotiq_2f140_diff_post.py` for the existing 2F-140.
For the generated populations, use
`configs/experiments/generated_two_finger_revolute_diff_post.py` or
`configs/experiments/generated_three_finger_high_dof_diff_post.py`.

Blue spheres show the exact 512-point cloud supplied to RL. The red sphere shows
the exact manifest-defined interaction center used by rewards and hand-state
construction. Periodic logs report commanded/measured closure, tracking error,
cloud shape and finiteness, bounding box, and gripper-link motion.
