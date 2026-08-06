# Minimal ablation plan for the oracle PointNet pipeline

## Question

Why does the fitted oracle PointNet perform much better on GG than the
rank-10 bottleneck policy from which its targets were extracted, and which
steps are actually necessary?

This study holds all unrelated variables fixed:

- newest 200 generated parallel grippers;
- canonical 128-bin gripper clouds;
- 0.05 m/s parallel-finger velocity limit;
- the migrated depth-12, joint-self TCE checkpoint;
- full-DGN objects, object scaling, observations, actions, rewards, PPO
  settings, total environments, and seed;
- exactly 8 GPUs and 8192 environments per run.

It does not regenerate contacts or pretrain an encoder.

## Results that already exist and must not be rerun

1. A randomly initialized PointNet does not learn adequately.
2. The depth-12 rank-10 bottleneck DGN-5k policy is weak.
3. The PointNet fitted to that bottleneck and then trained end-to-end on
   DGN-5k is substantially better.
4. The corresponding fitted-PointNet GG-15k transfer is the reference GG
   result.

## Round 1: three independent 8-GPU runs

### Node 1: full TCE, no bottleneck

Experiment:
`ce_prl_oracle_ablation_d12_full_tce_dgn_5k`

Use the frozen 128D depth-12 TCE tokens directly. This isolates whether the
rank-10 bottleneck itself discarded useful control information.

### Node 2: equal-budget bottleneck continuation

Experiment:
`ce_prl_oracle_ablation_d12_bottleneck_resume_to_dgn_10k`

Resume the completed rank-10 bottleneck DGN-5k run, including optimizer state,
for 5,000 additional iterations. The resulting policy has 10,000 total DGN
iterations, equal to the 5k bottleneck plus 5k PointNet DGN budget consumed by
the current pipeline.

This isolates whether the PointNet gain is merely extra RL optimization.

### Node 3: fitted PointNet frozen

Experiment:
`ce_prl_oracle_ablation_d12_fitted_pointnet_frozen_dgn_5k`

Load the same offline-fitted PointNet used by the successful pipeline, freeze
it, and train only the policy on DGN for 5,000 iterations. Compare it with the
already-completed trainable fitted-PointNet DGN-5k run.

This isolates whether PPO adaptation of the PointNet is necessary after
offline fitting.

## Round-1 commands

Run one command on each 8-GPU node:

```bash
./scripts/run_oracle_pointnet_necessity_round1.bash full_tce
./scripts/run_oracle_pointnet_necessity_round1.bash bottleneck_budget
./scripts/run_oracle_pointnet_necessity_round1.bash fitted_frozen
```

Each command enforces 8 GPUs and 8192 total environments.

## Decision after round 1

| Observation | Conclusion |
|---|---|
| Full TCE succeeds; bottleneck-10k stays weak | Rank-10 compression is harmful; do not keep it as the final controller. |
| Bottleneck-10k catches up | The apparent PointNet gain is largely an RL-budget effect; test the bottleneck directly on GG. |
| Frozen fitted PointNet matches the trainable reference | End-to-end PointNet adaptation is unnecessary; offline fitting plus policy learning is enough. |
| Frozen fitted PointNet is weak; trainable reference is good | End-to-end PointNet adaptation on DGN is essential. |
| Full TCE and fitted PointNet both succeed | Compare only their best DGN checkpoints on GG and prefer the simpler representation. |

Use `model_last.pt` for budget accounting and learning-curve comparisons.
Also report `model_best.pt`, but do not choose a conclusion only from a lucky
best-checkpoint spike.

## Conditional round 2

Run at most one new GG-15k job for the best viable round-1 candidate. If round
1 still cannot determine whether the RL-shaped bottleneck targets matter,
perform one offline static-PCA PointNet fit and one DGN-5k run. Do not launch
that run before reading round 1.

The clean final methodology should be the shortest branch that survives:

1. frozen pretrained geometry representation;
2. optional offline geometric PointNet fitting;
3. only the empirically necessary DGN adaptation;
4. GG transfer.

