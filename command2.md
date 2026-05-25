(isaac) root@pt-9efdca7e3bef4ece8a5e5a79761f8a46-worker-0:~/tool-generalist# CUDA_VISIBLE_DEVICES=0 python run_experiment.py --config configs/experiments/panda_hand_sdf.py 
[2026-05-16T13:28:18+00:00] [RUN] experiment=panda_hand_sdf config=configs/experiments/panda_hand_sdf.py mode=run
[2026-05-16T13:28:18+00:00] [RUN] paths_yaml=/mnt/home/zhengyixin/tool-generalist/paths_panda_hand.yaml
[2026-05-16T13:28:18+00:00] [SKIP] stage=contact_gen action=skipped artifact=/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/contact_gen_multitool_new/ded4300acdcb31c55ee93f2e86d0f96a0ead8fc4edaae22f749eb9ecbe362e61
[2026-05-16T13:28:18+00:00] [SKIP] stage=pretrain action=skipped artifact=/mnt/project/world_model/tool_generalist/artifacts/encoder/panda_hand_sdf/contact_gen_multitool_new/sdf_only_multitool_sdf/c81d1ed235640de2101c17175f88f97ae092751998877499d94e4638db9fb3aa
[2026-05-16T13:28:18+00:00] [RUN] stage=rl action=run artifact=/mnt/project/world_model/tool_generalist/artifacts/RL/panda_hand_sdf/contact_gen_multitool_new/TCE/rl_default/20260516T132818Z
[2026-05-16T13:28:18+00:00] [START] stage=rl entrypoint=utils.experiment.rl_stage:run_rl_stage
[rl_launcher] before importing isaaclab.app.AppLauncher
[WARN][AppLauncher]: There are no arguments attached to the ArgumentParser object. If you have your own arguments, please load your own arguments before calling the `AppLauncher.add_app_launcher_args` method. This allows the method to check the validity of the arguments and perform checks for argument names.
[rl_launcher] before AppLauncher task=tool-sdf-v0 args={'headless': True, 'livestream': -1, 'enable_cameras': False, 'xr': False, 'device': 'cuda:0', 'cpu': False, 'verbose': False, 'info': False, 'experience': '', 'rendering_mode': None, 'kit_args': '', 'anim_recording_enabled': False, 'anim_recording_start_time': 0, 'anim_recording_stop_time': 10, 'distributed': False}
[INFO][AppLauncher]: Using device: cuda:0
[INFO][AppLauncher]: Loading experience file: /mnt/home/zhengyixin/IsaacLab/apps/isaaclab.python.headless.kit
[Warning] [simulation_app.simulation_app] Modules: ['omni.kit_app'] were loaded before SimulationApp was started and might not be loaded correctly.
[Warning] [simulation_app.simulation_app] Please check to make sure no extra omniverse or pxr modules are imported before the call to SimulationApp(...)
Loading user config located at: '/mnt/home/zhengyixin/miniconda3/envs/isaac/lib/python3.11/site-packages/omni/data/Kit/Isaac-Sim/5.0/user.config.json'
[Info] [carb] Logging to file: /mnt/home/zhengyixin/miniconda3/envs/isaac/lib/python3.11/site-packages/omni/logs/Kit/Isaac-Sim/5.0/kit_20260516_132820.log
2026-05-16T13:28:20Z [12ms] [Warning] [carb.crashreporter-breakpad.plugin] [previous crash] preventing upload of minidump due to user opt-out: '/mnt/home/zhengyixin/miniconda3/envs/isaac/lib/python3.11/site-packages/omni/data/Kit/Isaac-Sim/5.0/25e17238-0b8f-424a-7b6699b7-e2608baa.dmp'
2026-05-16T13:28:20Z [14ms] [Warning] [carb.crashreporter-breakpad.plugin] [previous crash] preventing upload of minidump due to user opt-out: '/mnt/home/zhengyixin/miniconda3/envs/isaac/lib/python3.11/site-packages/omni/data/Kit/Isaac-Sim/5.0/6d71afb7-eeab-4424-bf0988b5-930bf634.dmp'
2026-05-16T13:28:20Z [16ms] [Warning] [carb.crashreporter-breakpad.plugin] [previous crash] preventing upload of minidump due to user opt-out: '/mnt/home/zhengyixin/miniconda3/envs/isaac/lib/python3.11/site-packages/omni/data/Kit/Isaac-Sim/5.0/06c1debb-1239-4377-43d68781-0630adcc.dmp'
2026-05-16T13:28:20Z [18ms] [Warning] [carb.crashreporter-breakpad.plugin] [previous crash] preventing upload of minidump due to user opt-out: '/mnt/home/zhengyixin/miniconda3/envs/isaac/lib/python3.11/site-packages/omni/data/Kit/Isaac-Sim/5.0/b3d983be-7836-41cb-fc9cc4a7-6b20099f.dmp'
2026-05-16T13:28:20Z [19ms] [Warning] [carb.crashreporter-breakpad.plugin] Failed to parse toml file '/mnt/home/zhengyixin/miniconda3/envs/isaac/lib/python3.11/site-packages/omni/data/Kit/Isaac-Sim/5.0/b6694fdf-62de-48e7-6b09d1a1-0c937cfc.dmp.toml': /mnt/home/zhengyixin/miniconda3/envs/isaac/lib/python3.11/site-packages/omni/data/Kit/Isaac-Sim/5.0/b6694fdf-62de-48e7-6b09d1a1-0c937cfc.dmp.toml could not be opened for parsing
2026-05-16T13:28:20Z [19ms] [Warning] [carb.crashreporter-breakpad.plugin] [previous crash] preventing upload of minidump due to user opt-out: '/mnt/home/zhengyixin/miniconda3/envs/isaac/lib/python3.11/site-packages/omni/data/Kit/Isaac-Sim/5.0/b6694fdf-62de-48e7-6b09d1a1-0c937cfc.dmp'
2026-05-16T13:28:26Z [6,480ms] [Warning] [omni.usd_config.extension] Enable omni.materialx.libs extension to use MaterialX
2026-05-16T13:28:31Z [10,885ms] [Warning] [omni.platforminfo.plugin] failed to open the default display.  Can't verify X Server version.
2026-05-16T13:28:32Z [12,245ms] [Warning] [omni.isaac.dynamic_control] omni.isaac.dynamic_control is deprecated as of Isaac Sim 4.5. No action is needed from end-users.
2026-05-16T13:29:10Z [49,809ms] [Warning] [carb.cudainterop.plugin] CUDA_VISIBLE_DEVICES environment variable is set.
2026-05-16T13:29:10Z [49,809ms] [Warning] [carb.cudainterop.plugin] Note CUDA device enumeration and Omniverse device enumeration are different.
2026-05-16T13:29:10Z [49,812ms] [Warning] [carb.cudainterop.plugin] Setting CUDA_VISIBLE_DEVICES can lead to undesired behavior or crashes.
2026-05-16T13:29:12Z [52,483ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,487ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
2026-05-16T13:29:12Z [52,487ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,487ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
2026-05-16T13:29:12Z [52,487ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,487ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
2026-05-16T13:29:12Z [52,488ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,488ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
2026-05-16T13:29:12Z [52,488ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,488ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
2026-05-16T13:29:12Z [52,488ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,489ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
2026-05-16T13:29:12Z [52,489ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,489ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
2026-05-16T13:29:12Z [52,490ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,490ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
2026-05-16T13:29:12Z [52,490ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,490ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
2026-05-16T13:29:12Z [52,490ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,491ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
2026-05-16T13:29:12Z [52,491ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,491ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
2026-05-16T13:29:12Z [52,491ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,491ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
2026-05-16T13:29:12Z [52,491ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,491ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
2026-05-16T13:29:12Z [52,491ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA L40S
2026-05-16T13:29:12Z [52,491ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.

|---------------------------------------------------------------------------------------------|
| Driver Version: 535.86.10     | Graphics API: Vulkan
|=============================================================================================|
| GPU | Name                             | Active | LDA | GPU Memory | Vendor-ID | LUID       |
|     |                                  |        |     |            | Device-ID | UUID       |
|     |                                  |        |     |            | Bus-ID    |            |
|---------------------------------------------------------------------------------------------|
| 0   | NVIDIA L40S                      | Yes: 0 |     | 46068   MB | 10de      | 0          |
|     |                                  |        |     |            | 26b9      | e73499fa.. |
|     |                                  |        |     |            | 2e        |            |
|---------------------------------------------------------------------------------------------|
| 1   | NVIDIA L40S                      |        |     | 46068   MB | 10de      | 0          |
|     |                                  |        |     |            | 26b9      | f9d24680.. |
|     |                                  |        |     |            | 30        |            |
|---------------------------------------------------------------------------------------------|
| 2   | NVIDIA L40S                      |        |     | 46068   MB | 10de      | 0          |
|     |                                  |        |     |            | 26b9      | c89b3c88.. |
|     |                                  |        |     |            | 40        |            |
|---------------------------------------------------------------------------------------------|
| 3   | NVIDIA L40S                      |        |     | 46068   MB | 10de      | 0          |
|     |                                  |        |     |            | 26b9      | 132e42dc.. |
|     |                                  |        |     |            | 41        |            |
|---------------------------------------------------------------------------------------------|
| 4   | NVIDIA L40S                      |        |     | 46068   MB | 10de      | 0          |
|     |                                  |        |     |            | 26b9      | 6af128f0.. |
|     |                                  |        |     |            | b0        |            |
|---------------------------------------------------------------------------------------------|
| 5   | NVIDIA L40S                      |        |     | 46068   MB | 10de      | 0          |
|     |                                  |        |     |            | 26b9      | e9d0f3ff.. |
|     |                                  |        |     |            | b1        |            |
|---------------------------------------------------------------------------------------------|
| 6   | NVIDIA L40S                      |        |     | 46068   MB | 10de      | 0          |
|     |                                  |        |     |            | 26b9      | 84774cb2.. |
|     |                                  |        |     |            | c2        |            |
|---------------------------------------------------------------------------------------------|
| 7   | NVIDIA L40S                      |        |     | 46068   MB | 10de      | 0          |
|     |                                  |        |     |            | 26b9      | 5aa69bc1.. |
|     |                                  |        |     |            | c3        |            |
|=============================================================================================|
| OS: 24.04.2 LTS (Noble Numbat) ubuntu, Version: 24.04.2, Kernel: 5.14.0-284.25.1.el9_2.x86_64
| Processor: Intel(R) Xeon(R) Gold 6448Y
| Bare Metal Cores: 64 | Bare Metal Logical Cores: 128
| Core Usage Quota: 112
|---------------------------------------------------------------------------------------------|
| Total Memory (MB): 491520 | Free Memory: 482394
| Total Page/Swap (MB): 0 | Free Page/Swap: 0
|---------------------------------------------------------------------------------------------|
2026-05-16T13:29:13Z [52,782ms] [Warning] [gpu.foundation.plugin] ECC is enabled on physical device 0
[rl_launcher] after AppLauncher creation
[rl_launcher] before RSL-RL training flow
[rl_launcher] before importing gymnasium/torch/IsaacLab/RSL-RL modules
/mnt/home/zhengyixin/tool-generalist/rsl_rl/modules/models/common.py:294: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  @custom_fwd
/mnt/home/zhengyixin/tool-generalist/rsl_rl/modules/models/common.py:304: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  @custom_bwd
[rl_launcher] after importing gymnasium/torch/IsaacLab/RSL-RL modules
[rl_launcher] before importing isaaclab_tasks
[rl_launcher] after importing isaaclab_tasks
[rl_launcher] before importing IsaacLab_nonPrehensile.tasks
[rl_launcher] after importing IsaacLab_nonPrehensile.tasks
[rl_launcher] before importing tool-sdf task registration module
[rl_launcher] after importing tool-sdf task registration module
[rl_launcher] gym task registered task=tool-sdf-v0
[rl_launcher] before hydra task config task=tool-sdf-v0 num_envs=64 max_iterations=1000000 artifact=/mnt/project/world_model/tool_generalist/artifacts/RL/panda_hand_sdf/contact_gen_multitool_new/TCE/rl_default/20260516T132818Z
[rl_launcher] calling hydra main
[INFO] Loading path config from /mnt/home/zhengyixin/tool-generalist/paths_panda_hand.yaml
[INFO] Loaded 1 tool variants from /mnt/project/world_model/tool_generalist/eef_panda/Robots
[INFO] Tool assignment rank=0/1 local_rank=0 envs=64 randomize=False
[WARNING] Asset sem-ToiletPaper-260949513aaba2eb90cadbd65232985b already exists, skipping...
[WARNING] Asset sem-Vase-d3978e26d3e0d773b3ffb0c309689ebd already exists, skipping...
[WARNING] Asset sem-Clock-758885e4c4e7bdd8e08074c1d83054ad already exists, skipping...
[WARNING] Asset core-jar-d56098d4d83f5976a2c59a4d90e63212 already exists, skipping...
[WARNING] Asset sem-Tank-52eb26e5a71f2c8fc36c7821ad0d5a86 already exists, skipping...
[WARNING] Asset core-camera-2153bc743019671ae60635d9e388f801 already exists, skipping...
[WARNING] Asset sem-TableClock-cebb0e2be3b2cc1b474465268b958bdc already exists, skipping...
[WARNING] Asset core-bowl-d1addad5931dd337713f2e93cbeac35d already exists, skipping...
[WARNING] Asset ddg-kit_LivioClassicOil already exists, skipping...
[WARNING] Asset sem-Fruit-473758ca6cb0506ee7697d561711bd2b already exists, skipping...
[WARNING] Asset core-mug-bf2b5e941b43d030138af902bc222a59 already exists, skipping...
[INFO] Object assignment rank=0/1 local_rank=0 envs=64 randomize=False
[INFO]: Parsing configuration from: IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool:NonPrehensileEnvCfg
[INFO]: Parsing configuration from: IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.agents.config.rsl_rl_ppo_cfg:TGPPORunnerCfg
[rl_launcher] entered hydra main
[rl_launcher] before gym.make task=tool-sdf-v0 env_device=cuda:0 agent_device=cuda:0 num_envs=64
Setting seed: 0
[INFO]: Base environment:
        Environment device    : cuda:0
        Environment seed      : 0
        Physics step-size     : 0.0125
        Rendering step-size   : 0.1
        Environment step-size : 0.1
2026-05-16T13:30:06Z [106,058ms] [Warning] [omni.fabric.plugin] removePath called on non-existent path /World/Template/Asset_0000/material