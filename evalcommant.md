torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/eval_single_tool.py \
    --distributed \
    --runtime_spec /mnt/project/world_model/tool_generalist/artifacts/RL/multitool_sdf/contact_gen_multitool_new/TCE/rl_default/20260515T013828Z/rl_runtime_spec.json \
    --paths_yaml /mnt/home/zhengyixin/tool-generalist/paths_panda_hand.yaml \
    --checkpoint /mnt/project/world_model/tool_generalist/artifacts/RL/multitool_sdf/contact_gen_multitool_new/TCE/rl_default/20260515T013828Z/model_8500.pt \
    --task tool-sdf-v0 \
    --tool 000_panda_hand_end_effector_var_000 \
    --num_envs 256 \
    --num_episodes 10 \
    --success_videos 32 \
    --failure_videos 32 \
    --video_width 512 \
    --video_height 512 \
    --video_dir /mnt/project/world_model/tool_generalist/artifacts/RL/multitool_sdf/contact_gen_multitool_new/TCE/rl_default/20260515T013828Z/videos/eval_panda \
    --headless

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/eval_single_tool.py \
    --distributed \
    --runtime_spec /mnt/project/world_model/tool_generalist/artifacts/RL/multitool_sdf/contact_gen_multitool_new/TCE/rl_default/20260515T013828Z/rl_runtime_spec.json \
    --paths_yaml /mnt/home/zhengyixin/tool-generalist/paths_panda_hand.yaml \
    --checkpoint /mnt/project/world_model/tool_generalist/artifacts/RL/multitool_sdf/contact_gen_multitool_new/TCE/rl_default/20260515T013828Z/model_8500.pt \
    --task tool-sdf-v0 \
    --tool 000_panda_hand_end_effector_var_000 \
    --num_envs 256 \
    --num_episodes 10 \
    --success_videos 32 \
    --failure_videos 32 \
    --video_width 2048 \
    --video_height 2048 \
    --video_fps 10 \
    --video_dir /mnt/project/world_model/tool_generalist/artifacts/RL/multitool_sdf/contact_gen_multitool_new/TCE/rl_default/20260515T013828Z/videos/eval_panda \
    --headless

c

CUDA_VISIBLE_DEVICE=0 python scripts/eval_tools.py     --runtime_spec /mnt/home/zhengyixin/tool-generalist/artifacts/RL/point2vec/no-contact/TCE/rl_default/20260511T090600Z/rl_runtime_spec.json     --checkpoint /mnt/home/zhengyixin/tool-generalist/artifacts/RL/point2vec/no-contact/TCE/rl_default/20260511T090600Z/model_0.pt     --num_envs 16     --num_episodes 1     --video     --video_length 300     --video_dir videos/point2vec_eval     --headless --video_width 1080 --video_height 1080

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/eval_single_tool.py \
    --distributed \
    --runtime_spec /mnt/project/world_model/tool_generalist/artifacts/RL/panda_hand_sdf/contact_gen_multitool_new/TCE/rl_default/20260516T142756Z/rl_runtime_spec.json \
    --checkpoint /mnt/project/world_model/tool_generalist/artifacts/RL/panda_hand_sdf/contact_gen_multitool_new/TCE/rl_default/20260516T142756Z/model_6000.pt \
    --task tool-sdf-v0 \
    --tool 000_panda_hand_end_effector_var_000 \
    --num_envs 256 \
    --num_episodes 10 \
    --success_videos 32 \
    --failure_videos 32 \
    --video_width 2048 \
    --video_height 2048 \
    --video_fps 10 \
    --video_dir /mnt/project/world_model/tool_generalist/artifacts/RL/panda_hand_sdf/contact_gen_multitool_new/TCE/rl_default/20260516T142756Z/videos/eval_panda \
    --headless

torchrun --standalone --nnodes=1 --nproc_per_node=2     scripts/eval_single_tool.py     --distributed     --runtime_spec /mnt/project/world_model/tool_generalist/artifacts/RL/panda_hand_sdf/contact_gen_multitool_new/TCE/rl_default/20260516T142756Z/rl_runtime_spec.json     --checkpoint /mnt/project/world_model/tool_generalist/artifacts/RL/panda_hand_sdf/contact_gen_multitool_new/TCE/rl_default/20260516T142756Z/model_6000.pt     --task tool-sdf-v0     --tool 000_panda_hand_end_effector_var_000     --num_envs 256     --num_episodes 10     --success_videos 16     --failure_videos 16     --video_width 512     --video_height 512     --video_fps 10     --video_dir /mnt/project/world_model/tool_generalist/artifacts/RL/panda_hand_sdf/contact_gen_multitool_new/TCE/rl_default/20260516T142756Z/videos/eval_panda     --headless

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/record_failure_videos.py \
    --distributed \
    --runtime_spec /mnt/project/world_model/tool_generalist/artifacts/RL/panda_hand_diff_post/contact_gen_multitool_new/TCE/panda_hand_diff_post/20260519T053045Z/rl_runtime_spec.json \
    --checkpoint /mnt/project/world_model/tool_generalist/artifacts/RL/panda_hand_diff_post/contact_gen_multitool_new/TCE/panda_hand_diff_post/20260519T053045Z/model_10000.pt \
    --num_envs 256 \
    --num_failure_videos 16 \
    --object_rerandomize_interval_steps 500 \
    --video_width 512 \
    --video_height 512 \
    --video_fps 10 \
    --headless

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/eval_tools.py \
    --distributed \
    --headless \
    --runtime_spec /mnt/project/world_model/tool_generalist/artifacts/RL/multitools_diff/contact_gen_multitool_new/TCE/multitools_diff_1/20260520T061818Z/rl_runtime_spec.json \
    --checkpoint /mnt/project/world_model/tool_generalist/artifacts/RL/multitools_diff/contact_gen_multitool_new/TCE/multitools_diff_1/20260520T061818Z/model_9000.pt \
    --num_envs 1024 \
    --num_episodes 100 \
    --randomize_objects \
    --object_random_seed 42

torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/eval_tools_steps.py \
    --distributed \
    --runtime_spec /mnt/project/world_model/tool_generalist/artifacts/RL/multitools_full_tool_diff_post/contact_gen_full_tool/TCE/multitools_full_tool_diff_post/20260521T165009Z/rl_runtime_spec.json \
    --paths_yaml /mnt/home/zhengyixin/tool-generalist/paths_test.yaml \
    --checkpoint /mnt/project/world_model/tool_generalist/artifacts/RL/multitools_full_tool_diff_post/contact_gen_full_tool/TCE/multitools_full_tool_diff_post/20260521T165009Z/model_10000.pt \
    --num_envs 1024 \
    --num_steps 17000 \
    --randomize_objects \
    --object_random_seed 42 \
    --headless

torchrun --nproc_per_node=2 scripts/record_multi_videos.py \
    --distributed \
    --runtime_spec /mnt/project/world_model/tool_generalist/artifacts/RL/multitools_full_tool_diff_post/contact_gen_full_tool/TCE/multitools_full_tool_diff_post/20260521T165009Z/rl_runtime_spec.json \
    --checkpoint  /mnt/project/world_model/tool_generalist/artifacts/RL/multitools_full_tool_diff_post/contact_gen_full_tool/TCE/multitools_full_tool_diff_post/20260521T165009Z/model_10000.pt \
    --num_envs 512 \
    --num_steps 1000 \
    --headless

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/eval_tools_steps.py \
    --distributed \
    --headless \
    --runtime_spec /mnt/project/world_model/tool_generalist/artifacts/RL/multitools_full_tool_diff_post/contact_gen_full_tool/TCE/multitools_full_tool_diff_post/20260521T165009Z/rl_runtime_spec.json \
    --paths_yaml /mnt/home/zhengyixin/tool-generalist/paths_test_tool_train_obj.yaml \
    --checkpoint /mnt/project/world_model/tool_generalist/artifacts/RL/multitools_full_tool_diff_post/contact_gen_full_tool/TCE/multitools_full_tool_diff_post/20260521T165009Z/model_10000.pt \
    --num_envs 1694 \
    --num_steps 20000 \
    --randomize_objects \
    --object_random_seed 42 \
    --output_dir /mnt/project/world_model/tool_generalist/eval_results/test_tool_train_obj

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/eval_tools_steps.py \
    --distributed \
    --headless \
    --runtime_spec /mnt/project/world_model/tool_generalist/artifacts/RL/multitools_full_tool_diff_post/contact_gen_full_tool/TCE/multitools_full_tool_diff_post/20260521T165009Z/rl_runtime_spec.json \
    --paths_yaml /mnt/home/zhengyixin/tool-generalist/paths_train_tool_test_obj.yaml \
    --checkpoint /mnt/project/world_model/tool_generalist/artifacts/RL/multitools_full_tool_diff_post/contact_gen_full_tool/TCE/multitools_full_tool_diff_post/20260521T165009Z/model_10000.pt \
    --num_envs 1694 \
    --num_steps 20000 \
    --randomize_objects \
    --object_random_seed 42 \
    --output_dir /mnt/project/world_model/tool_generalist/eval_results/train_tool_test_obj