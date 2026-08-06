#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/home/zhengyixin/tool-generalist"
cd "${REPO_ROOT}"

./run.bash panda_general_oracle_pointcloud_pointnet_rich_scratch_resume_to_5k
./run.bash panda_general_oracle_pointcloud_pointnet_rich_scratch_gg_from_resumed_full_yes_5k
