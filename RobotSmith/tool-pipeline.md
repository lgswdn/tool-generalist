# 生成

```bash
conda activate pamo
cd ./RobotSmith
python deterministic_pipeline.py --task_name eef --response_file soil_scoop.txt
```

# 后处理

```
conda activate pamo
./post_process/pamodet.sh ./eef/ tmp_trial ../../pamo/
```

```
conda deactivate
conda activate isaac
bash ./post_process/convert.sh ./eef/ ../../DexGraspNet/
python ./post_process/convert_urdf.py --eef-dir ./eef/ --headless
python ./post_process/batch_generate_franka_single_launch.py --tools-root ./eef/objects_usd/ --src-root ../FrankaEmika/ --output-root ../Robots --overwrite --reuse-output-root --mirror-tool-assets --disable-gravity --headless
cd ..
```

