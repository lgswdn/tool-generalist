# UniCORN 预训练复刻实现计划

目标：在本仓库现有实验框架内完整复刻 `docs/hamnet-unicorn-paper` 中的 UniCORN 预训练编码器，包括训练任务、编码器/解码器架构、数据构造、超参数、checkpoint/manifest 和现有 `run_experiment.py --config ...` 启动方式。本文只制定实现计划，不修改训练代码。

## 1. 论文中需要复刻的内容

### 1.1 预训练任务

UniCORN 的 representation pretraining 是二分类 contact prediction：

- 输入为两个任意几何的点云 `A` 和 `B`。
- 使用同一个 Siamese geometry encoder 分别编码 `A` 和 `B`。
- encoder 对每个点云输出 `N-1` 个局部 patch embedding 和 1 个全局 `[EMB]` embedding：
  - `z^A_1 ... z^A_{N-1}`, `z^A_N`
  - `z^B_1 ... z^B_{N-1}`, `z^B_N`
- contact decoder 对每个 `A` 的局部 patch，用 `(z^A_i, z^B_N)` 预测该 patch 是否与 `B` 接触。
- 训练时交替使用 `A -> B` 和 `B -> A` 两个方向，让同一 encoder 同时学会对两边几何生成局部和全局表征。
- loss 是 patch-wise binary cross entropy。

这与当前 `pretrain/model.py` 的 SDF/diff/postcontact 多头训练不是同一任务。实现时应新增 UniCORN contact pretrain 路径，而不是把它塞进现有 SDF head。

### 1.2 点云 tokenizer / encoder

论文给定的网络超参数：

- `num_points = 512`
- `num_patches = 16`
- `patch_size = 32`
- `embedding_dim = 128`
- `encoder_layers = 4`
- `self_attention_heads = 4`

单个点云编码流程：

1. 对 512 点 surface cloud 用 FPS 选 16 个 patch center。
2. 每个 center 用 kNN 收集 32 个点。
3. patch 内点减去 center 坐标做局部归一化。
4. 小型 MLP tokenizer 编码 patch shape。
5. 对 patch center 加 sinusoidal positional embedding，恢复全局位置信息。
6. 追加 learnable `[EMB]` token。
7. 经过 4 层 Transformer encoder，输出 16 个 patch token 和 1 个 global token。

注意：当前 `pretrain/model.py::TCEPointCloudEncoder` 是 joint tool/object encoder，会 concat 两个点云、加 type embedding 和 cls token。这不是论文 UniCORN 的 Siamese 单云 encoder。实现 UniCORN 时需要单独的 `UnicornGeometryEncoder`，可复用已有 FPS/kNN/PointMAE 基础模块，但语义必须是单点云 Siamese。

### 1.3 Contact decoder

论文附录给定 decoder：

- 三层 conditional MLP。
- 每层是 residual block。
- 使用 conditional batch normalization，conditioning input 是对侧点云的 global embedding `z^B_N`。
- decoder size 是 `(128, 128)`。
- 对每个 patch 预测一个 contact logit。

实现接口建议：

```text
forward(local_tokens: [B, P, D], global_token: [B, D]) -> logits: [B, P]
```

其中 `P=16`, `D=128`。训练时调用两次：

```text
logits_A = decoder(zA_patch, zB_global)
logits_B = decoder(zB_patch, zA_global)
loss = BCE(logits_A, label_A_patch) + BCE(logits_B, label_B_patch)
```

### 1.4 预训练超参数

按论文表格写入默认配置：

| 参数 | 值 |
|---|---|
| batch size | 1024 |
| optimizer | SAM |
| learning rate schedule | cosine |
| base learning rate | 0.0002 |
| min learning rate | 1e-6 |
| max gradient norm | 1000 |
| weight decay | 0.001 |
| rotational augmentation | `(-pi, +pi)` |
| translational augmentation | `(-0.1, +0.1)` |
| scale augmentation | `(exp(-1), exp(+1))` |
| Gaussian noise augmentation | `0.01` |
| positive patch fraction | `0.5` |
| decoder hidden dims | `(128, 128)` |

论文没有在 tex 中明确 SAM 的 `rho`、epoch 数和 warmup；计划中先增加显式配置字段，默认 `rho=0.05`、`epochs=20`、`warmup_epochs=0`，并在 manifest 中标记为 repo default。若后续拿到作者代码或补充材料，应优先覆盖这些字段。

## 2. 使用现有 in-contact 数据的适配策略

指定数据集：

```text
/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/contact_gen_multitool_new/ded4300acdcb31c55ee93f2e86d0f96a0ead8fc4edaae22f749eb9ecbe362e61
```

当前数据是 tool-object contact artifact，文件符合 `contact_pt_env_v1`，包含 object/tool mesh path、scale、bbox center、contact pose、postcontact/physics 字段等。UniCORN 只需要几何 pair 和 patch contact label，不需要 SDF/diff/postcontact 标签。

### 2.1 样本构造

新增 `UnicornContactPairDataset`，从现有 `.pt` contact case 构造样本：

- `cloud_A`：tool surface cloud，512 点。
- `cloud_B`：object surface cloud，512 点。
- pose：使用 contact case 中的 `tool_rotation_E/tool_translation_E` 和 `object_rotation_E/object_bbox_center_E` 将两边点云放到同一 env frame。
- 对 A/B 两边都生成 point-level contact label，再聚合成 patch-level label。

默认只读取最终 `.pt` 文件，不读取：

- `*.candidate.pt`
- `*.stabilized_success.pt`
- `*.physics_debug.pt`
- `*.stabilized.pt`

这与现有 `pretrain.dataset.collect_pt_files` 的过滤策略一致。

### 2.2 Patch-level 正负样本比例

现有数据主要是 in-contact contact cases。对 UniCORN 当前目标来说，这并不意味着 patch-level 正样本过多：即使一个 tool-object pair 处于接触状态，16 个 patches 中通常也只有少数 patch contact label 为 1，大多数 patch 仍然是 negative。

因此实现中不在线生成 pair-level negative / near-contact samples。训练集直接使用现有 in-contact cases；正负比例控制只发生在 contact decoder 的 patch-level loss 上。

论文中的 `Positive patch fraction = 0.5` 按 patch label 实现：

- positive patches 总权重约为 `f`。
- negative patches 总权重约为 `1 - f`。
- 默认 `f = 0.5`。
- 若某个 batch 或样本没有 positive patch，则只对 negative patches 计算有效 BCE，并记录 `empty_positive_patch_count` metric。

### 2.3 Contact label 生成

论文使用“点是否落在对方物体内部”作为 label，并为鲁棒性先做 convex decomposition。仓库现有数据已有 mesh 路径和 convex/tool mesh 信息，计划实现两级后端：

1. 首选 convex/SDF 后端：复用 `utils.geometry.sdf` 或已有 mesh utilities，在 env frame 中判断 `point inside other mesh` 或 `signed_sdf <= 0`。
2. fallback 后端：若精确 inside 不可用，用 `signed_sdf <= contact_eps` 或最近距离阈值生成 near-contact label，并在 manifest 中记录 `label_backend=fallback_distance`。

patch label 聚合：

- point label `label_A_point: [512]` 表示 A 点是否 inside/contact B。
- patchify 后用 patch index 聚合成 `label_A_patch: [16]`。
- 默认 `patch_positive_rule = any`，即 patch 内任一点为 positive 则 patch positive。
- 可配置 `positive_min_points`，用于后续消除单点噪声。

### 2.4 Patch 重采样

论文在 patchify 后调整 decoder 输入，让 positive patch fraction 约为 0.5：

- positive patches 采样概率 `f / P_pos`
- negative patches 采样概率 `(1 - f) / N_neg`
- `f = 0.5`

实现上不要改变 encoder 的 16 个 patch 输入；只对 decoder loss 做 weighted BCE 或采样 mask。推荐使用 weighted BCE，因为保持输出形状稳定、DDP 更简单。

## 3. 代码落点

### 3.1 配置

新增或扩展：

- `configs/config_pretrain.py`
  - 增加 `PretrainCfg.mode: str = "tce_multitask" | "unicorn_contact"`。
  - 增加 `UNICORN_CONTACT_CFG`。
  - 增加 SAM、augmentation、positive fraction、inside-label backend 等字段。
- `configs/config_model.py`
  - 增加 `UnicornCfg(EncoderCfg)`，默认使用论文超参数：
    - `num_points=512`
    - `num_patches=16`
    - `patch_size=32`
    - `encoder_channel=128`
    - `vit_depth=4`
    - `vit_heads=4`
  - `ModelCfg.encoder_backend` 支持 `"unicorn"`。
  - `actor_critic_class` 后续如需 RL 使用，映射到可加载 UniCORN checkpoint 的 adapter。
- 新实验：
  - `configs/experiments/fork_unicorn_pretrain.py`
  - `contact_gen.enabled = False`
  - `pretrain.enabled = True`
  - `pretrain.mode = "unicorn_contact"`
  - `pretrain.dataset_manifest = <指定数据集路径>`
  - `rl.enabled = False`

正式预训练入口保持如下，但该命令不是测试命令，测试阶段不得调用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_experiment.py --config configs/experiments/fork_unicorn_pretrain.py
```

测试只允许对全量配置运行 plan：

```bash
python run_experiment.py --config configs/experiments/fork_unicorn_pretrain.py --mode plan
```

### 3.2 Dataset

新增文件：

- `pretrain/unicorn_dataset.py`

职责：

- 复用 `utils.contact.schema.load_and_validate_contact_pt` 校验 `contact_pt_env_v1`。
- 复用现有 mesh sampling utilities 重建 512 点 tool/object cloud。
- 按 contact case index 展开数据。
- 应用论文 augmentation：
  - 对两个 clouds 同时施加随机 SE(3) rotation/translation。
  - 对两个 clouds 同时施加全局 scale。
  - 加 Gaussian point noise。
- 生成 point-level 和 patch-level contact labels。
- 返回：

```text
points_A: [512, 3]
points_B: [512, 3]
labels_A_point: [512]
labels_B_point: [512]
is_positive_pair: bool
metadata: pt_path, contact_index, tool_id, object_id, label_backend
```

patch index 由 model encoder 内部 FPS/kNN 产生，因此 patch label 可以在 model forward 中根据 encoder 返回的 patch index 聚合，避免 dataset 和 model patchify 不一致。

### 3.3 Model

新增文件：

- `pretrain/unicorn_model.py`

核心模块：

- `UnicornGeometryEncoder`
  - 单点云 encoder。
  - 返回 `patch_tokens`, `global_token`, `patch_idx`, `patch_centers`。
- `ConditionalBatchNorm1d`
- `CMLPResidualBlock`
- `UnicornContactDecoder`
- `UnicornPretrainModel`
  - Siamese 调用 encoder。
  - 双向 decoder。
  - 根据 patch indices 聚合 point labels。
  - 计算 balanced BCE loss 和 metrics。

后续 RL 需要复用 encoder 时，将 `UnicornGeometryEncoder` 移到或 re-export 到：

- `rsl_rl/modules/models/cloud/unicorn.py`

当前该文件已有 `MLPEncoder`，可作为参考，但不能保留 joint encoder 语义。

### 3.4 Trainer

扩展：

- `pretrain/train.py`

实现方式：

- `build_runtime_config` 中带上 `pretrain.mode`。
- `_run_training_loop` 根据 mode 分支：
  - `"tce_multitask"`：现有 `ContactDiffusionModel` 路径。
  - `"unicorn_contact"`：使用 `UnicornContactPairDataset` + `UnicornPretrainModel`。
- 保持 DDP spawn、manifest、checkpoint、wandb 命名逻辑不变。
- checkpoint payload 需要保存：
  - `model`
  - `optimizer`
  - `scheduler`
  - `epoch`
  - `best_val`
  - `metadata.model_family = "unicorn"`
  - `metadata.encoder_state_prefix`
  - `metadata.unicorn_hparams`
  - `metadata.dataset.path/hash`
  - `metadata.label_backend`

### 3.5 SAM optimizer

新增：

- `utils/optim/sam.py` 或 `pretrain/optim.py`

实现 SAM 两步更新：

1. forward/backward 得到梯度。
2. `first_step(zero_grad=True)`。
3. 第二次 forward/backward。
4. clip grad norm 到 1000。
5. `second_step(zero_grad=True)`。

base optimizer 使用 AdamW，参数：

- `lr=2e-4`
- `weight_decay=1e-3`
- `betas=(0.9, 0.999)`
- `eps=1e-8`

cosine scheduler 以 optimizer base lr 更新，min lr 为 `1e-6`。

## 4. Validation 和测试计划

测试阶段不启动 GPU 上的训练，也不把全量预训练作为测试项。测试只覆盖静态配置检查、单元测试和 CPU/极小规模 smoke run；真正的全量 GPU 训练应作为后续人工确认的运行任务，而不是实现计划里的自动测试要求。

### 4.1 单元测试

新增测试覆盖：

- dataset 能从指定 artifact 下读取至少一个 `.pt`，且返回 512 点 A/B。
- augmentation 后 shape 不变，label shape 正确。
- patch label 聚合与 encoder patch index 一致。
- encoder 输出：
  - patch tokens `[B, 16, 128]`
  - global token `[B, 128]`
  - patch idx `[B, 16, 32]`
- decoder 输出 logits `[B, 16]`。
- 双向 loss 可 backward，无 unused parameter。
- SAM optimizer 单步能更新参数。

### 4.2 CPU smoke run

先用小配置验证：

```text
max_files = 2
batch_size = 4
epochs = 1
num_workers = 0
logger = "none"
device = "cpu"
distributed = false
```

命令：

```bash
python run_experiment.py --config configs/experiments/fork_unicorn_pretrain_smoke.py
```

验收：

- 生成 `best.pt`、`last.pt` 和 `.manifest.json`。
- train/val loss 是 finite。
- metrics 中有：
  - `loss`
  - `bce_A`
  - `bce_B`
  - `patch_pos_frac_A`
  - `patch_pos_frac_B`
  - `pair_pos_frac`
  - `contact_acc`
  - `contact_precision/recall`

### 4.3 静态集成验收

不启动训练进程，只检查配置、artifact 路径和 checkpoint schema 兼容性。

验收标准：

- 能通过 `run_experiment.py --mode plan` 解析 UniCORN 预训练实验。
- 全量配置指向指定 contact artifact，但测试中不执行该配置的 `run`。
- batch size 1024 在配置语义上记录为 global batch size。
- artifact 输出路径仍由 `utils.artifacts.resolver` 管理。
- W&B project/run name 遵守现有规则。
- checkpoint manifest 完整记录论文超参数、未明示默认值和数据 hash。
- 只导出 encoder 权重给后续 RL 使用时，不依赖 contact decoder。
- `ModelCfg.encoder_backend="unicorn"` 能解析 checkpoint metadata 并检查 `num_points/num_patches/dim` 兼容性。

## 5. 实施顺序

1. 增加配置和实验文件，先让 `run_experiment.py --mode plan` 能解析 UniCORN pretrain。
2. 实现 `UnicornContactPairDataset`，先只读取现有 in-contact cases 并生成 contact label。
3. 实现 `UnicornGeometryEncoder` 和 `UnicornContactDecoder`，跑单 batch backward。
4. 接入 `pretrain/train.py` 的 `"unicorn_contact"` 分支。
5. 加 SAM + cosine + grad clip。
6. 加 patch-level balanced BCE。
7. 写 checkpoint/manifest，支持 encoder-only 导出。
8. 跑静态配置检查、单元测试和 CPU smoke test。
9. 后续再接 RL adapter，使用冻结 UniCORN encoder 替换现有 TCE。
10. 全量 GPU 预训练只作为单独运行任务保留，不纳入测试计划。

## 6. 需要特别避免的偏差

- 不要复用当前 joint TCE encoder 作为 UniCORN encoder；论文要求 Siamese 单点云 encoder。
- 不要用 SDF/diff/postcontact loss 代替 UniCORN 的 patch contact BCE。
- 不要让 dataset 和 model 各自 patchify；patch label 必须使用 model 实际 FPS/kNN index 聚合。
- 不要只训练 `tool -> object` 单方向；必须同时训练 `A -> B` 和 `B -> A`。
- 不要静默使用 fallback contact label；manifest 必须记录 label backend。
- 不要把 batch size 1024 理解成每卡 batch；默认应作为 global batch。
