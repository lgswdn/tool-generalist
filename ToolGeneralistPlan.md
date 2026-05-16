重要：
- 首先，任何可运行的 pretrain / contact generation 的python都不要有任何的cli argument，并且也不要有任何的在对应文件夹目录下的config.py
- 由于最后我们希望能够一键跑实验，所以我们希望每个实验对应一个 Config 类

关于 config：
这个非常重要。这个是一键跑实验并且保证大家CONSISTENT的关键模块。所以一定注意任何，任何参数都一定要写进 config 文件。每个训练文件和模型文件不能有任何的  default value 和 cli argument（唯一 cli argument 是 config）. 

首先，config_xxx.py (xxx!=exp) 里面需要建立自己 config 的 class. 然后一个 config_exp.py 需要建立 ExpCfg 的 class，然后 ExpCfg 需要包含剩下 config 的实例。也就是说，只要有一个 ExpCfg 的实例，我们就获得了整个项目的所有 parameters. 然后在 ExpCfg 中，我们可以对各个子 config 进行微调什么的

首先，config_model.py 记录了关于 pretrain&RL model 的所有模型上的参数，详细到包括到每一个模型的大小应该是什么样子的，多少layer多少head，MLP的hidden是怎么样子的。

然后，config_general.py 记录了比较广泛的参数，比如采多少点云之类的，以及采用的工具的工具列表（是 tools_selected.json 还是什么）。

其次，config_RL.py 记录了所有训 RL 时候需要的参数，比如任何出现在 rsl_rl_ppo_cfg 中的东西。我们希望我们不需要再手动修改 rsl_rl_ppo_cfg.py 的任何内容了。现在 rsl_rl 以及 cfg 中有很多不同的模型，我们现在希望只统一到一个模型，然后加载不同的训练 checkpoint 只需要修改 geometry encoder 的 checkpoint 就行了。但是不同 geometry encoder 的输出规模有所不同（TCE 的输出是 (2*P, B)，但是不同的 geometry checkpoint这个也确实不大一样，比如corn比如point2vec比如concerto），所以这里我们希望每个 encoder 都有一个自己的子 EncoderCfg，比如 TCECfg, P2VCfg, ConcertoCFG. 这都是继承自 config_RL 的. 还有，reward！environment 的 reward 也应该从 config_RL 中加载，所以 config_RL 也应该有一个 RewardCfg. 

config_pretrain.py 记录了 pretrain 的参数，比如用什么 head 训练的（sdf / diff / postcontact 可以用一个三个二进制位表示每个 task 是否参与训练），比如 learning rate / batch size / epoch 数等等。

config_contact.py 记录了 contact 的各种参数，比如 generate 多少个 contact 什么的，以及 contact generation 中的各种 paramters。

---

以上描述了 config。然后说一下如何做到一键跑实验。

不同实验之间可能共享了一点东西，比如不同 encoder 可能共享同一套 contact dataset（但是不同工具列表肯定不会），然后不同的 RL reward 可能共享同一套 encoder (但是也有可能不共享，而且有可能甚至都不需要我们 pretrain而是用的已有的 encoder)。

这个就需要我们给每个 contact dataset 和 encoder 起名字。每一个 contact dataset 唯一对应一个 （GeneralCfg, ContactGenCfg) 的 pair，所以将这个 cfg 的名字用 _ concat 起来得到 contact dataset 的名字，然后建立在 contact/{GeneralCfg name}/ 下，然后每个 pretrained-TCE encoder 唯一对应一个 (GeneralCfg, ContactGenCfg, PretrainCfg, ModelCfg) 的 pair，所以将 PretrainCfg 和 ModelCfg concat 起来得到 encoder 的名字 encodername，然后在 /{GeneralCfg name}/{ContactGenCfg name}/{encodername} 下存储该 encoder 的 checkpoint

哦对了，如果对于跑那种已有 encoder 的实验，那就直接 pretrain contact cfg 什么 = None 就行了。

然后我们跑一次实验的时候，就先检查这个名字的 dataset 是否被生成了，如果没有就生成 dataset；然后看 encoder 是否有，如果没有就 train。然后最后总是要跑一个 RL。

然后一键运行实验的时候，我们会在一开始给定可以用哪些 GPU，通过 CUDA_VISIBLE_DEVICES 方式。

train encoder 的时候需要 wandb，wandb 的 project name 就是 {GeneralCfg name}-{ContactGenCfg name}，曲线名称就是 {pretraincfg name}-{modelcfg name}. 这个 wandb 对 RL 也是同理的，不要用时间来起名字，而是用类似的取名方法。


# Tool-Generalist 重构实现报告

本报告分为两部分：

- 第一部分：实现方法和主要思路，说明整套 pipeline 应如何组织。
- 第二部分：附录和补充材料，给出坐标系、数据 schema、配置、阈值、验收和当前代码依据。

## 第一部分：实现方法和主要思路

### 1. 目标

项目目标是训练一个可跨 tool、跨 object 泛化的 non-prehensile manipulation 系统。Franka 机械臂末端安装不同工具，在桌面上推动/拨动物体，使物体完成目标 reorientation。

整体训练流程分为三阶段：

1. 生成工具-物体接触数据。
2. 用接触数据预训练几何编码器 TCE。
3. 将 TCE 冻结或微调后接入 Isaac Lab + RSL-RL 中训练 RL policy。

重构的目标不是改变研究任务本身，而是统一坐标系、数据格式、配置入口和实验命名，让同一套配置能够一键复现实验。

### 2. 总体 Pipeline

推荐的主流程如下：

```text
ExpCfg
  ├── paths.yaml
  ├── GeneralCfg
  ├── ContactGenCfg
  ├── PretrainCfg
  ├── ModelCfg
  └── RLCfg

run_experiment.py --config <ExpCfg>
  ├── validate full config
  ├── build or reuse contact dataset
  ├── build or reuse pretrained encoder
  └── launch RL training
```

一次实验必须唯一对应一个 `ExpCfg`。`ExpCfg` 不直接写大量参数，而是组合各子配置，并允许在实验级别覆盖少量字段。所有阶段启动前必须先做配置校验，包括路径存在性、工具资产 metadata/schema/hash、物体资产、点云维度、模型 checkpoint 兼容性、GPU 数量、contact dataset schema 和 Isaac Lab task 注册状态。

### 3. 核心几何约定

所有模块必须共享同一套几何约定：

- `center at (0,0,0)` 一律表示 Mesh 的 axis-aligned bounding box center ，即 AABB 长方体的中心，被平移到原点。点云的 center 定义为其采样 Mesh 的 center 。
- rotation 一律绕 bbox center 施加。
- 点云、mesh、USD 只要需要 centralize，都必须使用同一个 bbox center 工具函数。
- 不能在 pretrain、RL observation、contact generation 中各自手写 centroid/mean center 逻辑。

因此应新增共享几何模块，例如：

```text
utils/
  geometry/
    bbox.py
    pointcloud.py
    pose.py
    sdf.py
    mesh_io.py
```

其中至少提供：

- `bbox_center_mesh(mesh_or_verts) -> (center, extent)`
- `centralize_points_by_bbox(points, center) -> (points_centered, extent)`
- `apply_pose_about_bbox_center(points, center, R, t)`
- `signed_sdf(mesh, query_points, frame=...)`

这一步是整个重构的前置条件。当前代码库仍有多处使用 point cloud mean / surface centroid，不能直接视为满足任务目标。

#### 3.1 工具资产消费契约

重构后的代码库不重构工具生成管线（并且工具生成管线现在不在这个电脑上，你不需要关心），只消费类似 `/mnt/project/world_model/tool_generalist/eef` 文件夹中的、生成出的稳定产物。具体的工具文件夹路径从 `paths.yaml` 读取。工具相关代码必须把下面几项当作明确接口，而不是从目录结构或旧脚本里猜：

- 所有工具相关点云都从 `eef/meshdata_adjusted/<tool_id>/coacd/decomposed.obj` 采样。contact generation、pretrain loader、RL observation 和可视化若需要 tool cloud，都必须使用同一份 adjusted decomposed mesh。
- `eef/tools_adjusted.json` 是 adjusted 工具 mesh 对应的功能端 AABB manifest。每个 entry 至少包含 `name` 和 `head_area`，其中 `head_area` 是相对整工具 adjusted bbox 的归一化 axis-aligned AABB，格式为 `[[min_x,min_y,min_z],[max_x,max_y,max_z]]`。
- 工具 USD 焊接到 Franka 的固定挂载参数必须进入 `GeneralCfg`，尤其是 `tool_mount.scale_xyz = [0.1, 0.1, 0.1]`。contact generation 如果直接使用 `decomposed.obj` 做物理几何，也必须使用同一个工具尺度，避免 pretrain 几何和 RL 中机器人末端工具尺寸不一致。
- 工具生成管线是不可变的外部输入，runtime 不运行、不校验、也不记录工具生成 hash。运行时只消费 `tools.tools_adjusted_json`、`tools.tools_selected_json` 和 `tools.meshdata_adjusted_root`；实验仍可通过 `GeneralCfg.tools_selected_json` 覆盖本次选择的 tool manifest。
- `eef/objects_usd/<tool_id>/<tool_id>.usd` 和 `Robots/panda_instanceable_<tool_id>.usd` 只用于仿真资产加载；它们不是 tool point cloud 的 source of truth。

### 4. Contact Dataset 生成

Contact generation 的目标是生成有效的 tool-object 接触状态，并为 TCE pretrain 提供几何、物理和 post-contact 标签。

将 contact dataset 生成拆成三步：

#### 4.1 近似几何接触生成

对每个 `(object, tool, object_pose)`：

1. 从 object mesh 采样 object surface anchors。
2. 从 tool 的 `decomposed.obj` 采样 tool surface anchors，采样概率由 `ContactGenCfg.contact_mode_prob` 和 `tools_adjusted.json` 的 `head_area` 控制，默认 head area 占 70%，非 head Area 占 30%。
3. 对每个 tool-object anchor pair，随机采样 `M` 个 tool rotation。
4. 过滤掉不满足几何约束的候选。
5. 对每个 pair 保留一个 best candidate，写入中间 contact `.pt`。

当前 canonical rejection sampler 位于 `contact_generation.generator`，batch 入口位于 `contact_generation.batch_generate`。

#### 4.2 Isaac Sim Stabilize

近似几何接触只是候选。每个候选都要进入 Isaac Sim：

1. object 使用该 contact case 自己采样到的 mass/friction。
2. tool 使用该 contact case 自己采样到的 mass/friction。
3. ground 使用该 contact case 自己采样到的 friction。
4. 运行 `t_stabilize` steps。
5. 若稳定后仍满足接触判据，则保存稳定后的 object pose、tool pose 和 contact metric。
6. 若不满足接触，则丢弃。

Stabilize 阶段必须写清楚：

- 几何阶段已经过滤 no penetration / no under floor / no pointing upwards。
- Stabilize success 只判断 stabilize 后 tool-object 是否 contact / near-contact。
- 不使用速度或 penetration 作为 Isaac stabilize success 条件。

#### 4.3 Post-contact Rollout

对 stabilize 后的有效 contact：

1. 随机采样 tool delta pose。
2. 在 sim 中控制 tool 达到该 delta pose。
3. 运行 `t_postcontact` steps。
4. 记录 env-frame tool commanded delta pose `ΔP_tool_E` 和 env-frame object observed delta pose `ΔP_obj_E`。
5. 保存最终 contact `.pt`。

最终 `.pt` schema 见附录 B。训练 loader 应读取 schema version，并拒绝未知或不完整 schema。

### 5. Tool Contact Encoder

TCE 输入为 bbox-centered 的 tool point cloud 和 object point cloud：

```text
P'_tool: (B, N, 3)
P'_obj:  (B, N, 3)
```

推荐实现沿用当前 `SDFPointCloudEncoder` ，并更名为 `TCEPointCloudEncoder`：

1. 分别对 tool/object 点云做 FPS。
2. 对每个 FPS center 做 KNN 得到 patch。
3. patch 内点坐标使用 relative-to-patch-center 表示。
4. 共享 PointNet patch encoder 得到 patch token。
5. 加入 patch center positional embedding。
6. 加入 tool/object type embedding。
7. 使用 joint ViT self-attention。
8. 输出 `2P x D` patchwise feature。

默认配置：

```text
N = 512
patch_size = 32
P = 16
D = 128
vit_depth = 12
vit_heads = 4
```

这与当前 `rsl_rl/modules/models/cloud/sdf_encoder.py` 和 `pretrain/new_pretrain/config.py` 基本一致。

### 6. Pretrain Policy

Pretrain 的训练目标是让 TCE 学到可用于接触推理和 RL 的几何表征。推荐保留三类 head：

- `sdf`：预测 tool/object patch 或 point 的 signed distance。
- `diff`：从 noised pre-contact pose 预测 denoise delta。
- `postcontact`：从 contact case 和物理属性预测 object post-contact delta。

数据构造逻辑：

1. loader 读取有效 contact case。
2. 对每个 contact case 采样一个合法 noised tool pose。
3. 用 SLERP / interpolation 生成 `K` 个 pre-contact timestep。
4. 每个 timestep 构造当前 tool/object 点云、relative pose、physics、目标 delta。
5. SDF label 通过 on-the-fly SDF 计算，不依赖旧缓存。

Pretrain decoder 由 TCE 后接多个 condition-specific query decoder 组成。对每个 timestep $k$ ：

1. 当前 bbox-centered point clouds $P'^k_{tool}, P'^k_{obj}$ 先过 TCE，得到 $Z^k \in R^{2P \times D}$。
2. 构造四类 conditioning：
   - `A^k = tool_bbox_center_E^k - object_bbox_center_E`，当前 tool bbox center 相对 object bbox center 的 translation，仍使用 env/world axes。
   - `B = ΔP_tool_E`，post-contact tool 的 env-frame 9D delta pose。
   - `C = ΔP_obj_E`，post-contact object 的 env-frame 9D delta pose。
   - `D`，物理属性，包括 object/tool mass 与 object/tool/ground friction。
3. 四个 MLP decoder-query generators 分别将 `A^k, B, C, D` 映射成 query tokens：
   - `Q_A^k = M_A(A^k)`
   - `Q_B = M_B(B)`
   - `Q_C = M_C(C)`
   - `Q_D = M_D(D)`
4. 每组 query 对 TCE tokens 做 cross-attention，得到 condition features：
   - `F_A^k = CrossAttn(Q_A^k, Z^k)`
   - `F_B^k = CrossAttn(Q_B, Z^k)`
   - `F_C^k = CrossAttn(Q_C, Z^k)`
   - `F_D^k = CrossAttn(Q_D, Z^k)`
5. SDF decoder 使用 `F_A^k` 和每个 patch token 预测 patch signed distance。
6. Denoise decoder 在 `k > 0` 时使用 `F_A^k, F_B^k, F_C^k, F_D^k` 和 timestep sinusoidal embedding 预测 9D denoise pose `\hat ΔP_tool_E^k`。
7. Post-contact decoder 在 `k = 0` 时使用 `F_A^0, F_B^0, F_D^0` 预测 9D object post-contact delta `\hat ΔP_obj_E`。

训练目标：

- SDF head 使用 Soft-L1 loss，并通过 on-the-fly SDF 得到 label。
- Denoise pose 和 post-contact pose 使用 RPDiff 风格的 pose loss，例如 transform chamfer / rotation normalization regularization。
- 多 head 可按 `PretrainCfg.enabled_heads` 同时启用，最终 loss 为各 head 加权和。

需要避免 target leakage：

- 如果 `ΔP_tool_E` 或 `ΔP_obj_E` 是 label，则不能以原始 label 形式直接作为同一 head 的输入。
- conditioning 与 prediction target 必须在 `PretrainCfg` 和 dataset schema 中分开命名。
- 推荐字段命名使用 `cond_*` 和 `target_*`，不要混用 A/B/C。

### 7. RL Policy

RL 中使用冻结后的 TCE 作为 geometry encoder。Observation 推荐统一为：

```text
object_cloud:        (512, 3), env frame, bbox-centered
tool_cloud:          (512, 3), env frame, bbox-centered
object_bbox_center:  (3,), env frame
tool_bbox_center:    (3,), env frame
hand_state:          (9,)
robot_state:         (14,)
previous_action:     (action_dim,)
relative_goal_pose:  (9,)
physics:             (physics_dim,)
```

其中 `tool_cloud` 必须从对应的 `decomposed.obj` 采样，并施加 scale 后再转到 env frame。不要从 `normalized_models`、standalone USD 或 robot USD 反采样 tool cloud。每个 env 启动时应该只采一次点云，然后存下来。

RL env 必须显式创建桌子，不能再依赖没有几何厚度和可视外观的隐式 ground plane 来充当桌面。重构后每个 parallel env 都应有一张独立的 table prim / rigid body：

- 桌子尺寸、厚度、top surface 高度和颜色由 `RLCfg.table` 配置，例如 `size_xy`、`thickness`、`top_z`、`color_rgba`。
- object、tool 和目标 pose 的初始化高度必须以 table top surface 为基准，不应散落硬编码 `z=0`。
- 每个 env 的桌面材质必须可独立设置，避免多个 env 共享同一个 material prim 后无法做 per-env friction randomization。
- table config 属于 RL 实验语义，必须进入本地 manifest 和 W&B metadata。

Policy 计算：

1. 从 observation split 出 tool/object clouds 和 context。
2. 将 tool/object clouds 送入 TCE，得到 `2P x D` tokens。
3. 构造 context：

```text
ctx = [
  tool_bbox_center - object_bbox_center,
  object_bbox_center,
  hand_state,
  robot_state,
  previous_action,
  relative_goal_pose,
  physics
]
```

4. 用 context MLP 生成 `sd_num_query` 个 query。
5. query 对 TCE tokens 做 cross-attention。
6. fusion MLP 输出 actor 和 critic 特征。
7. actor 输出 action distribution，critic 输出 value。

当前 `ActorCriticTG` 已有相近实现，但 context 目前使用 centroid 字段而不是 bbox center 字段。

RL 的 Domain Randomization 必须由 `config_rl.py` 中的 `RLCfg.domain_randomization` 配置，不能继续硬编码在 Isaac env 文件里。`RLCfg` 负责声明随机化范围和开关，Isaac task wrapper 负责把它翻译成 `EventCfg`。至少覆盖当前环境已经使用的随机化项：

- object scale：`prestartup` 阶段采样 object scale。
- object mass/material：reset 阶段采样 object mass、static friction、dynamic friction、restitution。
- tool mass/material：reset 阶段采样 tool body mass、tool collision material。
- table material：reset 阶段对每个 env 单独采样 table top static friction、dynamic friction、restitution。
- ground material：reset 阶段采样 terrain / ground friction 和 restitution。

RL observation 中的 `physics` 字段必须记录本次 reset 后实际采样到的物理参数，而不是 config range。W&B 和本地 manifest 记录 config range、采样 seed、以及每个 run 的 domain randomization preset 名称。

`physics_dim` 必须从启用的物理参数列表推导，不能 hard code 为 7；启用 table surface randomization 时必须包含当前 env 的 table static/dynamic friction 和 restitution。

### 8. Config 和实验管理

推荐新增：

```text
configs/
  config_exp.py
  config_general.py
  config_contact_gen.py
  config_pretrain.py
  config_model.py
  config_rl.py
```

配置原则：

- 所有实验超参数都必须来自 config。
- Python 模块可以有 dataclass 默认值作为 preset，但实际运行必须通过 `ExpCfg` 固化。
- 训练脚本不应再散落业务参数 CLI。
- Isaac Lab 必需 runtime 参数可保留为 wrapper 参数，例如 `--headless`、`--device`、`--distributed`，但它们不应改变实验语义。
- `paths.yaml` 只存机器相关绝对路径；实验参数不放在 `paths.yaml`。
- 物体资产目录、`objects_selected.json`、工具资产目录、tool USD/robot USD 根目录、`tools_adjusted.json`、`tools_selected.json` 和 `meshdata_adjusted_root` 的绝对路径属于 `paths.yaml`
- W&B 的少量通用参数记录在 `GeneralCfg` 中，而不是放在 `PretrainCfg` 或 `RLCfg` 中。pretrain 和 RL 都读取同一组通用 logging 参数，并按固定命名规则生成各自 run。

每个 dataset/checkpoint 都必须保存 manifest：

```text
manifest.yaml
  schema_version
  artifact_type
  artifact_name
  exp_cfg_name
  config_hash
  git_commit
  created_at
  status: complete
  source_paths
  metrics
```

artifact 命名建议：

- contact dataset：`contact/<GeneralCfg.name>/<ContactGenCfg.name>/<config_hash>/`
- encoder checkpoint：`encoder/<GeneralCfg.name>/<ContactGenCfg.name>/<PretrainCfg.name>_<ModelCfg.name>/<config_hash>/`
- RL run：`RL/<GeneralCfg.name>/<ContactGenCfg.name>/<EncoderName>/<RLCfg.name>/<timestamp>/`

W&B run 组织建议：

- project：默认 `{GeneralCfg.name}-{ContactGenCfg.name}`。若本实验不生成或不依赖 contact dataset，则使用 `{GeneralCfg.name}-no-contact`。
- group：默认 `{ExpCfg.name}-{ExpCfg.hash_short}`，由一键入口自动生成，用于把同一次实验的 pretrain 和 RL run 归到一起。
- pretrain run name：默认 `{PretrainCfg.name}-{timestamp}`，`job_type="pretrain"`。
- RL run name：默认 `{RLCfg.name}-{timestamp}`，`job_type="rl"`。
- tags：由 `GeneralCfg.wandb_tags` 提供，并自动追加 `general:<name>`、`contact:<name>`、`pretrain:<name>`、`model:<name>`、`rl:<name>`、`encoder:<type>`。
- contact generation 默认不需要开单独 W&B run；如果要记录生成统计，可使用 `job_type="contact_gen"`，但仍归入同一 project/group。

### 9. 一键实验入口

新增 `scripts/run_experiment.py` 或 repo root `run_experiment.py`：

```bash
python run_experiment.py --config configs/experiments/<exp>.py
```

执行步骤：

1. 加载 `ExpCfg`。
2. 加载 `paths.yaml`。
3. 展开并 freeze 全部子配置。
4. 校验：
   - GPU 数量。
   - 路径存在。
   - object/tool list 存在且非空。
   - contact schema 版本。
   - encoder checkpoint 与 `ModelCfg` 维度一致。
   - Isaac Lab task 和 rsl_rl config entrypoint 可解析。
5. 若 contact dataset 不存在或 manifest 不完整，则生成。
6. 若 pretrain checkpoint 不存在，则训练 encoder。
7. 启动 RL。

每个阶段开始/结束时打印醒目标识、用时、artifact 路径和 manifest 路径。

### 10. 推荐实施顺序

1. 先冻结 schema、frame 定义和工具资产消费契约。
2. 新增 geometry utils，并给 bbox center / SDF / pose transform 写单元测试。
3. 新增 tool runtime asset validator，只检查 `tools_adjusted.json`、`tools_selected.json`、`meshdata_adjusted_root/<tool_id>/coacd/decomposed.obj` 和 Franka mount scale；runtime 不使用工具生成 metadata/hash。
4. 新增 contact schema validator。
5. 改 contact generation，使其输出 v1 schema。
6. 改 pretrain loader，使其只依赖 v1 schema。
7. 改 RL env scene construction，为每个 env 显式创建 table，并从 `RLCfg.table` 读取尺寸、厚度、颜色和默认材质。
8. 改 RL observation，使其使用 bbox center 而不是 centroid，并让 `physics_dim` 从启用的物理参数推导。
9. 新增 config package 和 `ExpCfg`。
10. 新增 artifact manifest 和一键入口。
11. 最后整理旧脚本路径，保留 compatibility wrapper，避免一次性移动破坏 Isaac Lab import。

## 第二部分：附录和补充材料

### A. 坐标系和变换定义

建议使用以下 frame 名称：

| 名称 | 记号 | 定义 |
| --- | --- | --- |
| raw mesh frame | `M_raw` | mesh 文件原始坐标系。 |
| scaled mesh frame | `M` | mesh vertices 乘以 scale 后的坐标系。bbox center 在此 frame 计算。 |
| env frame | `E` | Isaac env/world frame，multi-env 下已减去 env origin。 |
| object bbox-centered env-axes coordinates | `obj_centered_Eaxes` | 仅把原点平移到当前 object bbox center；坐标轴仍与 env/world frame 平行，不使用 object rotation 作为坐标轴。 |
| tool-center frame | `T` | 原点为当前 tool bbox center，朝向为当前 tool rotation。 |

核心公式：

```text
object_points_centered = object_points_M - object_bbox_center_M
tool_points_T   = tool_points_M - tool_bbox_center_M

object_vector_centered_Eaxes = x_E - object_bbox_center_E

R_tool_E = tool_rotation_E
t_tool_E = tool_translation_E
```

9D pose 格式固定为：

```text
pose9d = translation(3) + rotation_matrix[:, :2].reshape(6)
```

delta pose 必须额外声明 composition convention。推荐：

```text
R_after = delta_R @ R_before
t_after = delta_R @ t_before + delta_t
```

如果某个模块采用右乘或局部 frame delta，必须在字段名中显式写出。

#### A.1 Tool asset 坐标和尺度生命周期

后处理管线已经把工具从生成模型输出转成 IsaacLab 可加载资产。重构后的代码库只需要消费下列稳定阶段：

| 阶段 | 路径 / 文件 | 原点 | 尺度 | 用途 |
| --- | --- | --- | --- | --- |
| prepared EEF object | `eef/objects/<tool_id>.obj` | randomize canonical 原点，语义为手柄靠近功能端的端点 | `Z extent = 1`，后续还会变 | 只用于资产生成追溯，不作为训练点云源 |
| normalized model | `eef/normalized_models/<tool_id>.obj` | 整工具 bbox center | 包围球半径约 `1 / 1.03` | DGN/COACD 输入，不作为训练点云源 |
| adjusted decomposed mesh | `eef/meshdata_adjusted/<tool_id>/coacd/decomposed.obj` | 整工具 bbox 底面中心，通常最低点为 `z=0` | 继承 normalized mesh 尺度 | contact/pretrain/RL tool point cloud 唯一 source of truth |
| adjusted convex pieces | `eef/meshdata_adjusted/<tool_id>/coacd/coacd_convex_piece_*.obj` | 同 adjusted decomposed mesh | URDF mesh scale 为 `1` | 物理碰撞资产 pieces |
| tool USD | `eef/objects_usd/<tool_id>/<tool_id>.usd` | 继承 adjusted URDF | 无额外工具尺度 | standalone tool USD |
| Franka robot USD | `Robots/panda_instanceable_<tool_id>.usd` | Panda asset frame | `tool_mount.scale_xyz` 默认再乘 `[0.1,0.1,0.1]` | RL / IsaacLab robot asset |

`adjust_meshes.py` 只做平移：用 `tools.json` 里的 bbox-relative `base_center` 还原实际坐标并移动到原点，不缩放、不旋转。`tools_adjusted.json` 当前只保留 `name` 和 `head_area`，不再保留 `base_center`；使用 adjusted mesh 的代码不应依赖 `base_center` 字段。

#### A.2 工具功能端 AABB 格式

功能端 AABB 来自 `tools_adjusted.json`：

```json
{
  "name": "087_wood_chisel_end_effector_var_005",
  "head_area": [
    [0.08556569111063951, 0.240756837297709, 0.48820650818274736],
    [0.8466492090906699, 0.7821622049944021, 1.000528062516647]
  ]
}
```

约定：

- `name` 必须等于 `tool_id`，并与 `meshdata_adjusted/<tool_id>/coacd/decomposed.obj`、`objects_usd/<tool_id>/<tool_id>.usd`、`panda_instanceable_<tool_id>.usd` 对齐。
- `head_area` shape 固定为 `(2,3)`，第一行为 AABB min，第二行为 AABB max。
- `head_area` 坐标是相对 adjusted decomposed mesh 整体 bbox 的归一化坐标。还原公式为 `head_min = bbox_min + head_area[0] * bbox_size`，`head_max = bbox_min + head_area[1] * bbox_size`。
- 该 AABB 是 axis-aligned，不是 oriented bbox；坐标轴就是 adjusted mesh local frame。
- 由于 AABB 是从 randomize 阶段的功能端 bbox 传播而来，再经过简化/修复，数值可能略小于 `0` 或略大于 `1`。validator 应允许小容差，例如 `[-0.02, 1.02]`，不要静默裁剪 source metadata；采样 head 区域时若需要裁剪，应在采样函数中显式记录。

#### A.3 Franka 工具挂载参数

`batch_generate_franka_single_launch.py` 当前默认把 tool USD reference 到 `/panda/tool_mount`，并写入：

```text
tool_mount.translate = [0.08799998, -4.9709342e-8, 0.926]
tool_mount.rot_wxyz  = [-1.4551854e-11, 0.9238795, 0.38268346, -4.6566123e-10]
tool_mount.scale_xyz = [0.1, 0.1, 0.1]
```

固定关节参数：

```text
attach_link_name = "panda_link7"
joint_name       = "tool_weld_joint"
body0            = panda_link7
body1            = resolved tool rigid body prim
local_pos0       = [0.0, 0.0, 0.107]
local_rot0_wxyz  = [0.9238795, 0.0, 0.0, -0.38268346]
local_pos1       = [0.0, 0.0, 0.0]
local_rot1_wxyz  = [1.0, 0.0, 0.0, 0.0]
```

这组 scale 是工具从无量纲 adjusted mesh 进入机器人末端物理尺寸的关键参数，必须写入 `GeneralCfg.franka_mount.scale_xyz`，并进入 contact dataset metadata、RL run metadata 和本地 artifact manifest。当前值是 uniform scale，因此 contact schema 可把它展开为 `tool_scale_xyz=[0.1,0.1,0.1]`；如果未来改为非 uniform scale，bbox center/extent 和点云采样必须按 xyz scale 重新计算。

#### A.4 工具 runtime 输入契约

工具生成管线被视为不可变的外部输入。训练、contact generation、pretrain 和 RL runtime 不消费工具生成参数、不要求工具生成 metadata，也不记录工具生成 hash。runtime 只需要下面三项：

```text
tools:
  meshdata_adjusted_root: "eef/meshdata_adjusted"
  tools_adjusted_json: "eef/tools_adjusted.json"
  tools_selected_json: "eef/tools_selected.json"
```

`tools_selected_json` 是默认 tool selection manifest；单个实验可以用 `GeneralCfg.tools_selected_json` 覆盖它。运行时 artifact 继续保存普通 `config_hash`、dataset hash、`tools_adjusted.json` 路径和实际 selection manifest 路径，但不保存任何工具生成 hash。

#### A.5 当前工具管线固定参数

下表来自 `~/tool` 当前脚本默认值和实际命令。若实际生成命令覆盖了默认值，以实际命令写入 metadata 为准。

| 模块 | 参数 | 当前值 |
| --- | --- | --- |
| image edit | model / input / output | `"gemini-3.1-flash-image-preview"` / `tool_pictures` / `tool_pictures_edited` |
| image edit | prompt | 记录 prompt 原文或 sha256；当前 prompt 约束工具为“圆柱手柄 + 一个功能性末端”，手柄红色，末端金属色或黑色 |
| Hunyuan 3D | `model` / `region` / `endpoint` | `"3.1"` / `"ap-guangzhou"` / `"ai3d.tencentcloudapi.com"` |
| Hunyuan 3D | input / output / mesh detail | `tool_pictures_edited` / `hunyuan_3d_outputs` / 当前脚本说明为约 50K triangles |
| randomize Open3D | `count` / variants | `19`，实际每个工具写 `variant_0000` 到 `variant_0019` 共 20 个 |
| randomize Open3D | `seed` | `42` |
| randomize Open3D | handle/end scale ranges | `x,y,z` 均为 `[0.8, 1.2]`，`variant_0000` 固定 `[1,1,1]` |
| randomize Open3D | `red_threshold` | `0.55` |
| randomize Open3D | simplify / repair | `simplify_target_faces=1000`，`repair_method="pymeshfix"` |
| randomize Open3D | collapse check | bbox ratio `[0.4,2.5]`，face ratio min `0.5`，repair scale ratio max `2.5` |
| prepare EEF | metadata | `head_area`、`base_center`、`mesh_bounds`、`source_annotation` |
| prepare EEF | validation filter | 默认读取 `validate_tools.json`，实际产物为 1700 个 tool variants |
| normalize | formula | `center=(bbox_max+bbox_min)/2`，`dmax=max(norm(v-center))*1.03`，`v=(v-center)/dmax` |
| coarse DGN/COACD | command source | `post_process/convert.sh` 调用 DexGraspNet `asset_process/decompose.py` |
| coarse DGN/COACD | parameters | `t=0.08`，`k=0.3`，`POOL_WORKERS=32`，URDF mesh scale `1.0`，visual/collision origin `0` |
| fine COACD optional | parameters | `t=0.02`，`k=0.3`，`preprocess_mode="auto"`，`prep_resolution=90`，`resolution=8000`，MCTS `400/5/30`，`no_merge=true`，`seed=1` |
| URDF to USD | converter | `fix_base=False`，`merge_fixed_joints=True`，`force_usd_conversion=True`，joint drive stiffness/damping `0.0`，`target_type="none"` |
| Franka weld | mount xform | translate/rot/scale 见 A.3，scale 默认 `[0.1,0.1,0.1]` |
| Franka weld | physics defaults | `mass_kg=0.2`，`contact_offset=0.005`，`rest_offset=0.0`，`max_depenetration_velocity=5.0`；生成命令使用 `--disable-gravity` 时实际 disable gravity 为 true |

### B. Contact `.pt` Schema

本节定义 contact generation / stabilize / post-contact 之后保存的 `.pt` 文件格式。目标是让 pretrain loader、可视化脚本、Isaac Sim replay、schema validator 都读取同一套字段，避免各阶段重复猜测坐标系和维度。

#### B.1 基本格式

- 文件格式：`torch.save(dict, path)`。
- 顶层对象：Python `dict`。
- 单位：长度为 meter，质量为 kg，friction 为无量纲 Coulomb friction。
- dtype：除特别说明外，tensor 使用 `torch.float32`，路径/id/version 使用 Python `str`，计数使用 Python `int`。
- rotation 存储主格式为 rotation matrix；训练需要的 9D pose 由 `translation(3) + rotation_matrix[:, :2].reshape(6)` 得到。
- 所有 `center` 均表示 axis-aligned bounding box center，不表示 point cloud mean / mesh centroid。

#### B.2 Frame 约定

| 名称 | 记号 | 定义 |
| --- | --- | --- |
| mesh local raw frame | `M_raw` | mesh 文件原始坐标系，未 scale、未 center。 |
| mesh local scaled frame | `M` | 对 `M_raw` 乘以 `object_scale` 或 `tool_scale_xyz` 后的坐标系。bbox center 在该 frame 下计算。 |
| env frame | `E` | Isaac env/world frame；如果是多 env，已减去 `env_origin`。 |
| object bbox-centered env-axes coordinates | `obj_centered_Eaxes` | 原点为当前 object 的 bbox center，坐标轴仍与 env/world frame 平行。bbox-centered 点云 / 向量只减 `object_bbox_center_E`，不乘 `object_rotation_E`。 |
| tool-center frame | `T` | 原点为当前 tool 的 bbox center，朝向为当前 tool rotation。 |

#### B.3 顶层字段

| 字段名 | shape / type | required | frame | 含义 |
| --- | --- | --- | --- | --- |
| `schema_version` | `str` | yes | - | schema 版本，当前写 `"contact_pt_env_v1"`。 |
| `created_at` | `str` | yes | - | ISO-8601 时间戳。 |
| `generator` | `str` | yes | - | 生成脚本或模块名，例如 `"new_pretrain.contact_gen"`。 |
| `config_name` | `str` | yes | - | 生成该 dataset 的 `GeneralCfg.name + "_" + ContactGenCfg.name`。 |
| `config_hash` | `str` | yes | - | 规范化 config dump 的 hash，用于区分同名但参数不同的 dataset。 |
| `num_contacts` | `int` | yes | - | 当前文件内 contact case 数量，记为 `N`。 |
| `object_id` | `str` | yes | - | object 的稳定 id，通常来自 object list。 |
| `tool_id` | `str` | yes | - | tool 的稳定 id，通常来自 tools list。 |
| `object_mesh_path` | `str` | yes | - | object mesh 路径。路径应可由 `paths.yaml` 中的数据根目录解析。 |
| `tool_mesh_path` | `str` | yes | - | tool point cloud / SDF source mesh 路径。工具必须来自 `eef/meshdata_adjusted/<tool_id>/coacd/decomposed.obj`。 |
| `object_scale` | `scalar` | yes | - | object uniform scale。 |
| `tool_scale_xyz` | `(3,)` | yes | - | 应用于 `tool_mesh_path` 的工具几何尺度。当前应等于 Franka 挂载尺度 `[0.1,0.1,0.1]`；不要从 USD xform 里临时反查。 |
| `tool_head_area_aabb_norm` | `(2,3)` | yes | bbox-relative | 从 `tools_adjusted.json` 复制的功能端 AABB，格式为 `[[min],[max]]`，相对 `tool_mesh_path` adjusted mesh 的整工具 bbox。 |
| `object_bbox_center_M` | `(3,)` | yes | `M` | object scale 后、pose 前的 bbox center。centralize object mesh/point cloud 时必须减这个量。 |
| `tool_bbox_center_M` | `(3,)` | yes | `M` | tool scale 后、pose 前的 bbox center。centralize tool mesh/point cloud 时必须减这个量。 |
| `object_bbox_extent_M` | `(3,)` | yes | `M` | object scale 后 bbox size，用于校验 bbox center 和采样范围。 |
| `tool_bbox_extent_M` | `(3,)` | yes | `M` | tool scale 后 bbox size。 |
| `object_point_sample_seed` | `int` | yes | - | loader 重建 object point cloud 时使用的 seed。 |
| `tool_point_sample_seed` | `int` | yes | - | loader 重建 tool point cloud 时使用的 seed。 |

#### B.4 Per-contact 字段

所有 per-contact 字段的第一维必须等于 `num_contacts = N`。

| 字段名 | shape / type | required | frame | 含义 |
| --- | --- | --- | --- | --- |
| `object_rotation_E` | `(N, 3, 3)` | yes | `E` | 当前 object 在 env frame 下的 rotation。所有 object rotation 都绕 bbox center 施加；它不是 persisted tool/contact frame 的 basis。 |
| `object_bbox_center_E` | `(N, 3)` | yes | `E` | 当前 object bbox center 在 env frame 下的位置。若 dataset 完全 object-centered，可为全 0，但字段仍保留。 |
| `tool_translation_E` | `(N, 3)` | yes | `E` | tool bbox center 在 env frame 下的位置。 |
| `tool_rotation_E` | `(N, 3, 3)` | yes | `E` | tool 在 env frame 下的 rotation。 |
| `contact_point_E` | `(N, 3)` | yes | `E` | 任意一个有效 contact point，记录在 env frame。 |
| `object_mass` | `(N,)` | yes | - | 当前 contact case 的 object mass。 |
| `tool_mass` | `(N,)` | yes | - | 当前 contact case 的 tool mass。 |
| `object_friction` | `(N,)` | yes | - | 当前 contact case 的 object friction。 |
| `tool_friction` | `(N,)` | yes | - | 当前 contact case 的 tool friction。 |
| `ground_friction` | `(N,)` | yes | - | 当前 contact case 的 ground friction。 |
| `post_tool_delta_pose9d_E` | `(N, 9)` | yes | `E` | post-contact tool commanded delta pose，相对 stabilized contact tool pose，delta translation 和 rotation composition 都在 env frame。格式为 `delta_t(3) + delta_R[:, :2](6)`。 |
| `post_tool_achieved_delta_pose9d_E` | `(N, 9)` | yes | `E` | post-contact 后实际 tool delta pose，相对 stabilized contact tool pose。格式同上。 |
| `post_object_delta_pose9d_E` | `(N, 9)` | yes | `E` | post-contact 后 object observed delta pose，相对 stabilized contact object pose。格式同上。 |
| `stabilize_steps` | `(N,)` | yes | - | 该 contact case 在 Isaac Sim 中实际 stabilize 的 step 数。 |
| `postcontact_steps` | `(N,)` | yes | - | post-contact rollout 的 step 数。 |

#### B.5 可选 cache 字段

这些字段只用于加速训练或可视化，不作为 source of truth。loader 必须能在缺少这些字段时从 mesh path、scale、bbox center、pose 字段重建等价数据。

| 字段名 | shape / type | required | frame | 含义 |
| --- | --- | --- | --- | --- |
| `object_points_centered` | `(Q, 3)` | no | `obj_centered_Eaxes` | 从 object mesh 采样并 bbox-centered 后的 object point cloud cache；只做 translation centering，不使用 object rotation 作为坐标轴。 |
| `tool_points_T` | `(P, 3)` | no | `T` | 从 `tool_mesh_path` 指向的 adjusted `decomposed.obj` 采样、乘 `tool_scale_xyz`、再 bbox-centered 后的 tool point cloud cache。 |
| `contact_normal_E` | `(N, 3)` | no | `E` | contact point 附近的 object surface normal。 |
| `source_candidate_index` | `(N,)` | no | - | brute-force candidate 的索引，便于 debug。 |
| `debug_metrics` | `dict[str, Tensor]` | no | - | 不参与训练的调试指标。 |

#### B.6 不应存储为训练 schema 的字段

| 字段名 / 类型 | 原因 |
| --- | --- |
| point cloud mean / centroid | 本项目统一使用 bbox center。 |
| object-rotated persisted tool/contact fields | 持久化 `.pt` 不保存 object-rotated tool/contact pose；tool pose、contact point、postcontact delta 均直接保存 env-frame `_E` 字段。 |
| quaternion 和 rotation matrix 同时存 | rotation 只保留 matrix；quat/rot6d/9D 表示由 loader 按需转换。 |
| 成功/失败 contact 混在一个文件内 | contact `.pt` 只保存已通过 stabilize/post-contact 校验的有效 contact。失败样本应进入单独 debug 文件。 |

#### B.7 推荐校验规则

| 校验项 | 规则 |
| --- | --- |
| schema 版本 | `schema_version == "contact_pt_env_v1"`。 |
| shape | 所有 per-contact 字段第一维等于 `num_contacts`。 |
| rotation 正交性 | `R.T @ R` 与 `I` 的最大误差小于 config 中的 `rotation_orth_eps`。 |
| bbox center | 重新从 scaled mesh 计算 bbox center，应与 `*_bbox_center_M` 一致。 |
| tool pose | 直接读取 `tool_rotation_E` 与 `tool_translation_E`；不得通过 object rotation 重建 persisted tool pose。 |
| contact 合法性 | geometry 阶段要求不穿模、tool 不低于 floor、朝向不向上；Isaac stabilize 阶段要求 stabilized 后 tool-object contact / near-contact。 |
| 路径存在 | `object_mesh_path`、`tool_mesh_path` 必须能通过 `paths.yaml` 或绝对路径解析到文件。 |

### C. Contact Accept / Reject 条件

几何 contact generation 的推荐 accept 条件：

| 条件 | 公式 | 默认值来源 |
| --- | --- | --- |
| tool 朝向向下 | `R_tool[2, 2] <= upright_threshold` | 当前 `contact_generation.config` 默认为 `0.0` |
| 不低于桌面 | `min_z(tool_points_E) >= -floor_eps` | 当前默认 `0.0` |
| 不穿模 | `min_sdf(tool_points_E, object_mesh_E) > -epsilon` | 当前默认 `2e-3 m` |

Stabilize accept 条件建议：

```text
stabilized tool-object contact / near-contact is true
```

Post-contact accept 条件建议：

```text
object_delta_pose is finite
```

### D. TCE 结构细节

当前 `SDFPointCloudEncoder` 的核心结构可直接作为 TCE：

| 模块 | 说明 |
| --- | --- |
| input | `tool_pc (B,N,3)` 与 `obj_pc (B,N,3)` |
| grouping | 每个 cloud 内部 FPS + KNN |
| patch encoder | shared PointNet patch encoder |
| position | patch center MLP embedding |
| type | learnable tool/object type embedding |
| transformer | joint ViT self-attention |
| output | `fused_tokens (B,2P,D)` |

默认维度：

```text
N = 512
patch_size = 32
P = N / patch_size = 16
D = 128
vit_depth = 12
vit_heads = 4
```

### E. Pretrain Decoder 架构、数据与 Loss

Pretrain 使用 contact `.pt` 中的 contact case 作为起点。由于 contact `.pt` 只保存接触后的合法样本，loader 需要为每个 case 额外采样一个合法的 noised tool pose，要求该 pose 不穿模且不低于桌面。随后用 SLERP / interpolation 在 noised pose 到 contact pose 之间采样 `K` 个 pre-contact timestep。连同 `k=0` contact timestep，共有 `K+1` 个状态；每个状态都应构造成和 RL 可能观测到的状态一致的输入。

推荐 pretrain batch 字段：

| 字段 | shape | 用途 |
| --- | --- | --- |
| `tool_points_T` | `(B,N,3)` | tool canonical / bbox-centered cloud |
| `object_points_centered` | `(B,N,3)` | object bbox-centered cloud，translation-only centering，axes remain env/world axes |
| `tool_points_E_k` | `(B,K+1,N,3)` | timestep `k` 下 env frame、bbox-centered 后的 tool cloud |
| `object_points_E_k` | `(B,K+1,N,3)` | timestep `k` 下 env frame、bbox-centered 后的 object cloud |
| `rel_tool_object_t_k` | `(B,K+1,3)` | `A^k = tool_bbox_center_E^k - object_bbox_center_E`，env/world axes |
| `cond_tool_post_delta9d_E` | `(B,9)` | `B = ΔP_tool_E`，post-contact tool delta pose |
| `cond_object_post_delta9d_E` | `(B,9)` | `C = ΔP_obj_E`，post-contact object delta pose；只作为允许的条件输入时使用 |
| `physics` | `(B,7)` | `D`，object/tool mass 与 object/tool/ground friction |
| `target_tool_denoise_pose9d_k` | `(B,K,9)` | `k>0` 的 denoise target |
| `target_object_post_delta9d` | `(B,9)` | `k=0` 的 post-contact object delta target |

Decoder 输入和 query 构造：

1. 对每个 timestep `k`，TCE 接收当前 `P_tool'^k`、`P_obj'^k`，输出 `Z^k in R^{B x 2P x D}`。
2. 四类条件分别过 MLP query generator：
   - `Q_A^k = M_A(A^k)`，其中 `A^k in R^3`。
   - `Q_B = M_B(B)`，其中 `B in R^9`。
   - `Q_C = M_C(C)`，其中 `C in R^9`。
   - `Q_D = M_D(D)`，其中 `D in R^7`。
3. 每类 query 对 `Z^k` 做 cross-attention：
   - `F_A^k = CrossAttn(Q_A^k, Z^k, Z^k)`
   - `F_B^k = CrossAttn(Q_B, Z^k, Z^k)`
   - `F_C^k = CrossAttn(Q_C, Z^k, Z^k)`
   - `F_D^k = CrossAttn(Q_D, Z^k, Z^k)`
4. `F_A/F_B/F_C/F_D` 可以保留为 query tokens，也可以先做 mean pooling 后输入 MLP head；具体 pooling 方式应写进 `PretrainCfg`，保证复现。

三个预测 head：

| Head | 使用条件 | 输入 | 输出 |
| --- | --- | --- | --- |
| `sdf` | 所有 `k` | object/tool patch token 与 `F_A^k` | 每个 patch 或 point 的 signed distance `\hat SDF_i^k` |
| `diff` | `k > 0` pre-contact pose | `F_A^k, F_B^k, F_C^k, F_D^k` 与 timestep sinusoidal embedding `e(k)` | 9D denoise pose `\hat ΔP_tool_E^k` |
| `postcontact` | `k = 0` contact pose | `F_A^0, F_B^0, F_D^0` | 9D object post-contact delta `\hat ΔP_obj_E` |

SDF head 以 object 为例：

```text
z_i^k          = object patch token i from Z^k
f_A^k          = pool(F_A^k)
\hat SDF_i^k  = MLP_sdf([z_i^k, f_A^k])
```

真实 `SDF_i^k` 必须根据当前 timestep 下 tool/object 的相对位姿 on-the-fly 计算。这样可以避免坐标系、bbox center 或 mesh 版本变化后旧缓存失效。

Denoise head：

```text
e_k                 = SinusoidalEmbedding(k / K)
f_diff^k            = [pool(F_A^k), pool(F_B^k), pool(F_C^k), pool(F_D^k), e_k]
\hat ΔP_tool_E^k    = MLP_diff(f_diff^k)
```

`target_tool_denoise_pose9d_k` 表示从当前 noised/pre-contact pose 向 contact/post-contact tool pose 去噪的一步或直接目标。若采用 RPDiff 的 transform loss，需要把 9D pose 转换为 translation 与 rotation matrix，并用当前 tool point cloud 构造 transformed child point cloud。

Post-contact head：

```text
f_post              = [pool(F_A^0), pool(F_B^0), pool(F_D^0)]
\hat ΔP_obj_E       = MLP_postcontact(f_post)
```

这里不使用 `F_C`，因为 `C = ΔP_obj_E` 本身就是 post-contact object delta；如果把同一个 `ΔP_obj_E` 直接作为该 head 的输入，会形成 target leakage。

Loss：

```text
L_sdf  = sum_b sum_k sum_i SoftL1(\hat SDF_{b,i}^k, SDF_{b,i}^k)
L_diff = RPDiffPoseLoss(\hat ΔP_tool_E^k, target_tool_denoise_pose9d_k)
L_post = RPDiffPoseLoss(\hat ΔP_obj_E, target_object_post_delta9d)

L = w_sdf * L_sdf
  + w_diff * L_diff
  + w_post * L_post
```

训练循环要求：

1. 按 contact case 读取样本，过滤 `movement_delta_valid == False` 或缺少 post-contact 字段的 case。
2. 每个 batch 内为每个 case 采样 noised legal pose，并生成 `K` 个 timestep。
3. 对 `K+1` 个 timestep 共享 TCE 和 decoder 参数；`sdf` head 覆盖所有 timestep，`diff` head 只覆盖 `k>0`，`postcontact` head 只覆盖 `k=0`。
4. 按 `PretrainCfg.enabled_heads` 和 loss weights 聚合 loss。
5. validation 使用固定 seed 的 noised pose/timestep 采样，避免 val curve 被随机采样噪声主导。
6. checkpoint 至少保存 TCE 权重、decoder 权重、`PretrainCfg` dump、schema version 和 best metric。

### F. RL Observation 和 Policy 细节

推荐 flat observation 顺序：

```text
object_cloud_flat        512*3
tool_cloud_flat          512*3
object_bbox_center       3
tool_bbox_center         3
hand_state               9
robot_state              14
previous_action          action_dim
relative_goal_pose       9
physics                  physics_dim
```

`tool_cloud_flat` 的 source of truth 与 contact/pretrain 一致，必须是 adjusted `decomposed.obj` + `GeneralCfg.franka_mount.scale_xyz`，再用当前 tool pose 变换到 env frame 并按 bbox center centralize。

`ActorCriticTG` 的 split 逻辑必须与 env observation 顺序完全一致。若 action 维度变化，`previous_action` 的维度必须从 action config 推导，不能 hard code。

#### F.1 RL Domain Randomization 细节

RL domain randomization 是 RL 训练语义的一部分，因此归属 `RLCfg`，而不是 `GeneralCfg`、`ContactGenCfg` 或 Isaac env 的硬编码默认值。推荐结构：

```text
RLCfg.table:
  enabled: true
  per_env_instance: true
  size_xy: [1.2, 1.2]
  thickness: 0.05
  top_z: 0.0
  color_rgba: [0.45, 0.45, 0.45, 1.0]
  material:
    static_friction: 0.8
    dynamic_friction: 0.8
    restitution: 0.0

RLCfg.domain_randomization:
  name: "physics_dr_v1"
  enabled: true
  seed_offset: 0
  apply_on_train: true
  apply_on_eval: false

  object:
    scale:
      enabled: true
      mode: "prestartup"
      range: [0.1, 0.2]
    mass:
      enabled: true
      mode: "reset"
      distribution: "uniform"
      range: [0.1, 0.5]
      recompute_inertia: true
    material:
      enabled: true
      mode: "reset"
      static_friction_range: [0.7, 1.0]
      dynamic_friction_range: [0.7, 1.0]
      restitution_range: [0.1, 0.2]
      num_buckets: 256
      make_consistent: true

  tool:
    mass:
      enabled: true
      mode: "reset"
      body_name: "link_coacd_convex_piece_0"
      range: [0.1, 0.5]
    material:
      enabled: true
      mode: "reset"
      body_name: "link_coacd_convex_piece_0"
      static_friction_range: [0.8, 1.5]
      dynamic_friction_range: [0.8, 1.5]
      restitution_range: [0.0, 0.0]

  ground:
    material:
      enabled: true
      mode: "reset"
      static_friction_range: [0.3, 0.8]
      dynamic_friction_range: [0.3, 0.8]
      restitution_range: [0.0, 0.0]
      num_buckets: 256

  table_surface:
    material:
      enabled: true
      mode: "reset"
      per_env: true
      static_friction_range: [0.4, 1.2]
      dynamic_friction_range: [0.4, 1.2]
      restitution_range: [0.0, 0.0]
      num_buckets: 256
```

桥接到 Isaac Lab 时，`RLCfg.domain_randomization` 展开为 task `EventCfg`：

| `RLCfg` 字段 | Isaac event | mode |
| --- | --- | --- |
| `object.scale` | `randomize_rigid_body_scale` | `prestartup` |
| `object.mass` | `randomize_rigid_body_mass` | `reset` |
| `object.material` | `randomize_rigid_body_material` | `reset` |
| `tool.mass` | `randomize_tool_mass` | `reset` |
| `tool.material` | `randomize_tool_friction` | `reset` |
| `table_surface.material` | `randomize_table_surface_material` | `reset` |
| `ground.material` | `randomize_terrain_material` | `reset` |

实现要求：

1. `enabled=false` 时，对应 Isaac event 不应注册，或注册为 no-op。
2. train/eval 必须能选择不同 `domain_randomization` preset；默认 eval 关闭随机化或使用固定 seed。
3. `phys_params` observation 返回实际采样值，包括 object/tool mass、object/tool/table/ground friction 和 restitution；table friction 必须是当前 env reset 实际采样值。
4. 如果 contact dataset 的 mass/friction 范围与 RL domain randomization 范围需要一致，应通过 config validation 检查，而不是在代码中共享隐式常量。
5. 每次 reset 采样到的物理参数可以只进入 rollout observation，不需要逐 step 写入 W&B；W&B 记录范围、preset、seed 和必要的分布统计即可。

### G. Config 结构

建议字段如下：

#### `PathCfg` / `paths.yaml`

`paths.yaml` 只放机器相关绝对路径，建议至少包含：

```text
objects:
  candidates_json
  usd_dir
  obj_dir

tools:
  meshdata_adjusted_root
  objects_usd_root
  robots_usd_root
  tools_adjusted_json
  tools_selected_json
  franka_src_root

models:
  checkpoint_save_path
```

#### `GeneralCfg`

- `name`
- `seed`
- `num_points`
- `tools_manifest`
- `objects_manifest`
- `deterministic`
- `dtype`
- `wandb_enabled`
- `wandb_entity`
- `wandb_mode`
- `wandb_tags`
- `wandb_notes`
- `wandb_metadata_level`
- `wandb_log_code`

W&B 相关字段只保留实现任务目标所需的通用开关和补充信息。project、group、run name 不作为可随意修改的参数，而是由一键入口根据当前 `ExpCfg` 自动生成：

```text
project       = "{GeneralCfg.name}-{ContactGenCfg.name}"
pretrain_run  = "{PretrainCfg.name}-{timestamp}"
rl_run        = "{RLCfg.name}-{timestamp}"
group         = "{ExpCfg.name}-{ExpCfg.hash_short}"
```

如果 `ContactGenCfg=None`，project 使用 `"{GeneralCfg.name}-no-contact"`。

#### `ContactGenCfg`

- `name`
- `num_pairs`
- `num_object_poses`
- `B`
- `M`
- `chunk_B`
- `contact_mode_prob`
- `epsilon`
- `floor_eps`
- `upright_threshold`
- `stabilize_steps`
- `postcontact_steps`
- mass/friction sampling ranges

#### `ModelCfg`

- TCE dimensions：`num_points`、`patch_size`、`encoder_channel`、`vit_depth`、`vit_heads`
- policy fusion dimensions
- supported encoder cfg：`TCECfg`、`P2VCfg`、`ConcertoCfg`

#### `PretrainCfg`

- `name`
- enabled heads：`["sdf", "diff", "postcontact"]`
- `num_precontact_steps` / `K`
- noising schedule：translation/rotation noise range、SLERP/interpolation mode、合法 pose 采样上限
- decoder query counts：`num_query_A`、`num_query_B`、`num_query_C`、`num_query_D`
- decoder dimensions：condition MLP hidden dims、cross-attention layers/heads、pooling mode
- head dimensions：SDF head、denoise head、postcontact head hidden dims
- optimizer
- batch size
- epoch
- loss weights：`w_sdf`、`w_diff`、`w_post`
- validation noising seed / fixed validation sampling switch
- checkpoint policy

#### `RLCfg`

- `name`
- Isaac task id
- PPO params
- env params
- table params：`RLCfg.table`，包含每个 env 的 table enable/per-env instance、`size_xy`、`thickness`、`top_z`、`color_rgba` 和默认 material
- domain randomization：`domain_randomization`，包含 object scale/mass/material、tool mass/material、table surface material、ground material ranges 和 train/eval 开关
- reward config
- encoder checkpoint
- freeze/fine-tune switch

#### `ExpCfg`

- `name`
- `num_gpus`
- child config instances：`GeneralCfg`、`ContactGenCfg`、`PretrainCfg`、`ModelCfg`、`RLCfg`
- artifact policy：reuse / overwrite / fail-if-exists

`run_experiment.py` 负责根据 `GeneralCfg` 中的 W&B 字段创建具体 run。pretrain 脚本和 RL 脚本不再自己决定 project/name/group，只接收已经解析好的 logger 参数或 metadata dump。

### H. 验收标准

最小验收建议：

1. `python -m pytest tests/test_geometry.py`
   - bbox center 与 hand-computed values 一致。
   - rotation about bbox center 正确。
   - centralize 后 bbox center 为 0。
2. `python -m pytest tests/test_contact_schema.py`
   - 能 validate 一个小 contact `.pt`。
   - 缺 required field 会失败。
   - shape mismatch 会失败。
3. `python -m pytest tests/test_tool_assets.py`
   - `tools_selected.json` 中的 tool 都能解析到 `tools_adjusted.json` 和 adjusted `decomposed.obj`。
   - `head_area` 格式为 `(2,3)`，并能还原成功能端 local AABB。
4. contact generation smoke test：
   - 1 个 object、1 个 tool、1 个 pose。
   - 至少输出 1 个 valid contact。
   - manifest status 为 `complete`。
5. pretrain smoke test：
   - 2 个 `.pt` 文件。
   - 1 epoch。
   - 能保存 `best.pt`。
6. RL launch dry-run：
   - `num_envs=2`。
   - 能创建 env、policy、runner。
   - 能跑少量 iteration 或至少 reset/step。
7. full config validation：
   - `ExpCfg.validate()` 在启动前发现路径不存在、checkpoint 维度不匹配、GPU 数量不匹配。

### I. W&B Metadata 和复现记录

每个 W&B run 都必须记录足够 metadata，使得只看 W&B 页面就能还原本次实验依赖了哪些 config、dataset、checkpoint 和代码版本。pretrain 与 RL 是两个独立 run，但应共享同一 project 和 group。

#### I.1 所有 run 都要记录的 common metadata

| 字段 | 说明 |
| --- | --- |
| `exp_cfg_name` | `ExpCfg.name`。 |
| `exp_cfg_path` | 启动时传入的 config 路径。 |
| `exp_cfg_hash` | 完整 frozen `ExpCfg` 的 hash。 |
| `general_cfg_name` / `hash` | `GeneralCfg` 名称和 hash。 |
| `tools_selected_json` | 本次实验实际使用的 tool selection manifest；可由 `GeneralCfg.tools_selected_json` 覆盖。 |
| `contact_gen_cfg_name` / `hash` | `ContactGenCfg` 名称和 hash；若为 `None`，记录 `None`。 |
| `pretrain_cfg_name` / `hash` | `PretrainCfg` 名称和 hash；若跳过 pretrain，记录 `None`。 |
| `model_cfg_name` / `hash` | `ModelCfg` 名称和 hash。 |
| `rl_cfg_name` / `hash` | `RLCfg` 名称和 hash；pretrain run 中也记录，用于说明后续 RL 目标。 |
| `config_dump` | 完整 frozen config，作为 wandb config 或 artifact。 |
| `config_overrides` | `ExpCfg` 对子 config 的覆盖项。 |
| `git_commit` | 当前 repo commit。 |
| `git_dirty` | 工作区是否有未提交改动。 |
| `python_version` | Python 版本。 |
| `torch_version` | PyTorch 版本。 |
| `cuda_version` | CUDA 版本。 |
| `isaac_sim_version` | Isaac Sim 版本；非 Isaac 阶段可记录 `unknown`。 |
| `isaac_lab_version` | Isaac Lab 版本；非 Isaac 阶段可记录 `unknown`。 |
| `hostname` | 机器 hostname。 |
| `user` | 运行用户。 |
| `cwd` | 启动工作目录。 |
| `command` | 原始启动命令。 |
| `cuda_visible_devices` | `CUDA_VISIBLE_DEVICES`。 |
| `num_gpus_requested` | `ExpCfg.num_gpus`。 |
| `num_gpus_visible` | 实际可见 GPU 数。 |
| `seed` | `GeneralCfg.seed`。 |
| `timestamp` | run 创建时间。 |
| `paths_yaml` | 使用的 `paths.yaml` 路径。 |
| `output_root` | 代码/视频/log 输出根目录。 |
| `artifact_root` | checkpoint/dataset 根目录。 |

#### I.2 Contact dataset metadata

| 字段 | 说明 |
| --- | --- |
| `contact_dataset_name` | dataset 名称。 |
| `contact_dataset_path` | dataset 路径。 |
| `contact_dataset_manifest` | manifest 路径。 |
| `contact_dataset_hash` | dataset manifest 或 config hash。 |
| `contact_schema_version` | contact `.pt` schema version。 |
| `object_manifest_path` | object list 路径。 |
| `object_count` | object 数量。 |
| `tool_manifest_path` | tool list 路径。 |
| `tool_count` | tool 数量。 |
| `tool_mesh_root` | tool mesh 根路径。 |
| `tool_pointcloud_source_stage` | 当前应为 `"meshdata_adjusted"`。 |
| `tool_pointcloud_mesh_template` | 当前应为 `eef/meshdata_adjusted/{tool_id}/coacd/decomposed.obj`。 |
| `tool_head_area_manifest` | `tools_adjusted.json` 路径。 |
| `tool_head_area_format` | 固定为 `bbox_relative_aabb_minmax_2x3`。 |
| `tool_scale_xyz` | contact generation 中应用到 adjusted decomposed mesh 的工具尺度，当前应为 `[0.1,0.1,0.1]`。 |
| `franka_mount_scale_xyz` | Franka `tool_mount` 的 scale，必须与 `tool_scale_xyz` 一致。 |
| `object_mesh_root` | object mesh 根路径。 |
| `contact_num_pairs` | sampled pair 数量。 |
| `contact_num_object_poses` | 每个 pair 的 object pose 数量。 |
| `contact_B` / `contact_M` / `contact_chunk_B` | rejection sampler 参数。 |
| `contact_mode_prob` | tool head/body contact sampling 比例。 |
| `penetration_epsilon` | penetration tolerance。 |
| `contact_floor_eps` | floor tolerance。 |
| `contact_upright_threshold` | tool downward orientation threshold。 |
| `stabilize_steps` | stabilize step 数。 |
| `postcontact_steps` | post-contact rollout step 数。 |
| `mass_ranges` | object/tool mass sampling ranges。 |
| `friction_ranges` | object/tool/ground friction ranges。 |
| `num_contacts_generated` | 近似几何 contact 数。 |
| `num_contacts_stabilized` | stabilize 后保留数。 |
| `num_contacts_postcontact` | post-contact 后保留数。 |
| `acceptance_rate_geometry` | 几何阶段通过率。 |
| `acceptance_rate_stabilize` | stabilize 阶段通过率。 |

#### I.3 Pretrain run metadata

| 字段 | 说明 |
| --- | --- |
| `wandb_job_type` | 固定为 `"pretrain"`。 |
| `pretrain_run_name` | `{PretrainCfg.name}-{timestamp}`。 |
| `enabled_heads` | 启用的 heads，例如 `["sdf", "diff", "postcontact"]`。 |
| `num_points` | 每个点云点数。 |
| `patch_size` | TCE patch size。 |
| `num_patches` | 每个 cloud 的 patch 数。 |
| `encoder_channel` | TCE token dim。 |
| `vit_depth` / `vit_heads` | TCE transformer 参数。 |
| `cross_attn_heads` / `cross_attn_layers` | pretrain conditioning cross-attention 参数。 |
| `num_precontact_steps` | SLERP / interpolation 生成的 `K` 个 pre-contact timestep。 |
| `query_counts_A_B_C_D` | `M_A/M_B/M_C/M_D` 各自生成的 query 数量。 |
| `decoder_pooling` | query features 进入 head 前的 pooling/flatten 策略。 |
| `head_hidden_dims` | `sdf`、`diff`、`postcontact` 三个 head 的 hidden dims。 |
| `optimizer` | optimizer 类型。 |
| `learning_rate` | pretrain lr。 |
| `weight_decay` | weight decay。 |
| `batch_size` | pretrain batch size。 |
| `epochs` | epoch 数。 |
| `num_workers` | dataloader worker 数。 |
| `loss_weights` | sdf/diff/postcontact 等 loss weights。 |
| `noise_schedule` | diffusion/pre-contact noise 参数。 |
| `require_movement` | 是否要求 movement delta 字段存在。 |
| `movement_valid_filter` | 是否过滤 `movement_delta_valid == False`。 |
| `train_split_size` | train case 数。 |
| `val_split_size` | val case 数。 |
| `resume_checkpoint` | resume checkpoint 路径或 `None`。 |
| `output_checkpoint_dir` | checkpoint 输出目录。 |
| `best_checkpoint_path` | 当前 best checkpoint 路径。 |
| `ddp_world_size` | DDP world size。 |

Pretrain run 至少记录以下曲线：

- total loss。
- 各 head loss。
- train/val loss。
- SDF error 分布。
- denoise translation/rotation error。
- postcontact pose error。
- learning rate。
- gradient norm。
- epoch time。

#### I.4 RL run metadata

| 字段 | 说明 |
| --- | --- |
| `wandb_job_type` | 固定为 `"rl"`。 |
| `rl_run_name` | `{RLCfg.name}-{timestamp}`。 |
| `isaac_task_id` | Gym/Isaac task id。 |
| `num_envs` | RL parallel env 数。 |
| `max_iterations` | PPO max iterations。 |
| `num_steps_per_env` | PPO rollout length。 |
| `ppo_algorithm_cfg` | PPO hyperparameters dump。 |
| `reward_cfg` | reward terms、weights、thresholds。 |
| `termination_cfg` | termination terms。 |
| `command_cfg` | target command config。 |
| `action_cfg` | action type、scale、dim。 |
| `observation_layout` | flat observation 字段顺序和维度。 |
| `table_cfg` | 完整 `RLCfg.table` dump，包括 per-env instance、size、thickness、top_z、color 和默认 material。 |
| `tool_robot_usd_root` | `panda_instanceable_<tool_id>.usd` 所在根目录。 |
| `tool_objects_usd_root` | standalone tool USD 根目录。 |
| `tool_pointcloud_mesh_template` | RL tool cloud 采样使用的 adjusted `decomposed.obj` 模板。 |
| `tool_head_area_manifest` | RL 使用的 `tools_adjusted.json`。 |
| `franka_mount_translate` / `rot_wxyz` / `scale_xyz` | 工具焊接到 Franka 的固定 xform；当前 scale 应为 `[0.1,0.1,0.1]`。 |
| `tool_weld_joint_cfg` | attach link、joint name、local pos/rot。 |
| `domain_randomization_name` | `RLCfg.domain_randomization.name`。 |
| `domain_randomization_enabled` | 当前 run 是否启用 RL domain randomization。 |
| `domain_randomization_seed_offset` | 随机化 seed offset。 |
| `domain_randomization_apply_on_train` / `apply_on_eval` | train/eval 是否启用。 |
| `object_scale_range` | object scale randomization range。 |
| `object_mass_range` | object mass randomization range。 |
| `object_material_ranges` | object static/dynamic friction 与 restitution ranges。 |
| `tool_mass_range` | tool mass randomization range。 |
| `tool_material_ranges` | tool static/dynamic friction 与 restitution ranges。 |
| `table_surface_material_ranges` | table surface static/dynamic friction 与 restitution ranges；若关闭随机化则记录固定值。 |
| `ground_material_ranges` | ground static/dynamic friction 与 restitution ranges。 |
| `domain_randomization_cfg` | 完整 `RLCfg.domain_randomization` dump。 |
| `encoder_type` | `TCE`、`Point2Vec`、`Concerto` 等。 |
| `encoder_checkpoint_path` | encoder checkpoint 路径。 |
| `encoder_checkpoint_hash` | checkpoint hash 或 manifest hash。 |
| `freeze_encoder` | 是否冻结 encoder。 |
| `policy_class` | actor-critic class。 |
| `policy_hidden_dims` | actor/critic/fusion hidden dims。 |
| `sd_num_query` | state-dependent query 数。 |
| `log_dir` | 本地 RL log 目录。 |
| `resume_run` | resume 的 RL run/checkpoint。 |

RL run 至少记录以下曲线：

- episode reward。
- value loss。
- surrogate loss。
- entropy。
- KL。
- learning rate。
- success rate。
- object-goal distance。
- object rotation error。
- contact reward。
- episode length。
- action norm。
- per-object success 统计。
- per-tool success 统计。

#### I.5 Artifact 复现记录

每个本地 artifact 的 manifest 必须记录：

- 完整 config dump。
- config hash。
- git commit。
- Python、PyTorch、Isaac Sim、Isaac Lab、CUDA 版本。
- 使用的 object/tool manifests。
- 使用的 `tools_selected_json` 路径、`tools_adjusted.json` 路径和普通 dataset/config hash。
- 工具点云 source template 和 Franka mount scale/pose。
- dataset schema version。
- checkpoint source。
- wandb project、group、run id。

随机性控制：

- Python `random`
- NumPy
- PyTorch CPU/CUDA
- Isaac Lab env seed
- contact generation worker seed
- DDP rank offset seed

### J. 推荐最终文件结构

```text
tool-generalist/
  configs/
    config_exp.py
    config_general.py
    config_contact_gen.py
    config_pretrain.py
    config_model.py
    config_rl.py
  utils/
    geometry/
    config/
    logging/
    artifacts/
  scripts/
    run_experiment.py
    contact_gen/
    pretrain/
    rl/
  source/
    IsaacLab_nonPrehensile/
  rsl_rl/
  contact/
  paths.yaml
```
