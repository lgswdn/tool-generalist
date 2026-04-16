# Contact Configuration Generation

Generates collision-free contact configurations between a rigid tool and a rigid object using differentiable test-time optimisation (Kaolin + Adam).

## Prerequisites

```bash
# Kaolin (e.g. torch 2.7.0, CUDA 12.8)
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html
```

## Quick Start

**Generate for a single pair:**
```bash
python contact_gen.py \
    --object /mnt/afs/zhuwenxuan/DGN/coacd_normalized/core-pistol-5a2bb05af1dedd91e641b9ab504917bf.obj \
    --tool /mnt/afs/zhuwenxuan/project/inp/tool-generalist/RobotSmith/eef/normalized_models/006_claw_gripper_end_effector_var_001.obj \
    --tools-json /mnt/afs/zhuwenxuan/project/inp/tool-generalist/RobotSmith/eef/tools_adjusted.json \
    --device cuda:0 \
    --save-init init.pt
```

**Visualize results:**
```bash
python visualize_contacts.py \
    --input init.pt \
    --num-tools 1 \
    --save viz_init.png

python visualize_contacts.py --input contact_configs.pt --num-tools 1 --save viz.png

# Final converged poses
python visualize_contacts.py \
    --input contact_configs.pt \
    --num-tools 4 \
    --save viz.png
```

---

## Algorithm

### 1. Initialisation
- Sample independent random SO(3) rotations for both the **object** and the **tool**.
- Ground the object: shift it along Z so that its lowest point sits at $z = 0$.

### 2. Biased Tool-Surface Sampling
Sample surface points on the tool with a **70 / 30 head-area bias**:
- **70 %** of points are drawn from the **head area** (the contact-relevant region defined in `tools_adjusted.json`).
- **30 %** are drawn from the rest of the handle / body.

### 3. Placement & Floor Guard
1. Pick a random object surface point $o$ and a random (biased) tool point $t$.
2. Translate the tool so that $t$ lands at distance $d \in [0,\, 0.02]$ m outside $o$ along the object surface normal.
3. **Floor guard**: if any tool point has $z < 0$ after placement, lift the entire tool until all points are at or above $z = 0$.

### 4. Energy Minimisation ($K = 8$)
Run **200 steps** of Adam ($lr = 5 \times 10^{-3}$) on the 6-DoF tool pose $(\mathbf{t},\, R_{6D})$ to minimise a weighted sum of three terms:

| Term | Definition | Weight |
|------|-----------|--------|
| $\ell_{\text{pen}}$ | Sample $K$ points with $\text{SDF} < 0$ (inside the object); minimise their mean $|\text{SDF}|$ | **30** |
| $\ell_{\text{contact}}$ | Sample $K$ points with $\text{SDF} \ge 0$ (outside); minimise their mean SDF (attraction toward surface) | **1** |
| $\ell_{\text{floor}}$ | Sample $K$ points with $z < 0$; minimise their mean $|z|$ (keep tool above ground) | **20** |

**Rotation parameterisation**: 6-D continuous representation

### 5. Filtering & Output

#### Accepted configurations
A configuration is kept when **both** criteria are met:
- $\min_i \text{SDF}_i < 3 \times 10^{-4}$ (tool reaches the surface)
- $\ell_{\text{contact}} < 8 \times 10^{-3}$ (good surface contact)

#### Output `.pt` file contains:

| Field | Shape | Description |
|-------|-------|-------------|
| `object_rotation` | `(3,3)` | Rotation applied to the object |
| `tool_translations` | `(N,3)` | Optimised tool translations |
| `tool_rotations` | `(N,3,3)` | Optimised tool rotations |
| `pen_loss` / `contact_loss` | `(N,)` | Per-config quality metrics |
| `contact_pts_obj_frame` | `(N,5,3)` | 5 contact points in world/object frame |
| `contact_pts_tool_frame` | `(N,5,3)` | Same points in canonical tool frame |
| `contact_normals` | `(N,5,3)` | Outward object face normal at each contact point |
| `near_contact_pts` | `(N,64,3)` | Near-contact tool pts in canonical frame ($\text{SDF} < 0.03$) |
| `near_contact_sdf` | `(N,64)` | Unsigned SDF for each near-contact point |
| `all_pts_canonical` | `(N,P,3)` | **All** $P$ tool surface points in canonical frame |
| `all_pts_sdf` | `(N,P)` | Signed SDF of each tool point to object (+outside / −inside) |
| `obj_pts_world` | `(Q,3)` | $Q$ object surface points in world frame *(fixed across configs)* |
| `obj_pts_sdf` | `(N,Q)` | Signed SDF of each object point to tool (+outside / −inside) |

The last two pairs enable **mutual SDF supervision**: both the tool and the object can act as the query side, allowing an encoder to learn geometry from either perspective.

---

## Geometry Learning: SDFSegmentor

We use a joint-ViT architecture (`SDFSegmentor`) that processes concatenated tool and object point clouds to predict contact probability and surface alignment.

### Usage
```bash
# Train the model
python train_sdf_segmentor.py \
    --data-dir ./results \
    --batch-size 64 \
    --epochs 50 \
    --lr 1e-4

# Inference
python infer_sdf_segmentor.py --input scene.pt --model checkpoints/best.pth
```

### CLI Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--embed-dim` | `256` | ViT hidden dimension |
| `--depth` | `6` | Number of transformer blocks |
| `--num-heads` | `8` | Attention heads |
| `--use-sdf` | `True` | Include SDF values as input features |
| `--loss-weight-contact` | `1.0` | Weight for contact classification loss |

---

## Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--tools-json` | `""` | Path to `tools_adjusted.json` for head area lookup |
| `--save-init` | `""` | Save initial poses (pre-opt) as `.pt` for debugging |
| `--batch-size` | `512` | Number of random tool poses to optimise in parallel |
| `--num-pts` | `512` | Uniform surface cloud size (tool) |
| `--opt-steps` | `200` | Number of Adam steps |
| `--lr` | `5e-3` | Learning rate |
| `--w-pen` | `30` | Penetration loss weight |
| `--w-contact` | `1` | Contact (attraction) loss weight |
| `--w-floor` | `20` | Floor penalty weight |
| `--pen-max-eps` | `3e-4` | Max per-config penetration depth to accept |
| `--contact-eps` | `8e-3` | Max contact loss to accept |

---

## Export to OBJ (Debug)

`export_contacts_obj.py` merges the object + tool poses into a single `.obj` (+ `.mtl`) file,
viewable in MeshLab, Blender, VS Code .obj preview, or any 3D viewer.

```bash
# Default: writes <input_stem>_scene.obj next to the .pt file
python export_contacts_obj.py --input contact_configs.pt --num-tools 8

# Custom output path
python export_contacts_obj.py --input contact_configs.pt --num-tools 4 -o debug_scene.obj
```

---

## Isaac Sim Gallery Viewer

`view_contacts_isaac.py` spawns batch-generated contact configs in a tiled Isaac Sim
stage for visual inspection. Each `.pt` file gets its own grid cell.

Must be run inside the Isaac Lab Python environment:

```bash
# Interactive viewport — view multiple config files
isaaclab -p pretrain/view_contacts_isaac.py \
    --inputs results/config_001.pt results/config_002.pt results/config_003.pt \
    --num-tools-per-cell 4 \
    --spacing 3.0

# Glob all .pt files from a directory
isaaclab -p pretrain/view_contacts_isaac.py \
    --input-dir results/ \
    --num-tools-per-cell 4

# Headless screenshot
isaaclab -p pretrain/view_contacts_isaac.py \
    --inputs results/*.pt \
    --save gallery.png
```

| Flag | Default | Description |
|------|---------|-------------|
| `--inputs` | `[]` | One or more `.pt` files to visualise |
| `--input-dir` | `""` | Directory to glob for `*.pt` files |
| `--num-tools-per-cell` | `4` | Max tool poses per grid cell |
| `--spacing` | `3.0` | Grid cell spacing in metres |
| `--cols` | `0` | Grid columns (0 = auto √N) |
| `--save` | `""` | Screenshot output path (enables headless) |
| `--settle-steps` | `20` | Render steps before screenshot |

---

## Pretraining the Geometry Encoder (`SDFSegmentor`)

The `.pt` files are used to pretrain `SDFSegmentor`—a joint-ViT encoder that predicts
**mutual signed SDF values** (tool→object and object→tool) from 3-D surface clouds.

### Architecture

```
tool_pc  [B, N, 3] ──┐
                      ├─ cat → [B, 2N, 3]
obj_pc   [B, N, 3] ──┘
         │
         ▼  per-group FPS+KNN  (N points per stream, K points per patch)
         │
         ▼  patches  [B, 2P, K, 3]
         │
         ▼  PointNetPatchEncoder  (shared, permutation-invariant)
            per-point MLP → max-pool + mean-pool → projection
         │
         ▼  patch tokens  [B, 2P, D]
         │
         ├─ + positional embedding  (patch centre → D, sinusoidal MLP)
         ├─ + type embedding        (tool=0 / object=1, learnable)
         ├─ + CLS token prepended
         │
         ▼  joint ViT  (depth L, H heads)
            ← cross-stream attention between tool & object patches
         │
         ├─ global_feat  = CLS output      [B, D]
         ├─ tool_tokens  = first P tokens  [B, P, D]
         └─ obj_tokens   = last  P tokens  [B, P, D]
                 │
          ┌──────┴──────┐
          ▼             ▼
    tool SDF head   obj SDF head   (independent MLP heads)
    [B, N] or [B, P]
```

**Key design decisions:**
- **PointNet patch encoder** (max+mean pool): permutation-invariant, captures
  local geometry extrema better than a plain MLP applied to flattened coords.
- **Joint ViT**: tool and object patches attend to each other implicitly—patch
  tokens already encode cross-stream proximity before the SDF heads fire.
- **Type embeddings** (tool=0 / object=1): additive learnable tokens, following
  the `ActorCriticMultiICP` pattern.
- **Loss**: Huber (smooth-L1) on predicted vs. GT SDF for both streams.

### Quick Start

```bash
# Step 1: generate data
python gen_dataset.py --num-pairs 200 --gpus 2 3

# Step 2: train (point-level SDF, defaults)
python train.py --data-dir tmp_data/

# Patch-level SDF with a deeper ViT
python train.py --data-dir tmp_data/ --head-mode patch --patch-agg mean \
    --vit-depth 6 --vit-heads 8 --encoder-channel 256

# Multi-GPU
torchrun --nproc_per_node=2 train.py --data-dir tmp_data/

# Resume
python train.py --data-dir tmp_data/ --resume checkpoints/last.pt
```

### Encoder CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--num-pts` | `512` | Points per cloud (N) |
| `--patch-size` | `32` | Points per FPS patch (K) |
| `--encoder-channel` | `128` | Patch token dimension (D) |
| `--vit-depth` | `4` | Number of ViT transformer layers |
| `--vit-heads` | `4` | Number of ViT attention heads |
| `--freeze-encoder` | `False` | Freeze encoder; train SDF heads only |
| `--head-mode` | `point` | `point` = per-point SDF \| `patch` = per-patch SDF |
| `--patch-agg` | `mean` | GT aggregation for patch mode (`mean`/`min`/`max`) |