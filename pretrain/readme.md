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
- **Object pose** — the random rotation applied to the object.
- **Tool pose** — optimised translation and rotation for each accepted configuration.
- **Contact geometry** — for each accepted pose, sample 5 points from those with $\text{SDF} < 5 \times 10^{-3}$ and record their object-face normals.
- **Near-contact cloud** — randomly sample points with $\text{SDF} < 0.03$; record each point and its signed distance value.

---

## Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--tools-json` | `""` | Path to `tools_adjusted.json` for head area lookup |
| `--head-ratio` | `0.7` | Fraction of surface points sampled from the head area |
| `--save-init` | `""` | Save initial poses (pre-opt) as `.pt` for debugging |
| `--batch-size` | `512` | Number of random tool poses to optimise in parallel |
| `--opt-steps` | `80` | Number of Adam steps |
| `--lr` | `5e-3` | Learning rate |
| `--k-closest` | `32` | Number of closest points for the contact loss |
| `--w-pen` | `50` | Penetration loss weight |
| `--w-contact` | `1` | Contact (attraction) loss weight |
| `--w-floor` | `20` | Floor penalty weight |

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

## Pretraining the Geometry Encoder

The `.pt` files are used to pretrain a dual-encoder + diffusion model:

- **Encoder**: shared `ICPNet` applied separately to the tool point cloud (canonical frame)
  and the object point cloud (loaded from mesh, rotated by `object_rotation`).
- **Diffusion head**: small DDPM MLP conditioned on fused tool+object features.
- **Target (39D)**: contact points `(5×3)` + contact normals `(5×3)` + tool pose `(6D rot + 3D trans)`.
- **Loss**: noise prediction MSE + Chamfer distance on contact points.

```bash
# Generate data first
python gen_dataset.py --num-pairs 200 --gpus 2 3

# Then train (TBD — train.py is the planned entry point)
# python train.py --data-dir tmp_data/ --gpus 2 3
```