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
python visualize_contacts.py --input contact_configs.pt --num-tools 4 --save viz.png
```

---

## Algorithm

### 1. Mesh Loading & Object Grounding
- Load the **object** and **tool** meshes (watertight `.obj`).
- Apply a random SO(3) rotation to the object and shift it so that its lowest vertex sits at `z = 0` (ground plane).

### 2. Head-Area Biased Surface Sampling
- Look up the tool's **head area** from `tools_adjusted.json`. The head area is a cuboid defined in normalised bbox coordinates `[lo, hi]` — e.g. `z ∈ [0.5, 1.0]` means the top half of the tool's bounding box along Z.
- Sample `num_pts` points on the tool surface with a **70/30 bias**: 70% from the head area, 30% from the rest of the body. This ensures the loss focuses on the contact-relevant part of the tool.

### 3. Head → Surface Initialisation
For each of the `N` (batch_size) tool poses:
1. Draw a random rotation `R` and project it so the tool's +Z axis points downward (`z ≤ 0`).
2. Compute the **rotated head centroid**: `c_rot = R @ head_centroid`.
3. Sample a random target point on the **object surface** and offset it slightly (0–5 cm) along the outward normal.
4. Set translation `t = target - c_rot`, placing the head's centre directly at the object surface.
5. **Floor guard**: lift the tool if any point dips below `z = 0`.

This produces much better initial poses than random placement — the head area starts right at the object surface, so contact loss begins near zero.

### 4. Differentiable Test-Time Optimisation (Adam)
Run `opt_steps` iterations of Adam on the 6-DoF pose parameters `(trans, rot6d)` to minimise:

| Loss | Description | Weight |
|------|-------------|--------|
| **L_pen** (penetration) | Max unsigned distance of tool points that are **inside** the object mesh (per-sample max, via Kaolin `check_sign`) | `w_pen = 50` |
| **L_contact** (attraction) | Mean distance of the `k`-closest tool points to the object surface (via Kaolin `point_to_mesh_distance`) | `w_contact = 1` |
| **L_floor** (floor guard) | Mean `ReLU(-z)` for all tool points — penalises anything below `z = 0` | `w_floor = 20` |

**Rotation representation**: 6-D continuous representation (Zhou et al., 2019) to avoid gimbal lock and singularities during gradient descent.

### 5. Filtering & Output
A pose is accepted if:
- Per-sample penetration loss < `pen_eps` (default `1e-3`)
- Per-sample contact loss < `contact_eps` (default `5e-3`)

Output `.pt` file contains:
- `object_mesh_path`, `tool_mesh_path` — absolute paths for downstream use
- `object_rotation` — `(3, 3)` rotation applied to the object
- `object_vertices_grounded` — `(V, 3)` object vertices after rotation & grounding
- `tool_translations` — `(N, 3)` optimised translations
- `tool_rotations` — `(N, 3, 3)` optimised rotation matrices
- `pen_loss`, `contact_loss` — per-sample loss values
- `contact_pts_obj_frame` — `(N, 5, 3)` representative contact points in world frame
- `contact_pts_tool_frame` — `(N, 5, 3)` same points in canonical tool frame
- `contact_normals` — `(N, 5, 3)` outward object face normal at each contact point
- `near_contact_pts` — `(N, 64, 3)` near-contact tool surface points (canonical frame, SDF < `3e-2`)
- `near_contact_sdf` — `(N, 64)` unsigned SDF distance for each near-contact point

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