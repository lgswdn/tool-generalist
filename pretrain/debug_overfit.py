#!/usr/bin/env python3
"""debug_overfit.py — Overfit diagnostics for diffusion convergence.

Condition modes:
  (default)        Encoder features + centroid bias
  --cond-pose      Delta_pose (answer) as condition
  --cond-index     Unique learnable embedding per sample
  --cond-centroid  Centroids only (6D), no encoder
  --cond-init-pose Init pose (9D) as condition

Head modes:
  (default)        TransformerForDiffusion (cross-attention)
  --cond-mlp       MLP noise predictor (bypasses cross-attention)

Diagnostics:
  --diag           Encoder variance diagnostic before training
  --detach-enc     Stop gradient through encoder features

Usage:
    python debug_overfit.py --full --cond-mlp --warmup-reg 200 --lr 1e-3 --epochs 700
    python debug_overfit.py --full --cond-index --lr 1e-3 --epochs 400
    python debug_overfit.py --full --diag
    python debug_overfit.py --full --cond-init-pose --reg-only --lr 1e-3
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import TrainConfig
from model import JointModel, MLPNoisePredictor
from dataset import ContactDataset, collect_pt_files
from pytorch3d.ops import sample_farthest_points


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def compute_patch_centers(tool_pc, obj_pc, num_patches=16):
    """Compute FPS patch centers for tool and object clouds.
    Returns (B, 2*P, 3) — same ordering as encoder tokens."""
    tool_ctrs, _ = sample_farthest_points(tool_pc.contiguous(), K=num_patches, random_start_point=False)
    obj_ctrs, _ = sample_farthest_points(obj_pc.contiguous(), K=num_patches, random_start_point=False)
    return torch.cat([tool_ctrs, obj_ctrs], dim=1)


# --------------------------------------------------------------------------- #
# Encoder Variance Diagnostic
# --------------------------------------------------------------------------- #

def run_variance_diagnostic(model, device, ds):
    m = model.module if hasattr(model, 'module') else model
    m.eval()
    conds_raw, conds_biased, tool_cs, dps = [], [], [], []
    N = min(len(ds), 300)

    print(f"\n{'='*60}")
    print(f"ENCODER VARIANCE DIAGNOSTIC ({N} samples)")
    print(f"{'='*60}")

    with torch.no_grad():
        for i in range(N):
            s = ds[i]
            if s['tool_pc_init'] is None:
                continue
            tpc = s['tool_pc_init'].unsqueeze(0).to(device)
            opc = s['obj_pc'].unsqueeze(0).to(device)
            enc = m.encoder.encode(tpc, opc)
            cond = torch.cat([enc.tool_tokens, enc.obj_tokens], dim=1)
            tc = tpc.mean(dim=1)
            oc = opc.mean(dim=1)
            pb = m.cond_pos_proj(torch.cat([tc, oc], -1))
            conds_raw.append(cond.squeeze(0).cpu())
            conds_biased.append((cond + pb.unsqueeze(1)).squeeze(0).cpu())
            tool_cs.append(tc.squeeze(0).cpu())
            if s['delta_pose'] is not None:
                dps.append(s['delta_pose'])

    for name, tensor in [("RAW", torch.stack(conds_raw)),
                          ("WITH POS BIAS", torch.stack(conds_biased))]:
        Ns = tensor.shape[0]
        var_avg = tensor.var(dim=0).mean().item()
        pooled = tensor.mean(dim=1)
        pooled_var = pooled.var(dim=0).mean().item()
        flat = tensor.reshape(Ns, -1)
        i1, i2 = torch.randint(0, Ns, (2000,)), torch.randint(0, Ns, (2000,))
        mk = i1 != i2
        cos = F.cosine_similarity(flat[i1[mk]], flat[i2[mk]], dim=-1)
        pc = pooled - pooled.mean(0)
        _, S, _ = torch.linalg.svd(pc, full_matrices=False)
        ev = (S**2) / (S**2).sum()
        cum = ev.cumsum(0)
        n90 = (cum < 0.90).sum().item() + 1
        tag = ("⚠️ NEAR-IDENTICAL" if cos.mean() > 0.95
               else ("~ WEAK" if cos.mean() > 0.85 else "✓ OK"))
        print(f"\n  [{name}]  var={var_avg:.6f}  pooled_var={pooled_var:.6f}")
        print(f"    cos_sim: mean={cos.mean():.4f} std={cos.std():.4f} "
              f"min={cos.min():.4f} → {tag}")
        print(f"    PCA dims for 90%: {n90}  "
              f"top-3 sv: {[f'{v:.3f}' for v in S[:3].tolist()]}")

    tc_s = torch.stack(tool_cs)
    print(f"\n  Tool centroid std/axis: {[f'{v:.4f}' for v in tc_s.std(0).tolist()]}")
    if dps:
        dp = torch.stack(dps)
        print(f"  Delta pose std/dim:    {[f'{v:.3f}' for v in dp.std(0).tolist()]}")
    print(f"{'='*60}\n")
    m.train()


def quick_feature_stats(model, device, ds, n_samples=50):
    """Lightweight diagnostic: cosine sim + pooled var on a subsample."""
    m = model.module if hasattr(model, 'module') else model
    was_training = m.training
    m.eval()
    conds = []

    with torch.no_grad():
        for i in range(min(n_samples, len(ds))):
            s = ds[i]
            if s['tool_pc_init'] is None:
                continue
            tpc = s['tool_pc_init'].unsqueeze(0).to(device)
            opc = s['obj_pc'].unsqueeze(0).to(device)
            enc = m.encoder.encode(tpc, opc)
            cond = torch.cat([enc.tool_tokens, enc.obj_tokens], dim=1)
            tc = tpc.mean(dim=1)
            oc = opc.mean(dim=1)
            pb = m.cond_pos_proj(torch.cat([tc, oc], -1))
            conds.append((cond + pb.unsqueeze(1)).squeeze(0).cpu())

    t = torch.stack(conds)
    Ns = t.shape[0]
    pooled_var = t.mean(dim=1).var(dim=0).mean().item()
    flat = t.reshape(Ns, -1)
    i1, i2 = torch.randint(0, Ns, (500,)), torch.randint(0, Ns, (500,))
    mk = i1 != i2
    cos = F.cosine_similarity(flat[i1[mk]], flat[i2[mk]], dim=-1).mean().item()

    if was_training:
        m.train()
    return cos, pooled_var


# --------------------------------------------------------------------------- #
# Data Stats Diagnostic
# --------------------------------------------------------------------------- #

def run_data_stats(ds, device):
    """Print data statistics to diagnose discriminability."""
    init_poses, delta_poses, centroids = [], [], []
    N = len(ds)

    for i in range(N):
        s = ds[i]
        if s['delta_pose'] is not None:
            delta_poses.append(s['delta_pose'])
        if s.get('init_pose') is not None:
            init_poses.append(s['init_pose'])
        if s['tool_pc_init'] is not None:
            centroids.append(s['tool_pc_init'].mean(dim=0))

    print(f"\n{'='*60}")
    print(f"DATA STATISTICS ({N} samples)")
    print(f"{'='*60}")

    if delta_poses:
        dp = torch.stack(delta_poses)
        print(f"\n  delta_pose ({dp.shape}):")
        print(f"    mean: {[f'{v:.4f}' for v in dp.mean(0).tolist()]}")
        print(f"    std:  {[f'{v:.4f}' for v in dp.std(0).tolist()]}")

    if init_poses:
        ip = torch.stack(init_poses)
        print(f"\n  init_pose ({ip.shape}):")
        print(f"    mean: {[f'{v:.4f}' for v in ip.mean(0).tolist()]}")
        print(f"    std:  {[f'{v:.4f}' for v in ip.std(0).tolist()]}")
        dists = torch.cdist(ip.unsqueeze(0), ip.unsqueeze(0)).squeeze(0)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        pw = dists[mask]
        print(f"    pairwise L2: mean={pw.mean():.4f} min={pw.min():.6f}")

    if centroids:
        ct = torch.stack(centroids)
        print(f"\n  tool_centroid ({ct.shape}):")
        print(f"    std:  {[f'{v:.4f}' for v in ct.std(0).tolist()]}")

    print(f"{'='*60}\n")


# --------------------------------------------------------------------------- #
# Full-file overfit loop
# --------------------------------------------------------------------------- #

def run_full_overfit(model, device, cfg, args):
    files = collect_pt_files(cfg.data_dir)
    ds = ContactDataset(files[:1], augment=False)
    print(f"\nFull overfit: {len(ds)} configs from 1 file")

    m = model.module if hasattr(model, 'module') else model
    n_obs = 2 * m.encoder.num_patches  # 32
    D = cfg.encoder_channel  # 128

    # ---- Run diagnostic if requested ----
    if args.diag:
        run_data_stats(ds, device)
        run_variance_diagnostic(model, device, ds)

    # ---- Choose noise predictor ----
    if args.cond_mlp:
        predictor = MLPNoisePredictor(
            pose_dim=9, cond_dim=D, hidden=cfg.n_emb, n_layers=cfg.n_layer,
        ).to(device)
        pred_params = list(predictor.parameters())
        pred_name = "MLP"
    else:
        predictor = m.transformer
        pred_params = list(m.transformer.parameters())
        pred_name = "Transformer"

    # ---- Choose condition source ----
    cond_proj = None
    cond_emb = None
    cond_centroid_proj = None
    cond_init_pose_proj = None

    if args.cond_index:
        cond_emb = nn.Embedding(len(ds), D).to(device)
        cond_params = list(cond_emb.parameters())
        cond_name = f"sample_index Emb({len(ds)},{D})"
    elif args.cond_pose:
        cond_proj = nn.Linear(9, D).to(device)
        cond_params = list(cond_proj.parameters())
        cond_name = "delta_pose (answer)"
    elif args.cond_centroid:
        cond_centroid_proj = nn.Linear(6, D).to(device)
        cond_params = list(cond_centroid_proj.parameters())
        cond_name = "centroids only (6D)"
    elif args.cond_init_pose:
        cond_init_pose_proj = nn.Sequential(
            nn.Linear(9, D), nn.GELU(), nn.Linear(D, D),
        ).to(device)
        cond_params = list(cond_init_pose_proj.parameters())
        cond_name = "init_pose (9D, MLP)"
    else:
        cond_params = list(m.cond_pos_proj.parameters())
        if args.detach_enc:
            cond_name = "encoder(detached) + pos_bias"
        else:
            cond_params += list(m.encoder.parameters())
            cond_name = "encoder + pos_bias"

    # Per-token center re-injection
    center_pos_proj = None
    if args.inject_centers and not (args.cond_index or args.cond_pose or args.cond_centroid):
        center_pos_proj = nn.Linear(3, D).to(device)
        cond_params += list(center_pos_proj.parameters())
        cond_name += " + center_reinject"

    # Auxiliary regression head
    reg_head = None
    if args.aux_reg and not (args.cond_index or args.cond_pose):
        reg_head = nn.Sequential(
            nn.Linear(D, D), nn.GELU(),
            nn.Linear(D, 9),
        ).to(device)
        cond_params += list(reg_head.parameters())
        cond_name += f" + aux_reg(w={args.aux_weight})"

    params = pred_params + cond_params
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    print(f"  COND: {cond_name}  HEAD: {pred_name}")
    print(f"  lr={args.lr:.1e}, epochs={args.epochs}, warmup={args.warmup_reg}")
    print(f"  trainable params: {sum(p.numel() for p in params):,}")
    print(f"{'='*60}")

    # ---- DataLoader ----
    class IndexedDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            sample = self.ds[idx]
            sample["idx"] = idx
            return sample

    if args.cond_index:
        loader = DataLoader(IndexedDataset(ds), batch_size=min(64, len(ds)),
                            shuffle=True, num_workers=0)
    else:
        loader = DataLoader(ds, batch_size=min(64, len(ds)),
                            shuffle=True, num_workers=0)

    # Track encoder feature evolution
    use_encoder = not (args.cond_index or args.cond_pose or args.cond_centroid or args.cond_init_pose)
    diag_freq = 25

    if args.diag and use_encoder:
        cos0, var0 = quick_feature_stats(model, device, ds)
        print(f"  [DIAG epoch=0] cos_sim={cos0:.4f}  pooled_var={var0:.6f}")

    # ---- Training ----
    model.train()
    best_diff = float('inf')
    best_aux = float('inf')

    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_diff, epoch_aux, n_batches = 0.0, 0.0, 0

        for batch in loader:
            obj_pc = batch["obj_pc"].to(device)
            tool_pc_init = batch["tool_pc_init"].to(device) if batch["tool_pc_init"] is not None else None
            delta_pose = batch["delta_pose"].to(device) if batch["delta_pose"] is not None else None
            init_pose = batch["init_pose"].to(device) if batch.get("init_pose") is not None else None
            if delta_pose is None or tool_pc_init is None:
                continue

            B = delta_pose.shape[0]
            clean_data = delta_pose.unsqueeze(1)

            # ---- Build condition ----
            if args.cond_index:
                idx = batch["idx"].to(device)
                cond = cond_emb(idx).unsqueeze(1).expand(-1, n_obs, -1)
            elif args.cond_pose:
                cond = cond_proj(delta_pose).unsqueeze(1).expand(-1, n_obs, -1)
            elif args.cond_centroid:
                tc = tool_pc_init.mean(dim=1)
                oc = obj_pc.mean(dim=1)
                cond = cond_centroid_proj(torch.cat([tc, oc], -1)).unsqueeze(1).expand(-1, n_obs, -1)
            elif args.cond_init_pose:
                cond = cond_init_pose_proj(init_pose).unsqueeze(1).expand(-1, n_obs, -1)
            else:
                enc_result = m.encoder.encode(tool_pc_init, obj_pc)
                cond = torch.cat([enc_result.tool_tokens, enc_result.obj_tokens], dim=1)
                if args.detach_enc:
                    cond = cond.detach()
                if center_pos_proj is not None:
                    centers = compute_patch_centers(tool_pc_init, obj_pc, m.encoder.num_patches)
                    cond = cond + center_pos_proj(centers)
                tc = tool_pc_init.mean(dim=1)
                oc = obj_pc.mean(dim=1)
                pos_bias = m.cond_pos_proj(torch.cat([tc, oc], -1))
                cond = cond + pos_bias.unsqueeze(1)

            # ---- Flow matching forward ----
            in_warmup = (args.warmup_reg > 0 and epoch < args.warmup_reg)
            if not args.reg_only and not in_warmup:
                eps = torch.randn_like(clean_data)
                # Logit-normal time sampling (SD3 recipe)
                sigma_min = 1e-4
                u = torch.randn(B, device=device) * 0.5
                t = torch.sigmoid(u)
                t = t * (1.0 - sigma_min) + sigma_min
                t_expand = t[:, None, None]
                x_t = (1.0 - t_expand) * clean_data + t_expand * eps
                v_target = eps - clean_data
                v_pred = predictor(sample=x_t, timestep=t, cond=cond)
                diff_loss = F.mse_loss(v_pred, v_target)
            else:
                diff_loss = torch.tensor(0.0, device=device)

            # Auxiliary regression
            aux_loss_val = 0.0
            if reg_head is not None:
                pooled = cond.mean(dim=1)
                dp_pred = reg_head(pooled)
                aux_loss = F.mse_loss(dp_pred, delta_pose)
                aux_loss_val = aux_loss.item()
                if args.reg_only:
                    loss = aux_loss
                else:
                    loss = diff_loss + args.aux_weight * aux_loss
            else:
                loss = diff_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_diff += diff_loss.item()
            epoch_aux += aux_loss_val
            n_batches += 1

        avg_diff = epoch_diff / max(n_batches, 1)
        avg_aux = epoch_aux / max(n_batches, 1)
        best_diff = min(best_diff, avg_diff)
        if avg_aux > 0:
            best_aux = min(best_aux, avg_aux)
        dt = time.time() - t0

        phase = ("[WARMUP]" if (args.warmup_reg > 0 and epoch < args.warmup_reg)
                 else "[JOINT] " if args.warmup_reg > 0 else "")
        do_log = (epoch % 5 == 0 or epoch < 10 or epoch == args.epochs - 1
                  or (args.warmup_reg > 0 and epoch == args.warmup_reg))
        if do_log:
            aux_str = f"  aux={avg_aux:.6f}(best={best_aux:.6f})" if reg_head is not None else ""
            print(f"  {phase}Epoch {epoch:4d}/{args.epochs}: "
                  f"diff={avg_diff:.6f}  best={best_diff:.6f}{aux_str}  t={dt:.1f}s")

        # Periodic feature discriminability check
        if args.diag and use_encoder and (epoch % diag_freq == 0 or epoch == args.epochs - 1) and epoch > 0:
            cos_e, var_e = quick_feature_stats(model, device, ds)
            print(f"  [DIAG epoch={epoch}] cos_sim={cos_e:.4f}  pooled_var={var_e:.6f}")
            model.train()

    print(f"\n{'='*60}")
    print(f"FULL OVERFIT [{cond_name} + {pred_name}]: best_diff={best_diff:.6f}")
    if best_diff < 0.05:
        print("  ✓ PASS")
    elif best_diff < 0.2:
        print("  ~ PARTIAL — needs more epochs or tuning")
    else:
        print("  ✗ FAIL")
    print(f"{'='*60}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument("--full", action="store_true", help="Full-file overfit")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--warmup-reg", type=int, default=0,
                        help="Warmup epochs: regression-only before diffusion")

    # Condition modes (mutually exclusive)
    parser.add_argument("--cond-pose", action="store_true",
                        help="Use delta_pose (answer) as condition")
    parser.add_argument("--cond-index", action="store_true",
                        help="Use sample index as condition")
    parser.add_argument("--cond-centroid", action="store_true",
                        help="Use centroids only (6D), no encoder")
    parser.add_argument("--cond-init-pose", action="store_true",
                        help="Use init_pose (9D) as condition")

    # Head mode
    parser.add_argument("--cond-mlp", action="store_true",
                        help="Use MLP noise predictor instead of transformer")

    # Diagnostics
    parser.add_argument("--detach-enc", action="store_true",
                        help="Stop gradient through encoder features")
    parser.add_argument("--inject-centers", action="store_true",
                        help="Re-inject per-token FPS patch centers")
    parser.add_argument("--diag", action="store_true",
                        help="Run encoder variance diagnostic")

    # Auxiliary regression
    parser.add_argument("--aux-reg", action="store_true",
                        help="Add auxiliary regression loss")
    parser.add_argument("--aux-weight", type=float, default=1.0,
                        help="Weight for auxiliary regression loss")
    parser.add_argument("--reg-only", action="store_true",
                        help="Regression only, no diffusion")

    # Diffusion
    parser.add_argument("--diff-steps", type=int, default=0,
                        help="Override diffusion timesteps (0=use model default)")

    args = parser.parse_args()

    # Auto-enable aux_reg when needed
    if args.reg_only or args.warmup_reg > 0:
        args.aux_reg = True

    cfg = TrainConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("OVERFIT TEST: JointModel Diffusion Diagnostic")
    print("=" * 60)

    model = JointModel(
        head_mode=cfg.head_mode, patch_agg=cfg.patch_agg, head_hidden=cfg.head_hidden,
        num_pts=cfg.num_pts, patch_size=cfg.patch_size, encoder_channel=cfg.encoder_channel,
        vit_depth=cfg.vit_depth, vit_heads=cfg.vit_heads, freeze_encoder=cfg.freeze_encoder,
        n_layer=cfg.n_layer, n_head=cfg.n_head, n_emb=cfg.n_emb,
        p_drop_emb=cfg.p_drop_emb, p_drop_attn=cfg.p_drop_attn,
        sdf_weight=cfg.sdf_weight, diffusion_weight=cfg.diffusion_weight,
    ).to(device)

    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {params_total:,}")

    # --diff-steps is not applicable for flow matching (continuous t)
    if args.diff_steps > 0:
        print(f"  Note: --diff-steps ignored (flow matching uses continuous t)")

    if args.full:
        run_full_overfit(model, device, cfg, args)
    else:
        print("Use --full for full-file overfit test")


if __name__ == "__main__":
    main()