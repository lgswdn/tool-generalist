#!/usr/bin/env python3
"""debug_overfit.py — Overfit diagnostics for diffusion convergence.
Condition modes:
  (default)        Encoder features + centroid bias
  --cond-pose      Delta_pose (answer) as condition
  --cond-index     Unique learnable embedding per sample
  --cond-centroid  Centroids only (6D), no encoder
Head modes:
  (default)        TransformerForDiffusion (cross-attention)
  --cond-mlp       MLP noise predictor (bypasses cross-attention)
Diagnostics:
  --diag           Encoder variance diagnostic before training
  --detach-enc     Stop gradient through encoder features
Usage:
    python debug_overfit.py --full                             # encoder + transformer
    python debug_overfit.py --full --cond-mlp                  # encoder + MLP head
    python debug_overfit.py --full --detach-enc                # frozen encoder + transformer
    python debug_overfit.py --full --cond-centroid             # centroids only
    python debug_overfit.py --full --cond-centroid --cond-mlp  # centroids + MLP
    python debug_overfit.py --full --diag                      # diagnostic then train
"""
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import TrainConfig
from model import JointModel
from dataset import ContactDataset, collect_pt_files
from pytorch3d.ops import sample_farthest_points
def compute_patch_centers(tool_pc, obj_pc, num_patches=16):
    """Compute FPS patch centers for tool and object clouds.
    Returns (B, 2*P, 3) — same ordering as encoder tokens."""
    tool_ctrs, _ = sample_farthest_points(tool_pc.contiguous(), K=num_patches, random_start_point=False)
    obj_ctrs, _ = sample_farthest_points(obj_pc.contiguous(), K=num_patches, random_start_point=False)
    return torch.cat([tool_ctrs, obj_ctrs], dim=1)  # (B, 2P, 3)
# ---- MLP Noise Predictor (bypasses cross-attention bottleneck) ----
class SinTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)
class MLPNoisePredictor(nn.Module):
    """Direct MLP: [noisy_pose, time, pooled_cond] → noise.
    Bypasses cross-attention for horizon=1 where it collapses."""
    def __init__(self, pose_dim=9, cond_dim=128, hidden=256, n_layers=4):
        super().__init__()
        self.time_emb = SinTimeEmb(hidden)
        self.time_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, hidden), nn.GELU())
        self.input_proj = nn.Sequential(nn.Linear(pose_dim, hidden), nn.GELU())
        layers = []
        for i in range(n_layers):
            layers.extend([
                nn.Linear(hidden * 3 if i == 0 else hidden, hidden),
                nn.LayerNorm(hidden), nn.GELU(),
            ])
        layers.append(nn.Linear(hidden, pose_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, sample, timestep, cond):
        B = sample.shape[0]
        x = sample.squeeze(1)  # (B, 9)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
        t = self.time_proj(self.time_emb(timestep.expand(B)))
        c = self.cond_proj(cond.mean(dim=1))  # mean-pool tokens
        h = torch.cat([self.input_proj(x), t, c], dim=-1)
        return self.mlp(h).unsqueeze(1)  # (B, 1, 9)
# ---- Encoder Variance Diagnostic ----
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
            if s['tool_pc_init'] is None: continue
            tpc = s['tool_pc_init'].unsqueeze(0).to(device)
            opc = s['obj_pc'].unsqueeze(0).to(device)
            enc = m.encoder.encode(tpc, opc)
            cond = torch.cat([enc.tool_tokens, enc.obj_tokens], dim=1)
            tc = tpc.mean(dim=1); oc = opc.mean(dim=1)
            pb = m.cond_pos_proj(torch.cat([tc, oc], -1))
            conds_raw.append(cond.squeeze(0).cpu())
            conds_biased.append((cond + pb.unsqueeze(1)).squeeze(0).cpu())
            tool_cs.append(tc.squeeze(0).cpu())
            if s['delta_pose'] is not None: dps.append(s['delta_pose'])
    for name, tensor in [("RAW", torch.stack(conds_raw)), ("WITH POS BIAS", torch.stack(conds_biased))]:
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
        ev = (S**2) / (S**2).sum(); cum = ev.cumsum(0)
        n90 = (cum < 0.90).sum().item() + 1
        tag = "⚠️ NEAR-IDENTICAL" if cos.mean()>0.95 else ("~ WEAK" if cos.mean()>0.85 else "✓ OK")
        print(f"\n  [{name}]  var={var_avg:.6f}  pooled_var={pooled_var:.6f}")
        print(f"    cos_sim: mean={cos.mean():.4f} std={cos.std():.4f} min={cos.min():.4f} → {tag}")
        print(f"    PCA dims for 90%: {n90}  top-3 sv: {[f'{v:.3f}' for v in S[:3].tolist()]}")
    tc_s = torch.stack(tool_cs)
    print(f"\n  Tool centroid std/axis: {[f'{v:.4f}' for v in tc_s.std(0).tolist()]}")
    if dps:
        dp = torch.stack(dps)
        print(f"  Delta pose std/dim:    {[f'{v:.3f}' for v in dp.std(0).tolist()]}")
    print(f"{'='*60}\n")
    m.train()
def quick_feature_stats(model, device, ds, n_samples=50):
    """Lightweight diagnostic: cosine sim + pooled var on a subsample.
    Fast enough to call every few epochs during training."""
    m = model.module if hasattr(model, 'module') else model
    was_training = m.training
    m.eval()
    conds = []
    with torch.no_grad():
        for i in range(min(n_samples, len(ds))):
            s = ds[i]
            if s['tool_pc_init'] is None: continue
            tpc = s['tool_pc_init'].unsqueeze(0).to(device)
            opc = s['obj_pc'].unsqueeze(0).to(device)
            enc = m.encoder.encode(tpc, opc)
            cond = torch.cat([enc.tool_tokens, enc.obj_tokens], dim=1)
            tc = tpc.mean(dim=1); oc = opc.mean(dim=1)
            pb = m.cond_pos_proj(torch.cat([tc, oc], -1))
            conds.append((cond + pb.unsqueeze(1)).squeeze(0).cpu())
    t = torch.stack(conds)  # (N, 32, 128)
    Ns = t.shape[0]
    pooled_var = t.mean(dim=1).var(dim=0).mean().item()
    flat = t.reshape(Ns, -1)
    i1, i2 = torch.randint(0, Ns, (500,)), torch.randint(0, Ns, (500,))
    mk = i1 != i2
    cos = F.cosine_similarity(flat[i1[mk]], flat[i2[mk]], dim=-1).mean().item()
    if was_training: m.train()
    return cos, pooled_var
# ---- Phase-based single-sample test (B=1) ----
def run_phase(
    model, optimizer, device,
    tool_pc, obj_pc, tool_sdf_gt, obj_sdf_gt,
    tool_pc_init, delta_pose_gt,
    phase: int,
    steps: int = 2000,
    diffusion_only: bool = False,
):
    phase_name = {
        1: "Fixed noise + Fixed timestep",
        2: "Random noise + Fixed timestep",
        3: "Random noise + Random timestep (full DDPM)",
    }[phase]
    print(f"\n{'='*60}")
    print(f"PHASE {phase}: {phase_name}  (steps={steps})")
    print(f"{'='*60}")
    B = delta_pose_gt.shape[0]
    clean_data = delta_pose_gt.unsqueeze(1)
    fixed_noise = torch.randn_like(clean_data)
    fixed_timestep = torch.tensor([50], device=device, dtype=torch.long)
    m = model.module if hasattr(model, 'module') else model
    model.train()
    best_diff_loss = float('inf')
    for step in range(steps):
        optimizer.zero_grad()
        if diffusion_only:
            enc_result = m.encoder.encode(tool_pc_init, obj_pc)
            cond = torch.cat([enc_result.tool_tokens, enc_result.obj_tokens], dim=1)
            if phase == 1:
                noise, timesteps = fixed_noise, fixed_timestep
            elif phase == 2:
                noise, timesteps = torch.randn_like(clean_data), fixed_timestep
            else:
                noise = torch.randn_like(clean_data)
                timesteps = torch.randint(0, m.noise_scheduler.config.num_train_timesteps,
                                          (B,), device=device, dtype=torch.long)
            noisy_data = m.noise_scheduler.add_noise(clean_data, noise, timesteps)
            noise_pred = m.transformer(sample=noisy_data, timestep=timesteps, cond=cond)
            loss = F.mse_loss(noise_pred, noise)
            diff_loss_val = loss.item()
            sdf_loss_val = 0.0
        else:
            loss, metrics = m.loss(tool_pc, obj_pc, tool_sdf_gt, obj_sdf_gt,
                                   tool_pc_init=tool_pc_init, delta_pose_gt=delta_pose_gt)
            diff_loss_val = metrics.get('diffusion_loss', 0.0)
            sdf_loss_val = metrics['tool_sdf_loss'] + metrics['obj_sdf_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        best_diff_loss = min(best_diff_loss, diff_loss_val)
        if step % 100 == 0 or step < 20 or step == steps - 1:
            print(f"  Step {step:5d}: diff={diff_loss_val:.6f}  sdf={sdf_loss_val:.6f}  best={best_diff_loss:.6f}")
    status = "✓ PASS" if best_diff_loss < 0.01 else ("~ PARTIAL" if best_diff_loss < 0.1 else "✗ FAIL")
    print(f"  {status} (best={best_diff_loss:.6f})")
    return best_diff_loss < 0.01, best_diff_loss
# ---- Data Stats Diagnostic ----
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
        print(f"    min:  {[f'{v:.4f}' for v in dp.min(0).values.tolist()]}")
        print(f"    max:  {[f'{v:.4f}' for v in dp.max(0).values.tolist()]}")
    if init_poses:
        ip = torch.stack(init_poses)
        print(f"\n  init_pose ({ip.shape}):")
        print(f"    mean: {[f'{v:.4f}' for v in ip.mean(0).tolist()]}")
        print(f"    std:  {[f'{v:.4f}' for v in ip.std(0).tolist()]}")
        print(f"    min:  {[f'{v:.4f}' for v in ip.min(0).values.tolist()]}")
        print(f"    max:  {[f'{v:.4f}' for v in ip.max(0).values.tolist()]}")
        # Pairwise L2 distances
        dists = torch.cdist(ip.unsqueeze(0), ip.unsqueeze(0)).squeeze(0)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        pw = dists[mask]
        print(f"    pairwise L2: mean={pw.mean():.6f} std={pw.std():.6f} min={pw.min():.6f} max={pw.max():.6f}")
        n_dup = (pw < 1e-4).sum().item()
        print(f"    near-duplicates (<1e-4): {n_dup}/{pw.numel()}")
    if centroids:
        ct = torch.stack(centroids)
        print(f"\n  tool_centroid ({ct.shape}):")
        print(f"    mean: {[f'{v:.4f}' for v in ct.mean(0).tolist()]}")
        print(f"    std:  {[f'{v:.4f}' for v in ct.std(0).tolist()]}")
        dists_c = torch.cdist(ct.unsqueeze(0), ct.unsqueeze(0)).squeeze(0)
        mask_c = torch.triu(torch.ones(len(centroids), len(centroids), dtype=torch.bool), diagonal=1)
        pw_c = dists_c[mask_c]
        print(f"    pairwise L2: mean={pw_c.mean():.6f} std={pw_c.std():.6f} min={pw_c.min():.6f}")
    # Check: how many unique init_poses map to different delta_poses?
    if init_poses and delta_poses:
        ip = torch.stack(init_poses)
        dp = torch.stack(delta_poses)
        # Cluster init_poses by similarity
        dists = torch.cdist(ip.unsqueeze(0), ip.unsqueeze(0)).squeeze(0)
        close_pairs = (dists < 1e-3) & torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        if close_pairs.any():
            idxs = close_pairs.nonzero()
            print(f"\n  ⚠️  {idxs.shape[0]} pairs have near-identical init_poses (L2 < 1e-3):")
            for k in range(min(5, idxs.shape[0])):
                i, j = idxs[k]
                dp_dist = (dp[i] - dp[j]).norm().item()
                print(f"    samples {i},{j}: init_dist={dists[i,j]:.6f}, delta_dist={dp_dist:.4f}")
        else:
            print(f"\n  ✓ All init_poses are unique (min pairwise dist > 1e-3)")
    print(f"{'='*60}\n")
# ---- Full-file overfit (all configs from one .pt file) ----
# ---- Sanity: standalone MLP regression (no model infrastructure) ----
def run_sanity_regression(device, cfg, args):
    """Dead-simple MLP: init_pose(9D) → delta_pose(9D). No encoder, no transformer.
    If this can't memorize 424 samples, there's a data issue."""
    from torch.utils.data import DataLoader
    files = collect_pt_files(cfg.data_dir)
    ds = ContactDataset(files[:1], augment=False)
    print(f"\n{'='*60}")
    print(f"SANITY REGRESSION: MLP(9→9), {len(ds)} samples")
    print(f"{'='*60}")
    # Collect all init_pose, delta_pose pairs
    all_ip, all_dp = [], []
    for i in range(len(ds)):
        s = ds[i]
        if s.get('init_pose') is not None and s['delta_pose'] is not None:
            all_ip.append(s['init_pose'])
            all_dp.append(s['delta_pose'])
    ip = torch.stack(all_ip).to(device)  # (N, 9)
    dp = torch.stack(all_dp).to(device)  # (N, 9)
    N = ip.shape[0]
    print(f"  Valid pairs: {N}")
    print(f"  init_pose range: [{ip.min():.3f}, {ip.max():.3f}]")
    print(f"  delta_pose range: [{dp.min():.3f}, {dp.max():.3f}]")
    # Standalone MLP — no expand, no pool, no transformer
    mlp = nn.Sequential(
        nn.Linear(9, 256), nn.GELU(),
        nn.Linear(256, 256), nn.GELU(),
        nn.Linear(256, 256), nn.GELU(),
        nn.Linear(256, 9),
    ).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    print(f"  MLP params: {sum(p.numel() for p in mlp.parameters()):,}")
    print(f"  lr=1e-3, full-batch, 2000 steps")
    best = float('inf')
    for step in range(2000):
        pred = mlp(ip)
        loss = F.mse_loss(pred, dp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        best = min(best, loss.item())
        if step % 100 == 0 or step < 20 or step == 1999:
            print(f"  Step {step:5d}: loss={loss.item():.6f}  best={best:.6f}")
    status = "✓ PASS" if best < 0.01 else ("~ PARTIAL" if best < 0.1 else "✗ FAIL")
    print(f"\n  {status} (best={best:.6f})")
    print(f"{'='*60}")
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
    # ---- Choose noise predictor (transformer vs MLP) ----
    if args.cond_mlp:
        predictor = MLPNoisePredictor(pose_dim=9, cond_dim=D,
                                      hidden=cfg.n_emb, n_layers=cfg.n_layer).to(device)
        pred_params = list(predictor.parameters())
        pred_name = "MLP"
        print(f"  HEAD: MLPNoisePredictor (hidden={cfg.n_emb}, layers={cfg.n_layer})")
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
            nn.Linear(9, D), nn.GELU(), nn.Linear(D, D)
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
    # Per-token center re-injection (--inject-centers)
    center_pos_proj = None
    if args.inject_centers and not (args.cond_index or args.cond_pose or args.cond_centroid):
        center_pos_proj = nn.Linear(3, D).to(device)
        cond_params += list(center_pos_proj.parameters())
        cond_name += " + center_reinject"
    # Auxiliary regression head: pooled_cond → delta_pose (direct supervision)
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
    print(f"  lr={args.lr:.1e}, epochs={args.epochs}")
    print(f"  trainable params: {sum(p.numel() for p in params):,}")
    print(f"{'='*60}")
    # ---- IndexedDataset for --cond-index ----
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
    # Track encoder feature evolution during training
    use_encoder = not (args.cond_index or args.cond_pose or args.cond_centroid)
    diag_freq = 25  # run diagnostic every N epochs
    if args.diag and use_encoder:
        cos0, var0 = quick_feature_stats(model, device, ds)
        print(f"  [DIAG epoch=0] cos_sim={cos0:.4f}  pooled_var={var0:.6f}  (random init)")
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
                # Per-token center re-injection: each token gets its own 3D position
                if center_pos_proj is not None:
                    centers = compute_patch_centers(tool_pc_init, obj_pc, m.encoder.num_patches)  # (B, 32, 3)
                    cond = cond + center_pos_proj(centers)  # per-token position, not broadcast
                tc = tool_pc_init.mean(dim=1)
                oc = obj_pc.mean(dim=1)
                pos_bias = m.cond_pos_proj(torch.cat([tc, oc], -1))
                cond = cond + pos_bias.unsqueeze(1)
            # ---- Diffusion forward ----
            in_warmup = (args.warmup_reg > 0 and epoch < args.warmup_reg)
            if not args.reg_only and not in_warmup:
                noise = torch.randn_like(clean_data)
                timesteps = torch.randint(0, m.noise_scheduler.config.num_train_timesteps,
                                          (B,), device=device, dtype=torch.long)
                noisy_data = m.noise_scheduler.add_noise(clean_data, noise, timesteps)
                noise_pred = predictor(sample=noisy_data, timestep=timesteps, cond=cond)
                diff_loss = F.mse_loss(noise_pred, noise)
            else:
                diff_loss = torch.tensor(0.0, device=device)
            # Auxiliary regression: pooled condition → delta_pose
            aux_loss_val = 0.0
            if reg_head is not None:
                pooled = cond.mean(dim=1)          # (B, D)
                dp_pred = reg_head(pooled)          # (B, 9)
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
        if avg_aux > 0: best_aux = min(best_aux, avg_aux)
        dt = time.time() - t0
        phase = "[WARMUP]" if (args.warmup_reg > 0 and epoch < args.warmup_reg) else "[JOINT] " if args.warmup_reg > 0 else ""
        do_log = epoch % 5 == 0 or epoch < 10 or epoch == args.epochs - 1 or (args.warmup_reg > 0 and epoch == args.warmup_reg)
        if do_log:
            aux_str = f"  aux={avg_aux:.6f}(best={best_aux:.6f})" if reg_head is not None else ""
            print(f"  {phase}Epoch {epoch:4d}/{args.epochs}: diff={avg_diff:.6f}  best={best_diff:.6f}{aux_str}  t={dt:.1f}s")
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0, help="1,2,3 or 0=all")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=3000, help="Steps (B=1 mode)")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs (--full mode)")
    parser.add_argument("--full", action="store_true", help="Full-file overfit")
    parser.add_argument("--cond-pose", action="store_true",
                        help="Use delta_pose (answer) as condition")
    parser.add_argument("--cond-index", action="store_true",
                        help="Use sample index as condition")
    parser.add_argument("--cond-centroid", action="store_true",
                        help="Use centroids only (6D), no encoder")
    parser.add_argument("--cond-init-pose", action="store_true",
                        help="Use init_pose (9D: translation + 6D rotation) as condition")
    parser.add_argument("--cond-mlp", action="store_true",
                        help="Use MLP noise predictor instead of transformer")
    parser.add_argument("--detach-enc", action="store_true",
                        help="Stop gradient through encoder features")
    parser.add_argument("--inject-centers", action="store_true",
                        help="Re-inject per-token FPS patch centers after ViT (position signal)")
    parser.add_argument("--diag", action="store_true",
                        help="Run encoder variance diagnostic before training")
    parser.add_argument("--aux-reg", action="store_true",
                        help="Add auxiliary regression loss on pooled condition")
    parser.add_argument("--aux-weight", type=float, default=1.0,
                        help="Weight for auxiliary regression loss")
    parser.add_argument("--reg-only", action="store_true",
                        help="Train ONLY 1 head, no diffusion (tests if cond can predict delta_pose)")
    parser.add_argument("--diffusion-only", action="store_true",
                        help="Skip SDF loss")
    parser.add_argument("--sanity", action="store_true",
                        help="Standalone MLP regression: init_pose→delta_pose (no model)")
    parser.add_argument("--warmup-reg", type=int, default=0,
                        help="Warmup epochs: regression-only before diffusion starts")
    parser.add_argument("--diff-steps", type=int, default=0,
                        help="Override diffusion timesteps (default: use model's 100)")
    args = parser.parse_args()
    if args.reg_only:
        args.aux_reg = True  # reg-only needs the regression head
    if args.warmup_reg > 0:
        args.aux_reg = True  # warmup needs the regression head
    cfg = TrainConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Sanity test runs without model
    if args.sanity:
        run_sanity_regression(device, cfg, args)
        return
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
    params_transformer = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
    print(f"\nModel params: {params_total:,} (transformer: {params_transformer:,})")
    # Override diffusion timesteps if requested
    if args.diff_steps > 0:
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        model.noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.diff_steps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
        )
        print(f"  Overrode diffusion timesteps: {args.diff_steps}")
    if args.full:
        run_full_overfit(model, device, cfg, args)
    else:
        # B=1 phase-based test
        files = collect_pt_files(cfg.data_dir)
        ds = ContactDataset(files[:1], augment=False)
        sample = ds[0]
        if sample['delta_pose'] is None:
            print("\nERROR: delta_pose is None!")
            return
        tool_pc = sample['tool_pc'].unsqueeze(0).to(device)
        obj_pc = sample['obj_pc'].unsqueeze(0).to(device)
        tool_sdf_gt = sample['tool_pts_sdf'].unsqueeze(0).to(device)
        obj_sdf_gt = sample['obj_pts_sdf'].unsqueeze(0).to(device)
        tool_pc_init = sample['tool_pc_init'].unsqueeze(0).to(device) if sample['tool_pc_init'] is not None else None
        delta_pose_gt = sample['delta_pose'].unsqueeze(0).to(device)
        phases = [args.phase] if args.phase > 0 else [1, 2, 3]
        for phase in phases:
            if args.diffusion_only:
                diff_params = list(model.encoder.parameters()) + list(model.transformer.parameters())
                optimizer = torch.optim.AdamW(diff_params, lr=args.lr, weight_decay=1e-4)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            passed, _ = run_phase(model, optimizer, device,
                                  tool_pc, obj_pc, tool_sdf_gt, obj_sdf_gt,
                                  tool_pc_init, delta_pose_gt,
                                  phase=phase, steps=args.steps, diffusion_only=args.diffusion_only)
            if not passed and phase < 3:
                print(f"  ⚠ Phase {phase} failed — skipping harder phases")
                break
if __name__ == "__main__":
    main()