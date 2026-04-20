#!/usr/bin/env python3
"""debug_overfit.py — Overfit on ONE single sample to diagnose issues.

Tests JointModel with 2-step diffusion (essentially 1-step).
If pose fails, the diffusion architecture is broken.
If pose works, can increase steps gradually.

Usage:
    python debug_overfit.py
"""

import torch
import torch.nn.functional as F
from config import TrainConfig
from model import JointModel
from dataset import ContactDataset, collect_pt_files


def main():
    # Load config
    cfg = TrainConfig()
    cfg.epochs = 1000

    print("=" * 60)
    print("OVERFIT TEST: JointModel (SDF + 2-step diffusion)")
    print("=" * 60)

    # Load ONE sample from dataset
    data_dir = "/mnt/project/world_model/tool_generalist/teardrop_contact/000_asym_teardrop_contour_scraper_var_000/"
    files = collect_pt_files(data_dir)
    ds = ContactDataset(files[:1], augment=False)
    sample = ds[0]

    print("\nSample info:")
    print(f"  tool_pc shape:      {sample['tool_pc'].shape}")
    print(f"  obj_pc shape:       {sample['obj_pc'].shape}")
    print(f"  delta_pose shape:   {sample['delta_pose'].shape if sample['delta_pose'] is not None else 'None'}")
    print(f"  tool_pts_sdf range: [{sample['tool_pts_sdf'].min():.4f}, {sample['tool_pts_sdf'].max():.4f}]")
    print(f"  obj_pts_sdf range:  [{sample['obj_pts_sdf'].min():.4f}, {sample['obj_pts_sdf'].max():.4f}]")

    if sample['delta_pose'] is None:
        print("\nERROR: delta_pose is None! Run gen_initial.py first.")
        return

    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JointModel(
        head_mode=cfg.head_mode,
        patch_agg=cfg.patch_agg,
        head_hidden=cfg.head_hidden,
        num_pts=cfg.num_pts,
        patch_size=cfg.patch_size,
        encoder_channel=cfg.encoder_channel,
        vit_depth=cfg.vit_depth,
        vit_heads=cfg.vit_heads,
        freeze_encoder=cfg.freeze_encoder,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_emb=cfg.n_emb,
        p_drop_emb=cfg.p_drop_emb,
        p_drop_attn=cfg.p_drop_attn,
        sdf_weight=cfg.sdf_weight,
        diffusion_weight=cfg.diffusion_weight,
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel trainable params: {params:,}")
    print(f"Diffusion timesteps: {model.noise_scheduler.config.num_train_timesteps}")

    # Prepare single sample as batch (B=1)
    tool_pc = sample['tool_pc'].unsqueeze(0).to(device)
    obj_pc = sample['obj_pc'].unsqueeze(0).to(device)
    tool_sdf_gt = sample['tool_pts_sdf'].unsqueeze(0).to(device)
    obj_sdf_gt = sample['obj_pts_sdf'].unsqueeze(0).to(device)
    tool_pc_init = sample['tool_pc_init'].unsqueeze(0).to(device) if sample['tool_pc_init'] is not None else None
    delta_pose_gt = sample['delta_pose'].unsqueeze(0).to(device)

    print(f"\nDelta pose GT:")
    print(f"  translation: {delta_pose_gt[0, :3].tolist()}")
    print(f"  rotation_6d: {delta_pose_gt[0, 3:9].tolist()}")

    # Optimizer with higher LR for overfit
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    print("\n" + "=" * 60)
    print("TRAINING (should converge to near-zero)")
    print("=" * 60)

    for epoch in range(cfg.epochs):
        optimizer.zero_grad()

        loss, metrics = model.loss(
            tool_pc, obj_pc, tool_sdf_gt, obj_sdf_gt,
            tool_pc_init=tool_pc_init,
            delta_pose_gt=delta_pose_gt,
        )

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch < 10:
            sdf_loss = metrics['tool_sdf_loss'] + metrics['obj_sdf_loss']
            diff_loss = metrics.get('diffusion_loss', 0.0)
            print(f"Epoch {epoch:4d}: total={loss.item():.6f}  sdf={sdf_loss:.6f}  diff={diff_loss:.6f}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    sdf_loss = metrics['tool_sdf_loss'] + metrics['obj_sdf_loss']
    diff_loss = metrics.get('diffusion_loss', 0.0)

    if sdf_loss < 0.01 and diff_loss < 0.01:
        print("✓ SUCCESS: Both SDF and diffusion converge!")
        print("  → Architecture works. Can increase timesteps.")
    elif diff_loss > 0.1 and sdf_loss < 0.01:
        print("✗ DIFFUSION BUG: SDF works but diffusion doesn't.")
        print("  → Check transformer or noise scheduler.")
    elif diff_loss < 0.01 and sdf_loss > 0.1:
        print("✗ SDF BUG: Diffusion works but SDF doesn't.")
        print("  → Check encoder or SDF head.")
    else:
        print("✗ BOTH FAIL: Check encoder or training loop.")


if __name__ == "__main__":
    main()