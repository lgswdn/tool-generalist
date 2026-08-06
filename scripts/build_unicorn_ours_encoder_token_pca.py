#!/usr/bin/env python3
"""Fit PCA bases to frozen UniCORN-ours encoder tokens before contact MLPs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretrain.dataset import make_split
from pretrain.model import TCEPointCloudEncoder, TCEPointCloudEncoderCfg
from pretrain.train import collate_fn
from utils.config.loader import load_exp_cfg


DEFAULT_CHECKPOINT = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "unicorn_pretrain_ours_generated_gripper/contact_gen_generated_gripper/"
    "unicorn_contact_ours_generated_gripper_unicorn_contact_ours_generated_gripper/"
    "14fba2398c961a4fc6446b54914910f92471837326a0768ff674a423175b66f0/"
    "best.pt"
)
DEFAULT_DATA_DIR = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_generated_gripper/"
    "fdc5885d5d2a55727c19a6d984557275d2a7f5e48e70f6ef32e01a5bbc03daa3"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts/projections/unicorn_ours_encoder_pre_mlp_pca.pt"
)
DEFAULT_CONFIG = (
    REPO_ROOT / "configs/experiments/unicorn_pretrain_ours_generated_gripper.py"
)


class RunningCovariance:
    """Float64 sufficient statistics without retaining all patch tokens."""

    def __init__(self, dim: int):
        self.count = 0
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.cross = torch.zeros(dim, dim, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1, values.shape[-1]).to(
            device="cpu", dtype=torch.float64
        )
        self.count += values.shape[0]
        self.sum += values.sum(dim=0)
        self.cross += values.T @ values

    def pca(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.count < 2:
            raise RuntimeError("PCA requires at least two patch tokens")
        mean = self.sum / self.count
        covariance = (
            self.cross - self.count * torch.outer(mean, mean)
        ) / (self.count - 1)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        # Components are rows so projection is (x - mean) @ basis.T.
        basis = eigenvectors[:, order].T.contiguous()
        return mean.float(), basis.float(), eigenvalues[order].float()


def _checkpoint_encoder_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"checkpoint must contain a dictionary: {checkpoint_path}")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint has no model state dictionary: {checkpoint_path}")
    for prefix in ("module.encoder.", "encoder."):
        selected = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if selected:
            return selected
    raise RuntimeError(
        "checkpoint has no canonical encoder.* or module.encoder.* state: "
        f"{checkpoint_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-files", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_files < 2:
        raise ValueError("--max-files must be at least 2")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)

    cfg = load_exp_cfg(args.config)
    tce = cfg.model.tce
    encoder = TCEPointCloudEncoder(
        TCEPointCloudEncoderCfg(
            num_pts=tce.num_points,
            patch_size=tce.patch_size,
            encoder_channel=tce.encoder_channel,
            vit_depth=tce.vit_depth,
            vit_heads=tce.vit_heads,
            freeze=True,
            vit_attention_mode=tce.vit_attention_mode,
        )
    )
    encoder.load_state_dict(_checkpoint_encoder_state(args.checkpoint), strict=True)
    device = torch.device(args.device)
    encoder.to(device).eval()

    train_dataset, validation_dataset = make_split(
        data_dir=args.data_dir,
        val_ratio=0.1,
        seed=cfg.general.seed,
        augment=False,
        max_files=args.max_files,
        require_movement=False,
        num_points=tce.num_points,
        num_precontact_steps=cfg.pretrain.num_precontact_steps,
        allow_mock_physics=False,
        noise_max_trans=0.0,
        noise_max_rot_deg=0.0,
        noise_max_retries=1,
        floor_eps=cfg.pretrain.floor_eps,
        validation_seed=cfg.pretrain.validation_noising_seed,
        denoise_target_mode=cfg.pretrain.denoise_target_mode,
        tool_mesh_contract="adjusted_decomposed_mesh",
        include_meshes=False,
        use_geometry_candidates=cfg.pretrain.use_geometry_candidates,
        max_contacts_per_file=cfg.pretrain.max_contacts_per_file,
    )
    # Use every selected file, including the deterministic validation subset.
    dataset = torch.utils.data.ConcatDataset((train_dataset, validation_dataset))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    tool_stats = RunningCovariance(tce.encoder_channel)
    obj_stats = RunningCovariance(tce.encoder_channel)
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            tool = batch["tool_points_E_k"].to(device, non_blocking=True)
            obj = batch["object_points_E_k"].to(device, non_blocking=True)
            rel = batch["rel_tool_object_t_k"].to(device, non_blocking=True)
            if cfg.pretrain.encoder_input_centering == "object_center":
                tool = tool + rel.unsqueeze(-2)
            batch_size, steps, num_points, point_dim = tool.shape
            result = encoder.encode(
                tool.reshape(batch_size * steps, num_points, point_dim),
                obj.reshape(batch_size * steps, num_points, point_dim),
            )
            num_patches = encoder.num_patches
            tool_stats.update(result.fused_tokens[:, :num_patches])
            obj_stats.update(result.fused_tokens[:, num_patches:])
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(
                    f"processed {min(batch_index * args.batch_size, len(dataset))}/"
                    f"{len(dataset)} files",
                    flush=True,
                )

    tool_mean, tool_basis, tool_eigenvalues = tool_stats.pca()
    obj_mean, obj_basis, obj_eigenvalues = obj_stats.pca()
    payload = {
        "schema_version": "unicorn_encoder_token_pca_v1",
        "token_stage": "encoder_pre_mlp",
        "checkpoint": str(args.checkpoint.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "num_files": len(dataset),
        "tool_count": tool_stats.count,
        "obj_count": obj_stats.count,
        "tool_mean": tool_mean,
        "obj_mean": obj_mean,
        "tool_basis": tool_basis,
        "obj_basis": obj_basis,
        "tool_eigenvalues": tool_eigenvalues,
        "obj_eigenvalues": obj_eigenvalues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    tool_rank16 = tool_eigenvalues[:16].sum() / tool_eigenvalues.clamp_min(0).sum()
    obj_rank16 = obj_eigenvalues[:16].sum() / obj_eigenvalues.clamp_min(0).sum()
    print(f"saved: {args.output.resolve()}")
    print(f"rank-16 explained variance: tool={tool_rank16:.6f}, object={obj_rank16:.6f}")


if __name__ == "__main__":
    main()
