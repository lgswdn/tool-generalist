#!/usr/bin/env python3
"""Convert one-DoF gripper URDF assets declared by a manifest to fixed-base USD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gripper-id", action="append", default=[])
    parser.add_argument("--force", action="store_true", help="Reconvert USD files that already exist.")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _author_hard_mimic_constraints(asset) -> None:
    """Override importer defaults so four-bar mimic linkages cannot stretch."""

    from pxr import Usd

    stage = Usd.Stage.Open(str(asset.usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open converted USD: {asset.usd_path}")
    stage.SetEditTarget(stage.GetRootLayer())
    count = 0
    for prim in stage.TraverseAll():
        axes = {
            prop.split(":", 2)[1]
            for prop in prim.GetPropertyNames()
            if prop.startswith("physxMimicJoint:") and prop.endswith(":referenceJoint")
        }
        for axis in axes:
            prim.GetAttribute(f"physxMimicJoint:{axis}:naturalFrequency").Set(0.0)
            prim.GetAttribute(f"physxMimicJoint:{axis}:dampingRatio").Set(0.0)
            count += 1
    if count == 0:
        raise RuntimeError(f"Converted USD contains no PhysX mimic constraints: {asset.usd_path}")
    stage.GetRootLayer().Save()


def main() -> None:
    args = _parse_args()
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from utils.assets import load_one_dof_gripper_manifest, validate_one_dof_gripper_usd

        assets = load_one_dof_gripper_manifest(args.manifest, require_usd=False)
        selected = set(args.gripper_id)
        if selected:
            unknown = selected.difference(asset.gripper_id for asset in assets)
            if unknown:
                raise ValueError(f"Unknown --gripper-id values: {sorted(unknown)}")
            assets = [asset for asset in assets if asset.gripper_id in selected]

        for asset in assets:
            usd_is_fresh = (
                asset.usd_path.exists()
                and asset.usd_path.stat().st_mtime_ns >= asset.urdf_path.stat().st_mtime_ns
            )
            if usd_is_fresh and not args.force:
                validate_one_dof_gripper_usd(asset)
                print(f"[skip] {asset.gripper_id}: {asset.usd_path}")
                continue
            if asset.usd_path.exists() and not args.force:
                print(f"[stale] {asset.gripper_id}: rebuilding {asset.usd_path}")
            cfg = UrdfConverterCfg(
                asset_path=str(asset.urdf_path),
                usd_dir=str(asset.usd_path.parent),
                usd_file_name=asset.usd_path.name,
                fix_base=True,
                merge_fixed_joints=False,
                # Despite the Isaac Lab cfg field name, this value is passed to
                # importer.set_parse_mimic(). True creates PhysxMimicJointAPI;
                # False imports the followers as unconstrained ordinary joints.
                convert_mimic_joints_to_normal_joints=True,
                force_usd_conversion=True,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                    target_type="none",
                ),
            )
            converter = UrdfConverter(cfg)
            if Path(converter.usd_path).resolve() != asset.usd_path.resolve() or not asset.usd_path.is_file():
                raise RuntimeError(f"Conversion did not create expected USD for {asset.gripper_id}: {asset.usd_path}")
            if asset.control_adapter == "primary_joint_with_mimics":
                _author_hard_mimic_constraints(asset)
            validate_one_dof_gripper_usd(asset)
            print(f"[converted] {asset.gripper_id}: {asset.usd_path}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
