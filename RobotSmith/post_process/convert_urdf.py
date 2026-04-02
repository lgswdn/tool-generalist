import argparse
import os
import glob
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Batch URDF to USD Converter")
parser.add_argument("--eef-dir", type=str, default=None,
                    help="Path to the eef directory (default: ../eef relative to this script)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

def run_batch_conversion(eef_dir):
    input_base_dir = os.path.join(eef_dir, "meshdata_adjusted")
    output_base_dir = os.path.join(eef_dir, "objects_usd")

    search_pattern = os.path.join(input_base_dir, "*", "coacd", "coacd.urdf")
    urdf_files = glob.glob(search_pattern)

    print(f"Found {len(urdf_files)} URDF files. Starting batch conversion...")

    for urdf_path in urdf_files:
        dir_parts = urdf_path.split(os.sep)
        name = dir_parts[-3]

        output_dir = os.path.join(output_base_dir, name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Converting: {name}")

        urdf_cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=output_dir,
            usd_file_name=f"{name}.usd",
            fix_base=False,
            merge_fixed_joints=True,
            force_usd_conversion=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                target_type="none",
            ),
        )
        UrdfConverter(urdf_cfg)

    print("Batch conversion completed successfully.")

if __name__ == "__main__":
    if args_cli.eef_dir is None:
        eef_dir = str(Path(__file__).resolve().parent.parent / "eef")
    else:
        eef_dir = str(Path(args_cli.eef_dir).resolve())

    run_batch_conversion(eef_dir)
    # Shut down the application
    simulation_app.close()