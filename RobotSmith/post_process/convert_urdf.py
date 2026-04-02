import argparse
import os
import glob

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Batch URDF to USD Converter")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

def run_batch_conversion():
    input_base_dir = "/mnt/afs/zhuwenxuan/project/RobotSmith/eef/meshdata_adjusted"
    output_base_dir = "/mnt/afs/zhuwenxuan/project/RobotSmith/eef/objects_usd"

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
    run_batch_conversion()
    # Shut down the application
    simulation_app.close()