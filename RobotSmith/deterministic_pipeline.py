#!/usr/bin/env python3
"""
Deterministic RobotSmith pipeline.
Uses the LLM-returned JSON directly without any randomness:
- no scale variation sampling
- no rotation variance during attachment
- no surface noise during mesh export
"""
import os
import json
import argparse
import re
import traceback

import numpy as np

project_path = os.path.abspath(os.path.dirname(__file__))


def parse_json(response):
    """Extract JSON from an LLM response."""
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    return json.loads(response)


def export_voxel_object_deterministic(voxel_obj, output_path):
    """Export a voxel object without adding surface noise."""
    import trimesh
    import sys

    sys.path.insert(0, os.path.join(project_path, "utils"))
    from api_tool_design import grid_to_mesh

    mesh = grid_to_mesh(voxel_obj.grid)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("grid_to_mesh did not return a trimesh.Trimesh")
    mesh.export(output_path)


def execute_design_deterministic(design_json, log_dir, num_outputs=1):
    """Execute design JSON directly without any randomized variations."""
    import copy
    import sys
    import trimesh

    sys.path.insert(0, os.path.join(project_path, "utils"))
    from api_tool_design import (
        primitive,
        union_attach,
        subtract_attach,
    )
    import api_tool_design

    base_tool_name = design_json["name"]
    parts_template = design_json["parts"]
    operations = design_json["operations"]
    all_filenames = []

    for i in range(num_outputs):
        print(f"\nGenerating deterministic output {i+1}/{num_outputs}...")
        api_tool_design._BASE_CENTER = None

        parts = copy.deepcopy(parts_template)
        for part in parts:
            if part["geom"] == "arc":
                params = list(part["base_parameters"])
                params[3] = part["base_parameters"][3]
                part["parameters"] = params
            else:
                part["parameters"] = list(part["base_parameters"])

        objects = {}
        cached_bboxes = {}
        for idx, part in enumerate(parts):
            is_head = part.get("is_head", False)
            is_base = idx == 0
            initial_rotation = part.get("arc_plane", None)
            obj = primitive(
                part["geom"],
                part["parameters"],
                is_head=is_head,
                is_base=is_base,
                arc_plane=initial_rotation,
            )
            objects[idx] = obj
            from api_tool_design import get_axis_align_bounding_box

            cached_bboxes[idx] = get_axis_align_bounding_box(obj)

        for op in operations:
            op_type = op["op"]
            target_idx = op["target"]
            source_idx = op["source"]
            zero_variance = [0, 0, 0]

            if op_type == "union_attach":
                objects[target_idx] = union_attach(
                    objects[target_idx],
                    objects[source_idx],
                    op["target_point"],
                    op["source_point"],
                    op["rotation"],
                    zero_variance,
                    cached_target_bbox=cached_bboxes[target_idx],
                )
            elif op_type == "subtract_attach":
                objects[target_idx] = subtract_attach(
                    objects[target_idx],
                    objects[source_idx],
                    op["target_point"],
                    op["source_point"],
                    op["rotation"],
                    zero_variance,
                    cached_target_bbox=cached_bboxes[target_idx],
                )
            else:
                raise ValueError(f"Unsupported operation: {op_type}")

        result = objects[0]

        current_filename = f"{base_tool_name}_det_{i:03d}.obj"
        temp_filename = f"{base_tool_name}_det_{i:03d}_temp.obj"
        output_path = os.path.join(log_dir, temp_filename)
        export_voxel_object_deterministic(result, output_path)

        temp_mesh = trimesh.load(output_path)
        bounds = temp_mesh.bounds
        min_b, max_b = bounds[0], bounds[1]
        margin = 2.0
        old_min = api_tool_design.GLOBAL_VOXEL_MIN.copy()
        old_max = api_tool_design.GLOBAL_VOXEL_MAX.copy()
        api_tool_design.GLOBAL_VOXEL_MIN[:] = min_b - margin
        api_tool_design.GLOBAL_VOXEL_MAX[:] = max_b + margin

        api_tool_design._BASE_CENTER = None
        objects = {}
        cached_bboxes = {}
        for idx, part in enumerate(parts):
            is_head = part.get("is_head", False)
            is_base = idx == 0
            initial_rotation = part.get("arc_plane", None)
            obj = primitive(
                part["geom"],
                part["parameters"],
                is_head=is_head,
                is_base=is_base,
                arc_plane=initial_rotation,
            )
            objects[idx] = obj
            from api_tool_design import get_axis_align_bounding_box

            cached_bboxes[idx] = get_axis_align_bounding_box(obj)

        for op in operations:
            op_type = op["op"]
            target_idx = op["target"]
            source_idx = op["source"]
            zero_variance = [0, 0, 0]

            if op_type == "union_attach":
                objects[target_idx] = union_attach(
                    objects[target_idx],
                    objects[source_idx],
                    op["target_point"],
                    op["source_point"],
                    op["rotation"],
                    zero_variance,
                    cached_target_bbox=cached_bboxes[target_idx],
                )
            elif op_type == "subtract_attach":
                objects[target_idx] = subtract_attach(
                    objects[target_idx],
                    objects[source_idx],
                    op["target_point"],
                    op["source_point"],
                    op["rotation"],
                    zero_variance,
                    cached_target_bbox=cached_bboxes[target_idx],
                )
            else:
                raise ValueError(f"Unsupported operation: {op_type}")

        result = objects[0]

        final_output_path = os.path.join(log_dir, current_filename)
        export_voxel_object_deterministic(result, final_output_path)

        current_voxel_min = api_tool_design.GLOBAL_VOXEL_MIN.copy()
        current_voxel_max = api_tool_design.GLOBAL_VOXEL_MAX.copy()

        mesh = trimesh.load(final_output_path)
        tight_min = mesh.bounds[0]
        tight_max = mesh.bounds[1]

        api_tool_design.GLOBAL_VOXEL_MIN[:] = old_min
        api_tool_design.GLOBAL_VOXEL_MAX[:] = old_max
        os.remove(output_path)

        all_filenames.append(current_filename)

        head_voxels = np.argwhere(result.grid["data"] == 2)
        head_area = None
        if len(head_voxels) > 0:
            voxel_size = (current_voxel_max - current_voxel_min) / result.grid["res"]
            voxel_coords = current_voxel_min + head_voxels * voxel_size
            head_min = voxel_coords.min(axis=0)
            head_max = voxel_coords.max(axis=0)
            head_area = [
                ((head_min - tight_min) / (tight_max - tight_min)).tolist(),
                ((head_max - tight_min) / (tight_max - tight_min)).tolist(),
            ]

        base_center_normalized = None
        if api_tool_design._BASE_CENTER is not None:
            base_center_normalized = (
                (np.array(api_tool_design._BASE_CENTER) - tight_min) / (tight_max - tight_min)
            ).tolist()

        metadata = {
            "head_area": head_area,
            "base_center": base_center_normalized,
            "mesh_bounds": [tight_min.tolist(), tight_max.tolist()],
        }
        metadata_path = os.path.join(log_dir, f"{base_tool_name}_det_{i:03d}_metadata.json")
        with open(metadata_path, "w") as mf:
            json.dump(metadata, mf)

    return all_filenames


def render_outputs(log_dir, output_files):
    print("\nRendering first 3 outputs to images...")
    render_dir = os.path.join(log_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    try:
        import trimesh
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for obj_file in output_files[:3]:
            obj_path = os.path.join(log_dir, obj_file)
            mesh = trimesh.load(obj_path)
            vertices = mesh.vertices
            faces = mesh.faces

            head_area = None
            base_center = None
            metadata_file = obj_path.replace(".obj", "_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as mf:
                    metadata = json.load(mf)
                    mesh_bounds = metadata.get("mesh_bounds")
                    if mesh_bounds:
                        tight_min = np.array(mesh_bounds[0])
                        tight_max = np.array(mesh_bounds[1])
                        if metadata.get("head_area"):
                            norm_min = np.array(metadata["head_area"][0])
                            norm_max = np.array(metadata["head_area"][1])
                            head_area = [
                                (tight_min + norm_min * (tight_max - tight_min)).tolist(),
                                (tight_min + norm_max * (tight_max - tight_min)).tolist(),
                            ]
                        if metadata.get("base_center"):
                            norm_center = np.array(metadata["base_center"])
                            base_center = (tight_min + norm_center * (tight_max - tight_min)).tolist()

            fig = plt.figure(figsize=(10, 10))
            views = [(30, 45, "view1"), (30, 135, "view2"), (30, 225, "view3"), (90, 0, "top")]

            for elev, azim, name in views:
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_trisurf(
                    vertices[:, 0],
                    vertices[:, 1],
                    vertices[:, 2],
                    triangles=faces,
                    cmap="viridis",
                    alpha=0.5,
                    edgecolor="black",
                    linewidth=0.1,
                )

                if head_area is not None:
                    min_pt, max_pt = head_area[0], head_area[1]
                    edges = [
                        [[min_pt[0], min_pt[0]], [min_pt[1], max_pt[1]], [min_pt[2], min_pt[2]]],
                        [[min_pt[0], min_pt[0]], [min_pt[1], min_pt[1]], [min_pt[2], max_pt[2]]],
                        [[min_pt[0], max_pt[0]], [min_pt[1], min_pt[1]], [min_pt[2], min_pt[2]]],
                        [[max_pt[0], max_pt[0]], [min_pt[1], max_pt[1]], [min_pt[2], min_pt[2]]],
                        [[max_pt[0], max_pt[0]], [min_pt[1], min_pt[1]], [min_pt[2], max_pt[2]]],
                        [[min_pt[0], max_pt[0]], [max_pt[1], max_pt[1]], [min_pt[2], min_pt[2]]],
                        [[min_pt[0], min_pt[0]], [max_pt[1], max_pt[1]], [min_pt[2], max_pt[2]]],
                        [[max_pt[0], max_pt[0]], [max_pt[1], max_pt[1]], [min_pt[2], max_pt[2]]],
                        [[min_pt[0], max_pt[0]], [min_pt[1], min_pt[1]], [max_pt[2], max_pt[2]]],
                        [[min_pt[0], max_pt[0]], [max_pt[1], max_pt[1]], [max_pt[2], max_pt[2]]],
                        [[min_pt[0], min_pt[0]], [min_pt[1], max_pt[1]], [max_pt[2], max_pt[2]]],
                        [[max_pt[0], max_pt[0]], [min_pt[1], max_pt[1]], [max_pt[2], max_pt[2]]],
                    ]
                    for edge in edges:
                        ax.plot(edge[0], edge[1], edge[2], "r-", linewidth=2)

                if base_center is not None:
                    ax.scatter(
                        base_center[0],
                        base_center[1],
                        base_center[2],
                        color="blue",
                        s=500,
                        marker="s",
                        edgecolors="white",
                        linewidths=3,
                        depthshade=False,
                        label="Base Center",
                    )
                if head_area is not None or base_center is not None:
                    ax.legend(loc="upper right")

                ax.view_init(elev=elev, azim=azim)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")

                max_range = np.array(
                    [
                        vertices[:, 0].max() - vertices[:, 0].min(),
                        vertices[:, 1].max() - vertices[:, 1].min(),
                        vertices[:, 2].max() - vertices[:, 2].min(),
                    ]
                ).max() / 2.0
                mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
                mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
                mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

                output_path = os.path.join(render_dir, f"{os.path.basename(obj_file)}_{name}.png")
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                ax.clear()

            plt.close()

        print(f"Rendered images saved to: {render_dir}")
    except Exception as render_error:
        print(f"Rendering failed: {render_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, help="Task name for output directory")
    parser.add_argument("--response_file", type=str, required=True, help="LLM response file containing design JSON")
    parser.add_argument("--num_outputs", type=int, default=1, help="Number of deterministic rebuilds to export")
    args = parser.parse_args()

    log_dir = os.path.join(project_path, args.task_name, "deterministic_trial")
    os.makedirs(log_dir, exist_ok=True)
    n_tries = len([f for f in os.listdir(log_dir) if "." not in f])
    log_dir = os.path.join(log_dir, f"{n_tries:03d}")
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("USING EXISTING RESPONSE FILE...")
    print("=" * 80)
    print(f"Reading response from: {args.response_file}")
    with open(args.response_file, "r") as f:
        response = f.read()

    response_file = os.path.join(log_dir, "llm_response.txt")
    with open(response_file, "w") as f:
        f.write(response)
    print(f"Response saved to: {response_file}")

    try:
        design_json = parse_json(response)

        json_file = os.path.join(log_dir, "design.json")
        with open(json_file, "w") as f:
            json.dump(design_json, f, indent=4)
        print(f"Design JSON saved to: {json_file}")

        print(f"Generating {args.num_outputs} deterministic output(s)...")
        output_files = execute_design_deterministic(design_json, log_dir, args.num_outputs)

        output_files_json = os.path.join(log_dir, "output_files.json")
        with open(output_files_json, "w") as f:
            json.dump(output_files, f)

        render_outputs(log_dir, output_files)

        print("\n" + "=" * 80)
        print("DETERMINISTIC TOOL GENERATION SUCCEEDED")
        print("=" * 80)
        print(f"Tool name: {design_json['name']}")
        print(f"Output directory: {log_dir}")
        print(f"Generated .obj files: {output_files}")
        print("=" * 80)
    except Exception as e:
        print(f"\nError: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
