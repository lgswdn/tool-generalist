#!/usr/bin/env python3
"""
Automated RobotSmith pipeline using OpenAI API.
Generates prompt, calls GPT API, and creates tool.
"""
import os
import json
import argparse
import re
import subprocess
import traceback
import httpx
import numpy as np
from openai import OpenAI

project_path = os.path.abspath(os.path.dirname(__file__))

def parse_json(prompt, response):
    """Extract JSON from LLM response."""
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    return json.loads(response)

def execute_design_with_variations(design_json, log_dir, num_variations=3):
    """Execute design JSON directly with variations."""
    import sys
    import random
    import copy
    sys.path.insert(0, os.path.join(project_path, 'utils'))
    from api_tool_design import (
        primitive, union_attach, subtract_attach,
        GLOBAL_VOXEL_MIN, GLOBAL_VOXEL_MAX
    )
    import api_tool_design
    import trimesh

    base_tool_name = design_json['name']
    parts_template = design_json['parts']
    operations = design_json['operations']
    all_filenames = []

    for i in range(num_variations):
        print(f"\nGenerating variation {i+1}/{num_variations}...")
        api_tool_design._BASE_CENTER = None

        # Apply random scaling per semantic group
        parts = copy.deepcopy(parts_template)
        semantic_groups = {}
        for part in parts:
            group = part['semantic_group']
            if group not in semantic_groups:
                min_mult, max_mult = part['scale_variance_range']
                scale_mult = random.uniform(min_mult, max_mult)
                semantic_groups[group] = scale_mult
            multiplier = semantic_groups[group]
            # For arc, don't scale the arc_angle (4th parameter) — it's angular, not dimensional
            if part['geom'] == 'arc':
                scaled = [p * multiplier for p in part['base_parameters'][:3]]
                scaled.append(part['base_parameters'][3])  # keep arc_angle as-is
                part['parameters'] = scaled
            else:
                part['parameters'] = [p * multiplier for p in part['base_parameters']]

        # Create primitives and cache bounding boxes
        objects = {}
        cached_bboxes = {}
        for idx, part in enumerate(parts):
            is_head = part.get('is_head', False)
            is_base = (idx == 0)
            initial_rotation = part.get('arc_plane', None)
            obj = primitive(part['geom'], part['parameters'], is_head=is_head, is_base=is_base, arc_plane=initial_rotation)
            objects[idx] = obj
            from api_tool_design import get_axis_align_bounding_box
            cached_bboxes[idx] = get_axis_align_bounding_box(obj)

        # Execute operations - support hierarchical assembly
        for op in operations:
            op_type = op['op']
            target_idx = op['target']
            source_idx = op['source']

            if op_type == 'union_attach':
                objects[target_idx] = union_attach(
                    objects[target_idx], objects[source_idx],
                    op['target_point'], op['source_point'],
                    op['rotation'], op['rotation_variance'],
                    cached_target_bbox=cached_bboxes[target_idx]
                )
            elif op_type == 'subtract_attach':
                objects[target_idx] = subtract_attach(
                    objects[target_idx], objects[source_idx],
                    op['target_point'], op['source_point'],
                    op['rotation'], op['rotation_variance'],
                    cached_target_bbox=cached_bboxes[target_idx]
                )

        result = objects[0]

        # Export temp file to get bounds
        current_filename = f"{base_tool_name}_var_{i:03d}.obj"
        temp_filename = f"{base_tool_name}_var_{i:03d}_temp.obj"
        output_path = os.path.join(log_dir, temp_filename)
        result.export(output_path)

        # Refine bounds
        temp_mesh = trimesh.load(output_path)
        bounds = temp_mesh.bounds
        min_b, max_b = bounds[0], bounds[1]
        margin = 2.0
        old_min = api_tool_design.GLOBAL_VOXEL_MIN.copy()
        old_max = api_tool_design.GLOBAL_VOXEL_MAX.copy()
        api_tool_design.GLOBAL_VOXEL_MIN[:] = min_b - margin
        api_tool_design.GLOBAL_VOXEL_MAX[:] = max_b + margin
        object_size = max_b - min_b
        max_dim = max(object_size)

        # Regenerate with refined bounds
        api_tool_design._BASE_CENTER = None
        objects = {}
        cached_bboxes = {}
        for idx, part in enumerate(parts):
            is_head = part.get('is_head', False)
            is_base = (idx == 0)
            initial_rotation = part.get('arc_plane', None)
            obj = primitive(part['geom'], part['parameters'], is_head=is_head, is_base=is_base, arc_plane=initial_rotation)
            objects[idx] = obj
            from api_tool_design import get_axis_align_bounding_box
            cached_bboxes[idx] = get_axis_align_bounding_box(obj)

        result = objects[0]
        for op in operations:
            op_type = op['op']
            target_idx = op['target']
            source_idx = op['source']
            if op_type == 'union_attach':
                objects[target_idx] = union_attach(
                    objects[target_idx], objects[source_idx],
                    op['target_point'], op['source_point'],
                    op['rotation'], op['rotation_variance'],
                    cached_target_bbox=cached_bboxes[target_idx]
                )
            elif op_type == 'subtract_attach':
                objects[target_idx] = subtract_attach(
                    objects[target_idx], objects[source_idx],
                    op['target_point'], op['source_point'],
                    op['rotation'], op['rotation_variance'],
                    cached_target_bbox=cached_bboxes[target_idx]
                )

        result = objects[0]

        final_output_path = os.path.join(log_dir, current_filename)
        result.export(final_output_path)

        # Save current voxel bounds before restoring
        current_voxel_min = api_tool_design.GLOBAL_VOXEL_MIN.copy()
        current_voxel_max = api_tool_design.GLOBAL_VOXEL_MAX.copy()

        # Compute tight bounds from actual mesh geometry
        mesh = trimesh.load(final_output_path)
        tight_min = mesh.bounds[0]
        tight_max = mesh.bounds[1]

        # Restore globals
        api_tool_design.GLOBAL_VOXEL_MIN[:] = old_min
        api_tool_design.GLOBAL_VOXEL_MAX[:] = old_max

        # Remove temp file
        os.remove(output_path)

        all_filenames.append(current_filename)

        # Save metadata with normalized [0,1] coordinates
        head_voxels = np.argwhere(result.grid["data"] == 2)
        head_area = None
        if len(head_voxels) > 0:
            voxel_size = (current_voxel_max - current_voxel_min) / result.grid["res"]
            voxel_coords = current_voxel_min + head_voxels * voxel_size
            head_min = voxel_coords.min(axis=0)
            head_max = voxel_coords.max(axis=0)
            # Normalize to [0,1]
            head_area = [
                ((head_min - tight_min) / (tight_max - tight_min)).tolist(),
                ((head_max - tight_min) / (tight_max - tight_min)).tolist()
            ]

        base_center_normalized = None
        if api_tool_design._BASE_CENTER is not None:
            base_center_normalized = ((np.array(api_tool_design._BASE_CENTER) - tight_min) / (tight_max - tight_min)).tolist()

        metadata = {
            'head_area': head_area,
            'base_center': base_center_normalized,
            'mesh_bounds': [tight_min.tolist(), tight_max.tolist()]
        }
        metadata_path = os.path.join(log_dir, f"{base_tool_name}_var_{i:03d}_metadata.json")
        with open(metadata_path, 'w') as mf:
            json.dump(metadata, mf)

    return all_filenames



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True, help='Task name for output directory')
    parser.add_argument('--num_variations', type=int, default=3)
    parser.add_argument('--tips', type=str, default='', help='Optional design tips')
    parser.add_argument('--response_file', type=str, default=None, help='Use existing response file instead of calling API')
    args = parser.parse_args()

    task_name = args.task_name

    # Create output directory
    log_dir = os.path.join(project_path, task_name, 'tmp_trial')
    os.makedirs(log_dir, exist_ok=True)
    n_tries = len([f for f in os.listdir(log_dir) if not '.' in f])
    log_dir = os.path.join(log_dir, f"{n_tries:03d}")
    os.makedirs(log_dir, exist_ok=True)

    if not args.response_file:
        # Prompt for goal description
        print("Enter the tool description (e.g., 'spatula for flipping eggs', 'wrench for tightening bolts'):")
        goal_description = input("> ").strip()

        # Load template
        template = open(os.path.join(project_path, 'utils', 'template_tool_design_manual.txt'), 'r').read()
        prompt = template.replace("$3D_OBJECT_DESCRIPTION$", goal_description)
        prompt = prompt.replace("$GOAL_DESCRIPTION$", goal_description)
        prompt = prompt.replace("$TIPS_FOR_DESIGNER$", args.tips)

        # Save prompt
        prompt_file = os.path.join(log_dir, 'prompt.txt')
        with open(prompt_file, 'w') as f:
            f.write(prompt)

    print("\n" + "="*80)
    if args.response_file:
        print("USING EXISTING RESPONSE FILE...")
        print("="*80)
        print(f"Reading response from: {args.response_file}")
        with open(args.response_file, 'r') as f:
            response = f.read()
        prompt = None
    else:
        print("CALLING GPT API...")
        print("="*80)
        print(f"Prompt saved to: {prompt_file}")

        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(
            api_key=api_key,
            base_url="http://43.106.115.130:8080/v1",
            http_client=httpx.Client(trust_env=False)
        )
        response_j = client.responses.create(
            model="gpt-5.4",
            input=[
                {"role": "user", "content": prompt}
            ]
        )
        if isinstance(response_j, str):
            response = response_j
        else:
            response = response_j.output_text

    # Save response
    response_file = os.path.join(log_dir, 'claude_response.txt')
    with open(response_file, 'w') as f:
        f.write(response)
    print(f"Response saved to: {response_file}")

    # Parse and execute design
    try:
        design_json = parse_json(prompt, response)

        # Save design JSON
        json_file = os.path.join(log_dir, 'design.json')
        with open(json_file, 'w') as f:
            json.dump(design_json, f, indent=4)
        print(f"Design JSON saved to: {json_file}")

        # Execute design directly
        print(f"Generating {args.num_variations} variations...")
        output_files = execute_design_with_variations(design_json, log_dir, args.num_variations)

        # Save output files list
        output_files_json = os.path.join(log_dir, 'output_files.json')
        with open(output_files_json, 'w') as f:
            json.dump(output_files, f)

        # Render first 3 variations
        print(f"\nRendering first 3 variations to images...")
        render_dir = os.path.join(log_dir, 'renders')
        os.makedirs(render_dir, exist_ok=True)

        try:
            import trimesh
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np

            for obj_file in output_files[:3]:
                obj_path = os.path.join(log_dir, obj_file)
                mesh = trimesh.load(obj_path)
                vertices = mesh.vertices
                faces = mesh.faces

                # Load metadata if available
                head_area = None
                base_center = None
                contact_file = obj_path.replace('.obj', '_metadata.json')
                if os.path.exists(contact_file):
                    with open(contact_file, 'r') as cf:
                        metadata = json.load(cf)
                        mesh_bounds = metadata.get('mesh_bounds')
                        if mesh_bounds:
                            tight_min = np.array(mesh_bounds[0])
                            tight_max = np.array(mesh_bounds[1])
                            # Denormalize head_area
                            if metadata.get('head_area'):
                                norm_min = np.array(metadata['head_area'][0])
                                norm_max = np.array(metadata['head_area'][1])
                                head_area = [
                                    (tight_min + norm_min * (tight_max - tight_min)).tolist(),
                                    (tight_min + norm_max * (tight_max - tight_min)).tolist()
                                ]
                            # Denormalize base_center
                            if metadata.get('base_center'):
                                norm_center = np.array(metadata['base_center'])
                                base_center = (tight_min + norm_center * (tight_max - tight_min)).tolist()

                fig = plt.figure(figsize=(10, 10))
                views = [(30, 45, 'view1'), (30, 135, 'view2'), (30, 225, 'view3'), (90, 0, 'top')]

                for elev, azim, name in views:
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                   triangles=faces, cmap='viridis', alpha=0.5,
                                   edgecolor='black', linewidth=0.1)

                    # Draw head_area bounding box
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
                            [[max_pt[0], max_pt[0]], [min_pt[1], max_pt[1]], [max_pt[2], max_pt[2]]]
                        ]
                        for edge in edges:
                            ax.plot(edge[0], edge[1], edge[2], 'r-', linewidth=2)
                    if base_center is not None:
                        ax.scatter(base_center[0], base_center[1], base_center[2],
                                 color='blue', s=500, marker='s', edgecolors='white', linewidths=3, depthshade=False, label='Base Center')
                    if head_area is not None or base_center is not None:
                        ax.legend(loc='upper right')

                    ax.view_init(elev=elev, azim=azim)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')

                    # Equal aspect ratio
                    max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                                         vertices[:, 1].max()-vertices[:, 1].min(),
                                         vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
                    mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
                    mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
                    mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)

                    output_path = os.path.join(render_dir, f"{os.path.basename(obj_file)}_{name}.png")
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    ax.clear()

                plt.close()

            print(f"Rendered images saved to: {render_dir}")
        except Exception as render_error:
            print(f"Rendering failed: {render_error}")
            print(f"You can manually render using: python3 view_obj.py {output_files[0]} {render_dir}")

        print("\n" + "="*80)
        print("TOOL GENERATED SUCCESSFULLY!")
        print("="*80)
        print(f"Tool name: {design_json['name']}")
        print(f"Output directory: {log_dir}")
        print(f"Generated .obj files: {output_files}")
        print(f"Rendered images: {render_dir}/*.png")
        print("="*80)

    except Exception as e:
        print(f"\nError: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
