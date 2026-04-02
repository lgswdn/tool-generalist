#!/usr/bin/env python3
"""
Simplified RobotSmith pipeline for manual Gemini interaction.
No API calls, no critic loop - just generate prompt, accept manual input, and create tool.
"""
import os
import json
import argparse
import re
import subprocess
import traceback

project_path = os.path.abspath(os.path.dirname(__file__))

def parse_json(prompt, response):
    """Extract JSON from LLM response."""
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    return json.loads(response)

def add_overlap_to_assemble(assemble_func):
    """Add PRIMITIVE_OVERLAP to offset calculations in assemble function."""
    import re

    def is_max_bound(s):
        """Check if a bounding box reference is a MAX bound."""
        if 'max' in s.lower():
            return True
        match = re.search(r'\[(\d)\]', s)
        if match:
            return int(match.group(1)) >= 3
        return False

    def is_min_bound(s):
        """Check if a bounding box reference is a MIN bound."""
        if 'min' in s.lower():
            return True
        match = re.search(r'\[(\d)\]', s)
        if match:
            return int(match.group(1)) <= 2
        return False

    # Match bounding box subtractions
    pattern = r'((?:\w+_bb\[\d\]|\w*[xyz]\w*(?:min|max|Min|Max|MIN|MAX))\s*-\s*(?:\w+_bb\[\d\]|\w*[xyz]\w*(?:min|max|Min|Max|MIN|MAX)))'

    def replacer(match):
        expr = match.group(1)
        if 'PRIMITIVE_OVERLAP' in expr:
            return expr

        parts = expr.split('-')
        if len(parts) == 2:
            first = parts[0].strip()
            second = parts[1].strip()

            # MAX - MIN: subtract overlap
            if is_max_bound(first) and is_min_bound(second):
                return expr + ' - PRIMITIVE_OVERLAP'
            # MIN - MAX: add overlap
            elif is_min_bound(first) and is_max_bound(second):
                return expr + ' + PRIMITIVE_OVERLAP'

        return expr

    return re.sub(pattern, replacer, assemble_func)

def write_design_code(code_filename, design_json, num_variations=3):
    """Generate Python code from design JSON with parametric variations."""
    outp = ''

    # Copy entire api_tool_design.py content
    with open(os.path.join(project_path, 'utils', 'api_tool_design.py'), 'r') as fi:
        outp += fi.read()
    outp += '\n\n\n\n\n'

    # Add assemble function with automatic overlap injection
    assemble_func = add_overlap_to_assemble(design_json['assemble_func'])
    outp += assemble_func
    outp += '\n\n\n\n\n'

    # Add parts template
    outp += 'parts_template = '
    parts = json.dumps(design_json['parts'], indent=4)
    parts = parts.replace('true', 'True').replace('false', 'False').replace('null', 'None')
    outp += parts
    outp += '\n\n'

    # Add variation generation code
    outp += 'import random\n'
    outp += 'import copy\n'
    outp += 'import json\n\n'
    outp += f'base_tool_name = {repr(design_json["name"])}\n'
    outp += 'all_filenames = []\n'
    outp += f'for i in range({num_variations}):\n'
    outp += f'    print(f"\\nGenerating variation {{i+1}}/{num_variations}...")\n'
    outp += '    parts = copy.deepcopy(parts_template)\n'
    outp += '    semantic_groups = {}\n'
    outp += '    for part in parts:\n'
    outp += '        group = part["semantic_group"]\n'
    outp += '        if group not in semantic_groups:\n'
    outp += '            min_mult, max_mult = part["scale_variance_range"]\n'
    outp += '            semantic_groups[group] = random.uniform(min_mult, max_mult)\n'
    outp += '        multiplier = semantic_groups[group]\n'
    outp += '        part["parameters"] = [p * multiplier for p in part["base_parameters"]]\n'
    outp += '        if part.get("direction_variance_angle") != "base" and part.get("direction_variance_angle", 0) > 0:\n'
    outp += '            import numpy as np\n'
    outp += '            base_dir = part.get("direction", [0, 0, 1])\n'
    outp += '            if isinstance(base_dir, str):\n'
    outp += '                dir_map = {"x": [1,0,0], "+x": [1,0,0], "-x": [-1,0,0], "y": [0,1,0], "+y": [0,1,0], "-y": [0,-1,0], "z": [0,0,1], "+z": [0,0,1], "-z": [0,0,-1]}\n'
    outp += '                base_dir = dir_map.get(base_dir.lower(), [0,0,1])\n'
    outp += '            base_dir = np.array(base_dir, dtype=float)\n'
    outp += '            base_dir = base_dir / np.linalg.norm(base_dir)\n'
    outp += '            angle_rad = np.radians(part["direction_variance_angle"])\n'
    outp += '            perturb_angle = random.uniform(0, angle_rad)\n'
    outp += '            azimuth = random.uniform(0, 2*np.pi)\n'
    outp += '            perp1 = np.array([base_dir[1], -base_dir[0], 0]) if abs(base_dir[2]) < 0.9 else np.array([0, base_dir[2], -base_dir[1]])\n'
    outp += '            perp1 = perp1 / np.linalg.norm(perp1)\n'
    outp += '            perp2 = np.cross(base_dir, perp1)\n'
    outp += '            offset = np.sin(perturb_angle) * (np.cos(azimuth) * perp1 + np.sin(azimuth) * perp2)\n'
    outp += '            part["direction"] = list(base_dir * np.cos(perturb_angle) + offset)\n'
    outp += '    current_filename = f"{base_tool_name}_var_{i:03d}.obj"\n'
    outp += '    temp_filename = f"{base_tool_name}_var_{i:03d}_temp.obj"\n'
    outp += '    filenames = assemble(parts, temp_filename)\n'
    outp += '    import trimesh\n'
    outp += '    temp_mesh = trimesh.load(temp_filename)\n'
    outp += '    bounds = temp_mesh.bounds\n'
    outp += '    min_b, max_b = bounds[0], bounds[1]\n'
    outp += '    margin = 2.0\n'
    outp += '    old_min = GLOBAL_VOXEL_MIN.copy()\n'
    outp += '    old_max = GLOBAL_VOXEL_MAX.copy()\n'
    outp += '    GLOBAL_VOXEL_MIN[:] = min_b - margin\n'
    outp += '    GLOBAL_VOXEL_MAX[:] = max_b + margin\n'
    outp += '    print(f"  [refine] Tightening bounds from [{old_min[0]:.1f}, {old_max[0]:.1f}] to [{GLOBAL_VOXEL_MIN[0]:.1f}, {GLOBAL_VOXEL_MAX[0]:.1f}]")\n'
    outp += '    filenames = assemble(parts, current_filename)\n'
    outp += '    GLOBAL_VOXEL_MIN[:] = old_min\n'
    outp += '    GLOBAL_VOXEL_MAX[:] = old_max\n'
    outp += '    import os\n'
    outp += '    os.remove(temp_filename)\n'
    outp += '    all_filenames.extend(filenames)\n'
    outp += 'with open("output_files.json", "w") as f:\n'
    outp += '    json.dump(all_filenames, f)\n'
    outp += 'print(f"\\nGenerated {len(all_filenames)} files total.")\n'

    with open(code_filename, 'w') as f:
        f.write(outp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--task_prompt_json_dir', type=str, required=True)
    parser.add_argument('--num_variations', type=int, default=100)
    args = parser.parse_args()

    # Create output directory
    log_dir = os.path.join(project_path, args.task_name, 'manual_trial')
    os.makedirs(log_dir, exist_ok=True)
    n_tries = len([f for f in os.listdir(log_dir) if not '.' in f])
    log_dir = os.path.join(log_dir, f"{n_tries:03d}")
    os.makedirs(log_dir, exist_ok=True)

    # Load task prompt
    task_prompt = json.load(open(args.task_prompt_json_dir, 'r'))

    # Load simplified template for manual mode
    template = open(os.path.join(project_path, 'utils', 'template_tool_design_manual.txt'), 'r').read()
    prompt = template.replace("$3D_OBJECT_DESCRIPTION$", task_prompt['3D_OBJECT_DESCRIPTION'])
    prompt = prompt.replace("$GOAL_DESCRIPTION$", task_prompt['GOAL_DESCRIPTION'])
    prompt = prompt.replace("$TIPS_FOR_DESIGNER$", task_prompt['TIPS_FOR_DESIGNER'])

    # Save and display prompt
    prompt_file = os.path.join(log_dir, 'prompt.txt')
    with open(prompt_file, 'w') as f:
        f.write(prompt)

    print("\n" + "="*80)
    print("PROMPT TO COPY TO GEMINI:")
    print("="*80)
    print(prompt)
    print("="*80)
    print(f"\nPrompt saved to: {prompt_file}")
    print("\nNow copy this prompt to Gemini and paste the response below.")
    print("Paste the response and press Ctrl+D (Linux/Mac) or Ctrl+Z (Windows) when done:")
    print("-"*80)

    # Get manual input
    response_lines = []
    try:
        while True:
            line = input()
            response_lines.append(line)
    except EOFError:
        pass

    response = '\n'.join(response_lines)

    # Save response
    response_file = os.path.join(log_dir, 'gemini_response.txt')
    with open(response_file, 'w') as f:
        f.write(response)
    print(f"\nResponse saved to: {response_file}")

    # Parse and generate tool
    try:
        design_json = parse_json(prompt, response)

        # Save design JSON
        json_file = os.path.join(log_dir, 'design.json')
        with open(json_file, 'w') as f:
            json.dump(design_json, f, indent=4)
        print(f"Design JSON saved to: {json_file}")

        # Generate code
        code_file = os.path.join(log_dir, 'design.py')
        write_design_code(code_file, design_json, args.num_variations)
        print(f"Design code saved to: {code_file}")
        print(f"Generating {args.num_variations} variations...")

        # Execute code to generate tool
        print("\nGenerating tool mesh...")
        result = subprocess.run(
            ["python3", code_file],
            cwd=log_dir,
            timeout=600
        )

        if result.returncode != 0:
            print(f"Error: Process exited with code {result.returncode}")
            return

        # Read output files from JSON
        output_files_json = os.path.join(log_dir, 'output_files.json')
        with open(output_files_json, 'r') as f:
            output_files = json.load(f)

        # Render first 3 variations to images
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

                # Load metadata for head_area
                metadata_path = obj_path.replace('.obj', '_metadata.json')
                head_area = None
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as mf:
                        metadata = json.load(mf)
                        mesh_bounds = metadata.get('mesh_bounds')
                        if mesh_bounds and metadata.get('head_area'):
                            tight_min = np.array(mesh_bounds[0])
                            tight_max = np.array(mesh_bounds[1])
                            norm_min = np.array(metadata['head_area'][0])
                            norm_max = np.array(metadata['head_area'][1])
                            head_area = [
                                (tight_min + norm_min * (tight_max - tight_min)).tolist(),
                                (tight_min + norm_max * (tight_max - tight_min)).tolist()
                            ]

                fig = plt.figure(figsize=(10, 10))
                views = [(30, 45, 'view1'), (30, 135, 'view2'), (30, 225, 'view3'), (90, 0, 'top')]

                for elev, azim, name in views:
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                   triangles=faces, cmap='viridis', alpha=0.8,
                                   edgecolor='black', linewidth=0.1)

                    # Draw head_area bounding box
                    if head_area is not None:
                        min_pt, max_pt = head_area[0], head_area[1]
                        # Draw 12 edges of the bounding box
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
