#!/usr/bin/env python3
"""
Simple OBJ viewer for headless servers using matplotlib.
Usage: python3 view_obj.py <obj_file> [output_dir]
"""
import sys
import os
import json
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def render_obj_to_images(obj_file, output_dir=None):
    """Render OBJ file to PNG images from multiple angles using matplotlib."""

    if output_dir is None:
        output_dir = os.path.dirname(obj_file)
    os.makedirs(output_dir, exist_ok=True)

    # Load mesh
    mesh = trimesh.load(obj_file)
    vertices = mesh.vertices
    faces = mesh.faces

    # Load metadata for head_area
    metadata_path = obj_file.replace('.obj', '_metadata.json')
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

    # Create figure
    fig = plt.figure(figsize=(10, 10))

    # Different viewing angles
    views = [
        (30, 45, 'view1'),
        (30, 135, 'view2'),
        (30, 225, 'view3'),
        (90, 0, 'top'),
    ]

    for elev, azim, name in views:
        ax = fig.add_subplot(111, projection='3d')

        # Plot mesh
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

        # Set view angle
        ax.view_init(elev=elev, azim=azim)

        # Set labels
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

        # Save
        output_path = os.path.join(output_dir, f"{os.path.basename(obj_file)}_{name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

        ax.clear()

    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 view_obj.py <obj_file> [output_dir]")
        sys.exit(1)

    obj_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    render_obj_to_images(obj_file, output_dir)
    print(f"\nRendered images saved!")
