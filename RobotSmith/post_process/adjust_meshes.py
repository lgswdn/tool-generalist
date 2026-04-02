#!/usr/bin/env python3
"""Adjust tool meshes so base_center is at origin, and update head_area accordingly."""

import json
import numpy as np
import glob
import os
import xml.etree.ElementTree as ET
import shutil

def load_obj(filepath):
    vertices = []
    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                faces.append(line)
    return np.array(vertices), faces

def save_obj(filepath, vertices, faces):
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face_line in faces:
            f.write(face_line)

def repair_urdf(urdf_path):
    """Fix missing/zero mass and inertia values in URDF."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    modified = False

    for link in root.findall('link'):
        inertial = link.find('inertial')
        if inertial is None:
            inertial = ET.SubElement(link, 'inertial')
            modified = True

        mass = inertial.find('mass')
        if mass is None:
            mass = ET.Element('mass', {'value': '0.1'})
            inertial.insert(0, mass)
            modified = True
        elif mass.get('value') in ['0', '0.0']:
            mass.set('value', '0.1')
            modified = True

        inertia = inertial.find('inertia')
        if inertia is None:
            inertia = ET.SubElement(inertial, 'inertia',
                                    {'ixx': '0.001', 'ixy': '0', 'ixz': '0',
                                     'iyy': '0.001', 'iyz': '0', 'izz': '0.001'})
            modified = True
        elif inertia.get('ixx') in ['0', '0.0']:
            inertia.set('ixx', '0.001')
            inertia.set('iyy', '0.001')
            inertia.set('izz', '0.001')
            modified = True

    if modified:
        tree.write(urdf_path)
        print(f"  Repaired URDF: {urdf_path}")

def adjust_tool_meshes():
    meshdata_dir = "/mnt/afs/zhuwenxuan/project/RobotSmith/eef/meshdata"
    output_meshdata_dir = "/mnt/afs/zhuwenxuan/project/RobotSmith/eef/meshdata_adjusted"
    tools_json_path = "/mnt/afs/zhuwenxuan/project/RobotSmith/eef/tools.json"

    with open(tools_json_path, 'r') as f:
        tools_data = json.load(f)

    updated_tools = []

    for tool in tools_data:
        name = tool['name']
        base_center_norm = np.array(tool['base_center'])
        head_area_norm = np.array(tool['head_area'])

        coacd_dir = os.path.join(meshdata_dir, name, "coacd")
        obj_files = glob.glob(os.path.join(coacd_dir, "*.obj"))

        if not obj_files:
            print(f"[WARNING] No OBJ files for {name}, skipping")
            continue

        print(f"\n[INFO] Processing {name}")

        # Compute combined bbox from all pieces
        all_vertices = []
        for obj_file in obj_files:
            vertices, _ = load_obj(obj_file)
            all_vertices.append(vertices)
        all_vertices = np.vstack(all_vertices)

        bbox_min = all_vertices.min(axis=0)
        bbox_max = all_vertices.max(axis=0)
        bbox_size = bbox_max - bbox_min

        # Calculate offset to move base_center to origin
        base_center_actual = bbox_min + base_center_norm * bbox_size
        offset = -base_center_actual

        print(f"  Offset: {offset}")

        # Create output directory
        output_coacd_dir = os.path.join(output_meshdata_dir, name, "coacd")
        os.makedirs(output_coacd_dir, exist_ok=True)

        # Copy URDF file
        urdf_src = os.path.join(coacd_dir, "coacd.urdf")
        if os.path.exists(urdf_src):
            urdf_dst = os.path.join(output_coacd_dir, "coacd.urdf")
            shutil.copy(urdf_src, urdf_dst)
            repair_urdf(urdf_dst)

        # Apply offset and save to new location
        for obj_file in obj_files:
            vertices, faces = load_obj(obj_file)
            vertices_shifted = vertices + offset
            output_obj = os.path.join(output_coacd_dir, os.path.basename(obj_file))
            save_obj(output_obj, vertices_shifted, faces)

        # Update head_area
        head_min_actual = bbox_min + head_area_norm[0] * bbox_size
        head_max_actual = bbox_min + head_area_norm[1] * bbox_size
        head_min_shifted = head_min_actual + offset
        head_max_shifted = head_max_actual + offset

        new_bbox_min = bbox_min + offset
        new_bbox_max = bbox_max + offset
        new_bbox_size = new_bbox_max - new_bbox_min

        head_area_new_norm = np.array([
            (head_min_shifted - new_bbox_min) / new_bbox_size,
            (head_max_shifted - new_bbox_min) / new_bbox_size
        ])

        print(f"  New head_area_norm: {head_area_new_norm.tolist()}")

        updated_tools.append({
            "name": name,
            "head_area": head_area_new_norm.tolist()
        })

    # Save updated tools.json
    output_path = "/mnt/afs/zhuwenxuan/project/RobotSmith/eef/tools_adjusted.json"
    with open(output_path, 'w') as f:
        json.dump(updated_tools, f, indent=2)

    print(f"\n[INFO] Saved updated metadata to {output_path}")
    print(f"[INFO] Adjusted meshes saved to {output_meshdata_dir}")

if __name__ == "__main__":
    adjust_tool_meshes()
