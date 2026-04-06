in=/media/steve/PortableSSD/glbs/
out=/home/steve/Downloads/objaverse/usds
normalized_dir=/home/steve/Downloads/objaverse/normalized

# Directly process all OBJ files in the input directory
python3 -c "
import os
import trimesh
import numpy as np

def normalize_obj_to_unit_diameter(obj_path, normalized_obj_path):
    '''Normalize OBJ file to have diameter of 1.0 using farthest point distance'''
    # Load mesh
    mesh = trimesh.load(obj_path, force='mesh')

    if len(mesh.vertices) == 0:
        print(f'[ERROR] No vertices found in {obj_path}')
        return False

    # Center the mesh at origin first
    mesh.apply_translation(-mesh.centroid)

    # Convert from Y-up to Z-up coordinate system for IsaacSim/USD
    # Rotate +90 degrees around X-axis: Y-up -> Z-up
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.pi/2,  # +90 degrees in radians
        direction=[1, 0, 0],  # X-axis
        point=[0, 0, 0]  # Origin
    )
    mesh.apply_transform(rotation_matrix)
    print(f'[TRANSFORM] Applied Y-up to Z-up conversion (rotation +90° around X-axis)')

    # Calculate the maximum distance from any vertex to origin
    distances = np.linalg.norm(mesh.vertices, axis=1)
    max_distance = np.max(distances)

    if max_distance > 0:
        # Calculate scale factor so that max distance becomes 0.5
        # This makes the diameter (max distance * 2) equal to 1.0
        scale_factor = 0.5 / max_distance

        # Scale the mesh
        mesh.apply_scale(scale_factor)

        # Export normalized mesh
        mesh.export(normalized_obj_path)
        print(f'[NORMALIZE] {obj_path} -> max_distance: {max_distance:.4f} -> scale: {scale_factor:.4f}')
        return True
    else:
        print(f'[ERROR] Invalid mesh vertices for {obj_path}')
        return False

# Create normalized directory if it doesn't exist
os.makedirs('$normalized_dir', exist_ok=True)
print(f'[INFO] Normalized files will be stored in: $normalized_dir')

# Scan all OBJ and GLB files in the input directory
obj_files = []
for filename in os.listdir('$in'):
    base_name, ext = os.path.splitext(filename)
    if ext.lower() in ['.obj', '.glb']:
        obj_files.append((base_name, ext))

print(f'[INFO] Found {len(obj_files)} mesh files (OBJ/GLB) to process')

count = 0
for base, ext in obj_files:
    # if count == 5:
    #     break
    count += 1
    print(f'[INFO] Processing item {count}/{len(obj_files)}: {base}{ext}')

    obj_path = os.path.join('$in', f'{base}{ext}')
    out_path = os.path.join('$out', f'{base}')
    os.makedirs(out_path, exist_ok=True)

    # Create normalized OBJ path in dedicated directory (always export as OBJ for consistency)
    normalized_obj_path = os.path.join('$normalized_dir', f'{base}{ext}')
    usd_path = os.path.join(out_path, f'_{base}.usd')

    if not os.path.exists(obj_path):
        print(f'[ERROR] Input file not found: {obj_path}')
        continue

    if os.path.exists(usd_path):
        print(f'[SKIP] {usd_path} already exists')
        continue

    # Check if normalized file already exists
    if os.path.exists(normalized_obj_path):
        print(f'[SKIP] Normalized file already exists: {normalized_obj_path}')
    else:
        # Normalize input file to unit diameter preserving format
        print(f'[NORMALIZE] Processing {obj_path} -> {normalized_obj_path}')
        if not normalize_obj_to_unit_diameter(obj_path, normalized_obj_path):
            print(f'[SKIP] Failed to normalize {obj_path}')
            continue

    # Use normalized file for conversion (assuming convert_mesh.py supports both OBJ and GLB)
    print(f'[CONVERT] {normalized_obj_path} -> {usd_path}')
    os.system(f'python scripts/tools/convert_mesh.py \"{normalized_obj_path}\" \"{usd_path}\" --mass 0.5 --collision-approximation convexHull --make-instanceable --headless')

    # Post-process USD file to add rigid body properties
    if os.path.exists(usd_path):
        print(f'[POST-PROCESS] Adding rigid body properties to {usd_path}')
        os.system(f'python scripts/tools/post_process_usd.py \"{usd_path}\" --mass {0.5} --colliders-preset convexHull --headless')
    else:
        print(f'[ERROR] Failed to create {usd_path}')

    # Keep normalized OBJ file for potential reuse
    if os.path.exists(normalized_obj_path):
        print(f'[INFO] Normalized file saved: {normalized_obj_path}')

    # break

"