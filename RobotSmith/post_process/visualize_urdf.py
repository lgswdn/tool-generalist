import argparse
import pybullet as p
import numpy as np
from PIL import Image

def add_coordinate_frame(length, radius):
    """Draws physical RGB axes scaled to the object's size."""
    # X axis (Red)
    visual_x = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_x, basePosition=[length/2, 0, 0], baseOrientation=p.getQuaternionFromEuler([0, 1.5707, 0]))

    # Y axis (Green)
    visual_y = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_y, basePosition=[0, length/2, 0], baseOrientation=p.getQuaternionFromEuler([-1.5707, 0, 0]))

    # Z axis (Blue)
    visual_z = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=[0, 0, 1, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_z, basePosition=[0, 0, length/2], baseOrientation=[0, 0, 0, 1])

def render_urdf_dynamic_scale(urdf_path, output_image_path):
    p.connect(p.DIRECT)
    
    # 1. Load the URDF first to measure it
    try:
        robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        print(f"Successfully loaded {urdf_path}")
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        p.disconnect()
        return

    # 2. Calculate the global bounding box of all links
    p.stepSimulation() # Ensure transforms are updated
    min_bounds = []
    max_bounds = []
    
    # Loop through base (-1) and all child joints/links
    for i in range(-1, p.getNumJoints(robot_id)):
        aabb_min, aabb_max = p.getAABB(robot_id, i)
        min_bounds.append(aabb_min)
        max_bounds.append(aabb_max)
        
    global_min = np.min(min_bounds, axis=0)
    global_max = np.max(max_bounds, axis=0)
    
    # Find the largest dimension of the object to use as a scaling factor
    dimensions = global_max - global_min
    max_dim = np.max(dimensions)
    
    if max_dim < 1e-4: 
        max_dim = 1.0 # Fallback if measurement fails

    print(f"Object max dimension: {max_dim:.4f}")

    # 3. Add coordinate frame dynamically scaled to the object
    axis_length = max_dim * 1.2  # Axes 20% larger than the object
    axis_radius = max_dim * 0.015 # Thickness proportional to object size
    add_coordinate_frame(length=axis_length, radius=axis_radius)

    # 4. Adjust camera dynamically
    camera_target = [0, 0, 0] # Always look at the true origin
    cam_dist = max_dim * 1.5  # Pull camera back based on object size
    camera_eye = [cam_dist, cam_dist, cam_dist]
    camera_up = [0, 0, 1]

    view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up)
    projection_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.01, max_dim * 10.0)

    # 5. Render and save
    width, height = 1024, 1024
    _, _, rgbImg, _, _ = p.getCameraImage(
        width=width, height=height, 
        viewMatrix=view_matrix, projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER
    )

    img_array = np.reshape(rgbImg, (height, width, 4))
    img = Image.fromarray(img_array.astype('uint8'), 'RGBA')
    img.save(output_image_path)
    
    print(f"Saved dynamically scaled render to {output_image_path}")
    p.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a URDF with coordinate axes")
    parser.add_argument("--urdf", type=str, required=True,
                        help="Path to the URDF file to render")
    parser.add_argument("--output", type=str, default="urdf.png",
                        help="Output image path (default: urdf.png)")
    args = parser.parse_args()

    render_urdf_dynamic_scale(args.urdf, args.output)