import numpy as np
import torch
import trimesh
from typing import Optional

class Cloud:
    def __init__(self, obj_path, target_num_points=512):
        """
        Initialize the point cloud from a .obj mesh file.
        Samples points from the mesh surface and normalizes to fixed number of points.
        Always uses unit scale (1.0) for caching, scale is applied at runtime.
        
        Args:
            obj_path: Path to the .obj mesh file
            target_num_points: Target number of points in the point cloud (default: 512)
        """
        mesh = trimesh.load(obj_path, force='mesh')
        
        # Sample points from mesh surface
        sampled_vertices, _ = trimesh.sample.sample_surface(mesh, target_num_points)
        
        # Ensure we have exactly target_num_points
        assert len(sampled_vertices) == target_num_points, f"Expected {target_num_points} points, got {len(sampled_vertices)}"
        
        self.points = sampled_vertices.tolist()  # Convert numpy array to list
        self.target_num_points = target_num_points
        stable_poses, probs = mesh.compute_stable_poses()
        self._stable_poses_cache = (stable_poses, probs)
        # Per-device Torch tensor cache of base points
        self._points_torch = {}

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _get_points_torch(self, device: torch.device) -> torch.Tensor:
        pts = self._points_torch.get(device)
        if pts is None:
            pts = torch.tensor(self.points, dtype=torch.float32, device=device)
            self._points_torch[device] = pts.to(dtype=torch.float16)
        return pts

    @staticmethod
    def _quat_wxyz_to_rotmat_torch(quat_wxyz: torch.Tensor) -> torch.Tensor:
        # quat: (N, 4) [w,x,y,z]
        w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        norm = torch.clamp(torch.sqrt(w*w + x*x + y*y + z*z), min=1e-9)
        w = w / norm; x = x / norm; y = y / norm; z = z / norm
        two = 2.0
        xx = two * x * x; yy = two * y * y; zz = two * z * z
        xy = two * x * y; xz = two * x * z; yz = two * y * z
        wx = two * w * x; wy = two * w * y; wz = two * w * z
        r00 = 1 - (yy + zz)
        r01 = xy - wz
        r02 = xz + wy
        r10 = xy + wz
        r11 = 1 - (xx + zz)
        r12 = yz - wx
        r20 = xz - wy
        r21 = yz + wx
        r22 = 1 - (xx + yy)
        Rm = torch.stack([
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ], dim=-2)
        return Rm

    @staticmethod
    def _euler_xyz_to_rotmat_torch(euler: torch.Tensor, degrees: bool = True) -> torch.Tensor:
        # euler: (N,3) [roll,pitch,yaw]
        angles = euler if not degrees else (euler * (torch.pi / 180.0))
        rx, ry, rz = angles[:, 0], angles[:, 1], angles[:, 2]
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)
        # R = Rz * Ry * Rx for 'xyz' intrinsic
        r00 = cy * cz
        r01 = cz * sx * sy - cx * sz
        r02 = sx * sz + cx * cz * sy
        r10 = cy * sz
        r11 = cx * cz + sx * sy * sz
        r12 = cx * sy * sz - cz * sx
        r20 = -sy
        r21 = cy * sx
        r22 = cx * cy
        Rm = torch.stack([
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ], dim=-2)
        return Rm

    def get_pointcloud(self, translation=None, rotation=None, scale=None, degrees=True, order='xyz'):
        """
        Returns transformed point clouds for multiple environments with batch processing.
        Scale is applied at runtime for flexibility.

        Parameters:
        - translation: (N, 3) batch translations for N environments.
        - rotation: (N, 3), (N, 4), or (N, 3, 3) batch rotations for N environments.
        - scale: (N, 3) batch scales for N environments (required for batch processing).
        - degrees: Whether the Euler angles are in degrees (default: True).
        - order: Order of Euler rotations (default: 'xyz').

        Returns:
        - torch.Tensor (N, M, 3): transformed point clouds on the input tensors' device
        """
        # Decide device/dtype from first tensor input; default cpu float32
        device = None
        for t in (translation, rotation, scale):
            if isinstance(t, torch.Tensor):
                device = t.device
                break
        if device is None:
            device = torch.device('cpu')
        base_points = self._get_points_torch(device)  # (M,3)

        if scale is None:
            raise ValueError("Scale is required for batch point cloud processing")
        scale_t = scale if isinstance(scale, torch.Tensor) else torch.as_tensor(scale, device=device, dtype=base_points.dtype)
        if scale_t.ndim != 2 or scale_t.shape[1] != 3:
            raise ValueError("Scale must be (N, 3) batch array for multiple environments")
        # Apply scale: (N,1,3) * (1,M,3)
        scaled = base_points.unsqueeze(0) * scale_t.unsqueeze(1)

        # Rotation
        if rotation is not None:
            rot_t = rotation if isinstance(rotation, torch.Tensor) else torch.as_tensor(rotation, device=device, dtype=base_points.dtype)
            if rot_t.ndim == 2 and rot_t.shape[1] == 3:
                rot_mats = self._euler_xyz_to_rotmat_torch(rot_t, degrees=degrees)
            elif rot_t.ndim == 2 and rot_t.shape[1] == 4:
                rot_mats = self._quat_wxyz_to_rotmat_torch(rot_t)
            elif rot_t.ndim == 3 and rot_t.shape[1:] == (3, 3):
                rot_mats = rot_t
            else:
                raise ValueError("Rotation must be (N, 3)|(N, 4)|(N, 3, 3)")
            # Ensure inputs contiguous to avoid implicit copies
            rot_mats = rot_mats.contiguous()
            scaled_t = scaled.transpose(1, 2).contiguous()
            transformed = (rot_mats @ scaled_t).transpose(1, 2)
        else:
            transformed = scaled

        # Translation
        if translation is not None:
            trans_t = translation if isinstance(translation, torch.Tensor) else torch.as_tensor(translation, device=device, dtype=base_points.dtype)
            if trans_t.ndim != 2 or trans_t.shape[1] != 3:
                raise ValueError("Translation must be (N, 3) batch array for multiple environments")
            if trans_t.shape[0] != transformed.shape[0]:
                raise ValueError(f"Translation batch size {trans_t.shape[0]} doesn't match scale batch size {transformed.shape[0]}")
            transformed = transformed + trans_t.unsqueeze(1)

        # Return as half precision to reduce memory/bandwidth
        return transformed

    def sample_stable_pose_trimesh(self, sample_num=64, scale=(1.0, 1.0, 1.0)):
        """
        Sample a stable pose using trimesh's stable_poses method.
        Scale is applied at runtime for flexibility.
        
        Returns: (position, quaternion)
        - position: (3,) numpy array, z-coordinate is the scaled mesh centroid z
        - quaternion: (4,) numpy array, [w, x, y, z] format to avoid gimbal lock
        """
        # Only compute stable poses once per Cloud instance
        if self._stable_poses_cache is None:
            mesh = trimesh.Trimesh(vertices=np.array(self.points))
            stable_poses, probs = mesh.compute_stable_poses()
            self._stable_poses_cache = (stable_poses, probs)
        else:
            stable_poses, probs = self._stable_poses_cache

        if sample_num > len(stable_poses):
            sample_num = len(stable_poses)
        stable_poses = stable_poses[:sample_num]
        probs = probs[:sample_num]

        if len(stable_poses) == 0:
            T = np.eye(4)
        else:
            # Normalize probabilities to sum to 1
            probs_normalized = probs / probs.sum()
            idx = np.random.choice(len(stable_poses), p=probs_normalized)

            T = stable_poses[idx]

        rot = T[:3, :3]
        pos = T[:3, 3]

        # Apply scale to position
        if scale is not None:
            scale = self._to_numpy(scale)
            if scale.ndim == 0:  # scalar
                pos = pos * scale
            elif scale.ndim == 1 and scale.shape[0] == 3:  # (3,) vector
                pos = pos * scale
            else:
                raise ValueError("Scale must be scalar or (3,) array")

        # Convert rotation matrix to quaternion
        from scipy.spatial.transform import Rotation as R
        quat = R.from_matrix(rot).as_quat()  # [x, y, z, w] format
        quat = np.roll(quat, 1).astype(np.float32)  # [w, x, y, z] format (IsaacLab standard)
        
        # Normalize quaternion to ensure it's a unit quaternion
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 1e-12:  # Avoid division by zero
            quat = quat / quat_norm

        return pos, quat

    def sample_stable_pose_trimesh_batch(self, sample_num=64, scale=None):
        """
        Batch version of sample_stable_pose_trimesh.
        
        Args:
            sample_num: Number of samples per environment.
            scale: Batch scales (N, 3) or scalar/(3,) to broadcast.
        
        Returns:
            pos: (N, 3) positions.
            quat: (N, 4) quaternions [w, x, y, z] to avoid gimbal lock.
        """
        # Handle scale input
        if scale is None:
            scale = np.array([1.0, 1.0, 1.0])
        scale = self._to_numpy(scale)
        if scale.ndim == 1:
            scale = scale.reshape(1, 3)
        if scale.ndim != 2 or scale.shape[1] != 3:
            raise ValueError("Batch scale must be (N, 3)")
        batch_size = scale.shape[0]
        
        stable_poses, probs = self._stable_poses_cache
        
        # Truncate to sample_num if provided
        if sample_num > len(stable_poses):
            sample_num = len(stable_poses)
        stable_poses = stable_poses[:sample_num]
        probs = probs[:sample_num]
        
        # Normalize probabilities
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / np.clip(probs.sum(), 1e-12, None)
        
        # Vectorized sampling of indices
        idx = np.random.choice(len(stable_poses), size=batch_size, p=probs)
        Ts = np.asarray(stable_poses)[idx]              # (N, 4, 4)
        rots = Ts[:, :3, :3]                            # (N, 3, 3)
        pos = Ts[:, :3, 3].astype(np.float32)          # (N, 3)
        
        # Apply scale per batch
        pos = pos * scale.astype(np.float32)
        
        # Convert rotation matrix to quaternion
        from scipy.spatial.transform import Rotation as _R
        quat = _R.from_matrix(rots).as_quat()  # [x, y, z, w] format
        quat = np.roll(quat, 1, axis=1).astype(np.float32)  # [w, x, y, z] format (IsaacLab standard)
        
        # Normalize quaternions to ensure they are unit quaternions
        quat_norms = np.linalg.norm(quat, axis=1, keepdims=True)
        quat = np.where(quat_norms > 1e-12, quat / quat_norms, quat)
        
        return pos, quat
