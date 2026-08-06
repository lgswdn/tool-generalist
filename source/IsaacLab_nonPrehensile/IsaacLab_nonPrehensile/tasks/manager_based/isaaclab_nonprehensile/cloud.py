import numpy as np
import torch
import meshio
import trimesh
from pathlib import Path


class Cloud:
    def __init__(
        self,
        obj_path,
        target_num_points=512,
        device=None,
        dtype=torch.float16,
        initial_scale=None,
        pointcloud_source="mesh_sampled",
        preprocessed_pointcloud_path=None,
    ):
        """Initialize an object geometry handle with lazily loaded sampled points.

        Args:
            obj_path: Path to OBJ file
            target_num_points: Target number of points (default: 512)
            device: Device to store point cloud (default: None, auto-detect)
            dtype: Data type (default: torch.float16)
            initial_scale: Optional initial scale (default: None)
            pointcloud_source: ``mesh_sampled`` or ``preprocessed``.
            preprocessed_pointcloud_path: Required sampled-point NPY path when
                ``pointcloud_source`` is ``preprocessed``.
        """
        obj_path = Path(obj_path)
        if pointcloud_source not in {"mesh_sampled", "preprocessed"}:
            raise ValueError(
                "pointcloud_source must be 'mesh_sampled' or 'preprocessed', got "
                f"{pointcloud_source!r}"
            )
        if pointcloud_source == "preprocessed" and preprocessed_pointcloud_path is None:
            raise ValueError(
                "preprocessed_pointcloud_path is required when "
                "pointcloud_source='preprocessed'"
            )

        self.target_num_points = int(target_num_points)
        if self.target_num_points <= 0:
            raise ValueError(
                f"target_num_points must be positive, got {self.target_num_points}"
            )
        self.device = torch.device(device) if isinstance(device, (str, int)) else device
        self.dtype = dtype
        self.pointcloud_source = pointcloud_source
        self.preprocessed_pointcloud_path = (
            Path(preprocessed_pointcloud_path).expanduser()
            if preprocessed_pointcloud_path is not None
            else None
        )
        self._initial_scale = (
            np.asarray(initial_scale, dtype=np.float32).reshape(1, 3)
            if initial_scale is not None
            else None
        )
        self._obj_path = str(obj_path)
        self._mesh_point_cache_path = obj_path.parent / "pc_npy_cache" / f"{obj_path.stem}.npy"

        # Sampled points, full vertices, and stable poses are independent lazy
        # resources. Geometry-only command/reset paths therefore do not read a
        # sampled point-cloud file.
        self._points = None
        self._points_torch = {}
        self._vertices_torch = {}
        self._stable_poses_cache = None

    @property
    def points(self) -> torch.Tensor:
        """Return sampled local-frame points, loading the configured source once."""
        if self._points is None:
            points_np = self._load_points_np()
            if self._initial_scale is not None:
                points_np = points_np * self._initial_scale
            self._points = torch.tensor(
                points_np,
                dtype=self.dtype,
                device=self.device,
            )
        return self._points

    def _load_points_np(self) -> np.ndarray:
        if self.pointcloud_source == "preprocessed":
            return self._load_preprocessed_points_np()
        return self._load_mesh_sampled_points_np()

    def _load_preprocessed_points_np(self) -> np.ndarray:
        path = self.preprocessed_pointcloud_path
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"Preprocessed point-cloud file does not exist: {path}"
            )
        try:
            points_np = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load preprocessed point cloud: {path}"
            ) from exc
        points_np = np.asarray(points_np)
        expected_shape = (self.target_num_points, 3)
        if points_np.shape != expected_shape:
            raise RuntimeError(
                f"Preprocessed point cloud has invalid shape: {path} has "
                f"{points_np.shape}, expected {expected_shape}"
            )
        if not np.issubdtype(points_np.dtype, np.number) or not np.isfinite(points_np).all():
            raise RuntimeError(
                f"Preprocessed point cloud contains non-numeric or non-finite data: {path}"
            )
        print(
            "[preprocessed_cloud] load "
            f"obj={self._obj_path} points={path} target_num_points={self.target_num_points}",
            flush=True,
        )
        return points_np.astype(np.float32, copy=False)

    def _load_mesh_sampled_points_np(self) -> np.ndarray:
        obj_path = Path(self._obj_path)
        cache_path = self._mesh_point_cache_path
        target_num_points = self.target_num_points

        if cache_path.exists():
            try:
                cache_size = cache_path.stat().st_size
            except OSError:
                cache_size = -1
            print(
                "[cloud_cache] load "
                f"obj={obj_path} cache={cache_path} bytes={cache_size} "
                f"target_num_points={target_num_points}",
                flush=True,
            )
            try:
                points_np = np.load(cache_path, allow_pickle=False)
            except Exception as exc:
                raise RuntimeError(
                    "Point-cloud cache load failed. "
                    f"obj_path={obj_path} cache_path={cache_path} bytes={cache_size} "
                    f"error={type(exc).__name__}: {exc}. "
                    "The cache file is likely empty/corrupt; remove that .npy file to regenerate it."
                ) from exc
            if points_np.ndim != 2 or points_np.shape[1] != 3:
                raise RuntimeError(
                    "Point-cloud cache has invalid shape. "
                    f"obj_path={obj_path} cache_path={cache_path} shape={points_np.shape}; "
                    "remove that .npy file to regenerate it."
                )
            if points_np.shape[0] != target_num_points:
                print(
                    "[cloud_cache] resample "
                    f"cache={cache_path} cached_points={points_np.shape[0]} "
                    f"target_num_points={target_num_points}",
                    flush=True,
                )
                if points_np.shape[0] > target_num_points:
                    indices = np.random.choice(
                        points_np.shape[0], target_num_points, replace=False
                    )
                    points_np = points_np[indices]
                else:
                    repeats = target_num_points // points_np.shape[0] + 1
                    points_np = np.tile(points_np, (repeats, 1))[:target_num_points]
        else:
            print(
                "[cloud_cache] miss "
                f"obj={obj_path} cache={cache_path} target_num_points={target_num_points}",
                flush=True,
            )
            mesh = trimesh.load(str(obj_path), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            if hasattr(mesh, "faces") and len(mesh.faces) > 0:
                points = mesh.sample(target_num_points)
            else:
                mesh_data = meshio.read(str(obj_path))
                points = mesh_data.points[:, :3]

            if len(points) == target_num_points:
                points_np = points
            elif len(points) > target_num_points:
                indices = np.random.choice(
                    len(points), target_num_points, replace=False
                )
                points_np = points[indices]
            else:
                repeats = target_num_points // len(points) + 1
                tiled = np.tile(points, (repeats, 1))
                points_np = tiled[:target_num_points]

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, points_np.astype(np.float32))
            print(
                "[cloud_cache] saved "
                f"obj={obj_path} cache={cache_path} points={points_np.shape[0]}",
                flush=True,
            )
        return np.asarray(points_np, dtype=np.float32)

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _get_points_torch(self, device: torch.device) -> torch.Tensor:
        """Get points on specified device."""
        if self.device is not None and device == self.device:
            return self.points
        pts = self._points_torch.get(device)
        if pts is None:
            pts = self.points.to(device=device)
            self._points_torch[device] = pts
        return pts

    def _get_vertices_torch(self, device: torch.device) -> torch.Tensor:
        """Get full OBJ mesh vertices on specified device for conservative placement."""
        verts = self._vertices_torch.get(device)
        if verts is not None:
            return verts

        mesh = trimesh.load(self._obj_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
            verts_np = np.asarray(mesh.vertices, dtype=np.float32)
        else:
            mesh_data = meshio.read(self._obj_path)
            verts_np = np.asarray(mesh_data.points[:, :3], dtype=np.float32)

        verts = torch.as_tensor(verts_np, dtype=torch.float32, device=device)
        self._vertices_torch[device] = verts
        return verts

    @staticmethod
    def _quat_wxyz_to_rotmat_torch(quat_wxyz: torch.Tensor) -> torch.Tensor:
        # quat: (N, 4) [w,x,y,z]
        w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        norm = torch.clamp(torch.sqrt(w * w + x * x + y * y + z * z), min=1e-9)
        w = w / norm
        x = x / norm
        y = y / norm
        z = z / norm
        two = 2.0
        xx = two * x * x
        yy = two * y * y
        zz = two * z * z
        xy = two * x * y
        xz = two * x * z
        yz = two * y * z
        wx = two * w * x
        wy = two * w * y
        wz = two * w * z
        r00 = 1 - (yy + zz)
        r01 = xy - wz
        r02 = xz + wy
        r10 = xy + wz
        r11 = 1 - (xx + zz)
        r12 = yz - wx
        r20 = xz - wy
        r21 = yz + wx
        r22 = 1 - (xx + yy)
        Rm = torch.stack(
            [
                torch.stack([r00, r01, r02], dim=-1),
                torch.stack([r10, r11, r12], dim=-1),
                torch.stack([r20, r21, r22], dim=-1),
            ],
            dim=-2,
        )
        return Rm

    @staticmethod
    def _euler_xyz_to_rotmat_torch(
        euler: torch.Tensor, degrees: bool = True
    ) -> torch.Tensor:
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
        Rm = torch.stack(
            [
                torch.stack([r00, r01, r02], dim=-1),
                torch.stack([r10, r11, r12], dim=-1),
                torch.stack([r20, r21, r22], dim=-1),
            ],
            dim=-2,
        )
        return Rm

    def get_pointcloud(
        self, translation=None, rotation=None, scale=None, degrees=True, order="xyz"
    ):
        """Get transformed point clouds for batch processing.

        Args:
            translation: (N, 3) batch translations
            rotation: (N, 3), (N, 4), or (N, 3, 3) batch rotations
            scale: Optional (N, 3) batch scales (for backward compat). If None, uses baked-in scale.
            degrees: Whether Euler angles are in degrees (default: True)
            order: Euler rotation order (default: 'xyz')
        Returns:
            torch.Tensor: (N, M, 3) transformed point clouds
        """
        # Get device from input tensors
        device = None
        for t in (translation, rotation, scale):
            if isinstance(t, torch.Tensor):
                device = t.device
                break
        if device is None:
            device = torch.device("cpu")

        base_points = self._get_points_torch(device)  # (M,3)

        base_points = base_points.to(dtype=self.dtype)

        batch_size = None
        for t in (translation, rotation, scale):
            if isinstance(t, torch.Tensor):
                batch_size = t.shape[0]
                break
        if batch_size is None:
            batch_size = 1

        # Apply runtime scale if provided (backward compat), otherwise use baked-in points
        if scale is not None:
            scale_t = scale if isinstance(scale, torch.Tensor) else torch.as_tensor(scale, device=device, dtype=base_points.dtype)
            if scale_t.ndim == 2 and scale_t.shape[1] == 3:
                scaled = base_points.unsqueeze(0) * scale_t.unsqueeze(1)
            else:
                scaled = base_points.unsqueeze(0).expand(batch_size, -1, -1).clone()
        else:
            # Expand base points for batch (scale already baked in via initial_scale)
            scaled = base_points.unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Rotation
        if rotation is not None:
            rot_t = (
                rotation
                if isinstance(rotation, torch.Tensor)
                else torch.as_tensor(rotation, device=device, dtype=self.dtype)
            )
            rot_t = rot_t.to(device=device, dtype=self.dtype)
            if rot_t.ndim == 2 and rot_t.shape[1] == 3:
                rot_mats = self._euler_xyz_to_rotmat_torch(rot_t, degrees=degrees)
            elif rot_t.ndim == 2 and rot_t.shape[1] == 4:
                rot_mats = self._quat_wxyz_to_rotmat_torch(rot_t)
            elif rot_t.ndim == 3 and rot_t.shape[1:] == (3, 3):
                rot_mats = rot_t
            else:
                raise ValueError("Rotation must be (N, 3)|(N, 4)|(N, 3, 3)")
            rot_mats = rot_mats.contiguous()
            scaled_t = scaled.transpose(1, 2).contiguous()
            transformed = (rot_mats @ scaled_t).transpose(1, 2)
        else:
            transformed = scaled

        # Translation
        if translation is not None:
            trans_t = (
                translation
                if isinstance(translation, torch.Tensor)
                else torch.as_tensor(translation, device=device, dtype=self.dtype)
            )
            trans_t = trans_t.to(device=device, dtype=self.dtype)
            if trans_t.ndim != 2 or trans_t.shape[1] != 3:
                raise ValueError(
                    "Translation must be (N, 3) batch array for multiple environments"
                )
            if trans_t.shape[0] != transformed.shape[0]:
                raise ValueError(
                    f"Translation batch size {trans_t.shape[0]} doesn't match point cloud batch size {transformed.shape[0]}"
                )
            transformed = transformed + trans_t.unsqueeze(1)

        transformed = transformed.to(dtype=self.dtype)

        return transformed

    # ------------------------------------------------------------------
    # Stable pose utilities (used by commands.py for object reset poses)
    # ------------------------------------------------------------------

    def _ensure_stable_poses(self):
        """Lazily compute and cache stable poses."""
        if self._stable_poses_cache is not None:
            return
        mesh = trimesh.load(self._obj_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        stable_poses, probs = mesh.compute_stable_poses()
        self._stable_poses_cache = (stable_poses, probs)

    def sample_stable_pose_trimesh(self, sample_num=64, scale=(1.0, 1.0, 1.0)):
        """
        Sample a stable pose using trimesh's stable_poses method.
        Scale is applied at runtime for flexibility.

        Returns: (position, quaternion)
        - position: (3,) numpy array, z-coordinate is the scaled mesh centroid z
        - quaternion: (4,) numpy array, [w, x, y, z] format to avoid gimbal lock
        """
        self._ensure_stable_poses()
        stable_poses, probs = self._stable_poses_cache

        # Handle case where stable poses are disabled (None)
        if stable_poses is None:
            stable_poses = []
            probs = []

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

        self._ensure_stable_poses()
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
