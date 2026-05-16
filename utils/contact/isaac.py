"""Isaac Sim adapter for contact generation physics rollouts.

This module is import-safe on machines without Isaac Sim.  It intentionally does
not import ``isaacsim``, ``omni`` or ``pxr`` at module import time.  The real
adapter creates ``SimulationApp`` first, then imports Omniverse/USD modules, as
required by Isaac Sim standalone Python.  Stabilization acceptance uses the
runtime object/tool poses and a cheap unsigned mesh vertex-cloud distance.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from utils.geometry import pose9d_from_transform_np, rotation_from_pose9d_np
from utils.geometry.mesh_io import load_mesh_vertices_faces
from utils.io import write_json

from .stabilize import IsaacCandidateResult, PhysicsRunConfig


class IsaacAdapterUnavailable(RuntimeError):
    """Raised when the real Isaac adapter cannot initialize Isaac Sim."""


@dataclass(frozen=True)
class _RuntimeBodyState:
    rotation: np.ndarray
    translation: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    source: str = "physx_dynamic_control"


@dataclass(frozen=True)
class _MeshData:
    points: np.ndarray
    faces: np.ndarray
    bbox_center: np.ndarray


def _scale_array(scale: float | Sequence[float]) -> np.ndarray:
    arr = np.asarray(scale, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(3, float(arr), dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"scale must be scalar or shape (3,), got {arr.shape}")
    return arr


def _load_centered_mesh(path: str | Path, scale: float | Sequence[float]) -> _MeshData:
    vertices, faces = load_mesh_vertices_faces(path, process=False)
    scaled = vertices * _scale_array(scale).reshape(1, 3)
    bbox_min = scaled.min(axis=0)
    bbox_max = scaled.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    return _MeshData(
        points=(scaled - center.reshape(1, 3)).astype(np.float64),
        faces=faces.astype(np.int64),
        bbox_center=center.astype(np.float64),
    )


def _tensor_np(value: torch.Tensor | Sequence[float], shape: tuple[int, ...], key: str) -> np.ndarray:
    arr = torch.as_tensor(value, dtype=torch.float64).detach().cpu().numpy()
    if arr.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{key} contains non-finite values")
    return arr


def _pose9d_from_transform(delta_t: np.ndarray, delta_r: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(pose9d_from_transform_np(delta_t, delta_r), dtype=torch.float32)


def _transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T + np.asarray(
        translation, dtype=np.float64
    ).reshape(1, 3)


def _unsigned_pointcloud_distance_min(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> float:
    """Cheap symmetric unsigned distance between two transformed mesh vertex clouds."""

    a = np.asarray(points_a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(points_b, dtype=np.float64).reshape(-1, 3)
    if a.size == 0 or b.size == 0:
        raise ValueError("Cannot compute unsigned object-tool distance with empty point clouds")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Cannot compute unsigned object-tool distance with non-finite point clouds")
    if a.shape[0] > b.shape[0]:
        a, b = b, a
    best_sq = float("inf")
    step = max(1, int(chunk_size))
    for start in range(0, a.shape[0], step):
        chunk = a[start : start + step]
        diff = chunk[:, None, :] - b[None, :, :]
        chunk_best = float(np.min(np.einsum("ijk,ijk->ij", diff, diff)))
        if chunk_best < best_sq:
            best_sq = chunk_best
    return float(np.sqrt(best_sq))


def _to_vector3(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if all(hasattr(value, name) for name in ("x", "y", "z")):
        arr = np.array([float(value.x), float(value.y), float(value.z)], dtype=np.float64)
    else:
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)[:3]
        except Exception:
            return None
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None
    return arr


def _rotation_angle(rotation: np.ndarray) -> float:
    trace = float(np.trace(rotation))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def _rotation_vector_from_delta(rotation: np.ndarray) -> np.ndarray:
    angle = _rotation_angle(rotation)
    if angle <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    denom = 2.0 * np.sin(angle)
    if abs(float(denom)) <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    return axis / denom * angle


def _axis_angle_from_rotation(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    angle = _rotation_angle(rotation)
    if angle < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0
    denom = 2.0 * np.sin(angle)
    if abs(denom) < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) / denom
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis = axis / norm
    return axis, angle


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-9 or abs(angle) < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _rotation_from_pose9d(pose9d: torch.Tensor) -> np.ndarray:
    return rotation_from_pose9d_np(pose9d.detach().cpu().numpy())


def _env_offset(index: int, count: int, spacing: float) -> np.ndarray:
    cols = max(1, int(np.ceil(np.sqrt(max(1, count)))))
    row = int(index) // cols
    col = int(index) % cols
    return np.array([float(col) * spacing, float(row) * spacing, 0.0], dtype=np.float64)


def _create_and_set_attr(create_attr: Any, value: float) -> None:
    try:
        attr = create_attr()
    except TypeError:
        create_attr(value)
        return
    if hasattr(attr, "Set"):
        attr.Set(value)


def _default_launch_config(headless: bool) -> dict[str, Any]:
    if not headless:
        return {"headless": False}
    return {
        "headless": True,
        "hide_ui": True,
        "disable_viewport_updates": True,
        "enable_cameras": False,
        "width": 1,
        "height": 1,
        "window_width": 1,
        "window_height": 1,
        "display_options": 0,
    }


def _force_headless_argv(*, enable_cameras: bool = False) -> None:
    if enable_cameras:
        try:
            sys.argv.remove("--/app/viewport/enabled=false")
        except ValueError:
            pass
    forced_args = [
        "--headless",
        "--no-window",
        "--/app/window/enabled=false",
        "--/app/livestream/enabled=false",
    ]
    if not enable_cameras:
        forced_args.append("--/app/viewport/enabled=false")
    else:
        forced_args.append("--enable_cameras")
    for arg in forced_args:
        if arg not in sys.argv:
            sys.argv.append(arg)


def _tuple3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 3-vector, got {value!r}")
    return float(arr[0]), float(arr[1]), float(arr[2])


def _vec3_from_runtime(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"Runtime rigid body {name} is unavailable")
    if all(hasattr(value, attr) for attr in ("x", "y", "z")):
        arr = np.asarray([float(value.x), float(value.y), float(value.z)], dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)[:3]
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise RuntimeError(f"Runtime rigid body {name} must be a finite 3-vector, got {value!r}")
    return arr


def _quat_xyzw_from_runtime(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"Runtime rigid body {name} quaternion is unavailable")
    if all(hasattr(value, attr) for attr in ("x", "y", "z", "w")):
        quat_xyzw = np.asarray(
            [float(value.x), float(value.y), float(value.z), float(value.w)],
            dtype=np.float64,
        )
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape != (4,):
            raise RuntimeError(f"Runtime rigid body {name} quaternion must have 4 values, got {value!r}")
        quat_xyzw = arr
    if not np.isfinite(quat_xyzw).all():
        raise RuntimeError(f"Runtime rigid body {name} quaternion is non-finite: {value!r}")
    norm = float(np.linalg.norm(quat_xyzw))
    if norm <= 1e-12:
        raise RuntimeError(f"Runtime rigid body {name} quaternion has zero norm")
    return quat_xyzw / norm


def _rotation_from_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_xyzw_from_matrix(rotation: np.ndarray) -> np.ndarray:
    quat_wxyz = _quat_wxyz_from_matrix(rotation)
    return np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)


def _quat_wxyz_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])


def _look_at_ros_quat_wxyz(
    camera_pos: Sequence[float],
    camera_target: Sequence[float],
) -> tuple[float, float, float, float]:
    pos = np.asarray(_tuple3(camera_pos, "visualization_camera_pos"), dtype=np.float64)
    target = np.asarray(_tuple3(camera_target, "visualization_camera_target"), dtype=np.float64)
    forward = target - pos
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-9:
        raise ValueError("visualization camera_pos and camera_target must be different")
    forward /= norm
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    down = np.cross(forward, right)
    down /= max(float(np.linalg.norm(down)), 1e-12)
    # ROS optical camera frame: +X right, +Y down, +Z forward.
    rotation = np.stack([right, down, forward], axis=1)
    return _quat_wxyz_from_matrix(rotation)


def _start_ffmpeg_writer(path: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for contact visualization video output")
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            str(int(fps)),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def _read_camera_rgb(camera_sensor: Any, *, width: int, height: int) -> np.ndarray | None:
    camera_sensor.update(dt=0.0, force_recompute=True)
    data = camera_sensor.data.output.get("rgb")
    if isinstance(data, MappingABC):
        data = data.get("data", data.get("rgb"))
    if data is None:
        return None
    if hasattr(data, "detach"):
        data = data.detach().cpu()
    frame = np.asarray(data)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return None
    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[0] != int(height) or frame.shape[1] != int(width):
        return None
    return np.ascontiguousarray(frame.copy())


def _write_ppm(path: Path, frame: np.ndarray) -> None:
    image = np.asarray(frame, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"PPM frame must have shape HxWx3, got {image.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(image.shape[0]), int(image.shape[1])
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(np.ascontiguousarray(image).tobytes())


def _concat_frames_horizontally(frames: Sequence[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("No frames available for contact visualization picture")
    images = [np.asarray(frame, dtype=np.uint8)[..., :3] for frame in frames]
    max_height = max(int(image.shape[0]) for image in images)
    total_width = sum(int(image.shape[1]) for image in images)
    canvas = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    cursor = 0
    for image in images:
        height, width = int(image.shape[0]), int(image.shape[1])
        canvas[:height, cursor : cursor + width] = image
        cursor += width
    return canvas


def _patch_isaaclab_camera_destructor(Camera: Any) -> None:
    if bool(getattr(Camera, "_tool_generalist_safe_del", False)):
        return
    original_del = getattr(Camera, "__del__", None)
    if not callable(original_del):
        return

    def _safe_del(self):
        try:
            original_del(self)
        except AttributeError as exc:
            if "_rep_registry" not in str(exc):
                raise

    Camera.__del__ = _safe_del
    Camera._tool_generalist_safe_del = True


def _patch_isaaclab_camera_pose_update(Camera: Any) -> None:
    if bool(getattr(Camera, "_tool_generalist_safe_update_poses", False)):
        return

    def _safe_update_poses(self, env_ids):
        if len(self._sensor_prims) == 0:
            raise RuntimeError("Camera prim is None. Please call 'sim.play()' first.")
        poses, quat = self._view.get_world_poses(env_ids)
        if not torch.is_tensor(poses):
            poses = torch.as_tensor(poses, dtype=self._data.pos_w.dtype, device=self._device)
        else:
            poses = poses.to(device=self._device, dtype=self._data.pos_w.dtype)
        if not torch.is_tensor(quat):
            quat = torch.as_tensor(quat, dtype=self._data.quat_w_world.dtype, device=self._device)
        else:
            quat = quat.to(device=self._device, dtype=self._data.quat_w_world.dtype)
        from isaaclab.utils.math import convert_camera_frame_orientation_convention

        self._data.pos_w[env_ids] = poses
        self._data.quat_w_world[env_ids] = convert_camera_frame_orientation_convention(
            quat, origin="opengl", target="world"
        )

    Camera._update_poses = _safe_update_poses
    Camera._tool_generalist_safe_update_poses = True


def _initialize_isaaclab_camera_sensor(camera_sensor: Any, *, label: str) -> None:
    if hasattr(camera_sensor, "_timestamp"):
        return
    initialize = getattr(camera_sensor, "initialize", None)
    if callable(initialize):
        initialize()
    if not hasattr(camera_sensor, "_timestamp"):
        initialize_impl = getattr(camera_sensor, "_initialize_impl", None)
        if not callable(initialize_impl):
            raise RuntimeError(
                f"IsaacLab Camera for {label} has no initialize/_initialize_impl method "
                "and is missing _timestamp."
            )
        try:
            initialize_impl()
            camera_sensor._is_initialized = True
        except Exception as exc:
            raise RuntimeError(
                f"IsaacLab Camera initialization failed for {label}: {type(exc).__name__}: {exc}"
            ) from exc
    reset = getattr(camera_sensor, "reset", None)
    if callable(reset):
        reset()
    if not hasattr(camera_sensor, "_timestamp"):
        raise RuntimeError(
            f"IsaacLab Camera initialization for {label} did not create _timestamp; "
            "cannot capture contact visualization deterministically."
        )


class _ContactVideoRecorder:
    def __init__(
        self,
        *,
        path: Path,
        width: int,
        height: int,
        fps: int,
        camera_pos: Sequence[float],
        camera_target: Sequence[float],
        debug_log: Callable[[str], None],
    ):
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.camera_pos = _tuple3(camera_pos, "visualization_camera_pos")
        self.camera_target = _tuple3(camera_target, "visualization_camera_target")
        self._debug_log = debug_log
        self._writer: subprocess.Popen | None = None
        self._camera_sensor = None
        self.frames = 0
        self.last_frame: np.ndarray | None = None
        self._opened = False

    def open(self, *, camera_prim_path: str) -> None:
        if self._opened:
            return
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.sensors import Camera, CameraCfg
        except Exception as exc:  # pragma: no cover - Isaac-host only.
            raise IsaacAdapterUnavailable(
                "Contact visualization uses the IsaacLab Camera sensor path, matching eval_tools.py. "
                f"Import failed: {type(exc).__name__}: {exc}"
            ) from exc
        _patch_isaaclab_camera_destructor(Camera)
        _patch_isaaclab_camera_pose_update(Camera)
        self._debug_log(f"visual camera create prim={camera_prim_path} path={self.path}")
        camera_cfg = CameraCfg(
            prim_path=camera_prim_path,
            update_period=0.0,
            height=self.height,
            width=self.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=16.0,
                focus_distance=2.0,
                horizontal_aperture=24.0,
                clipping_range=(0.05, 20.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=self.camera_pos,
                rot=_look_at_ros_quat_wxyz(self.camera_pos, self.camera_target),
                convention="ros",
            ),
        )
        self._camera_sensor = Camera(camera_cfg)
        _initialize_isaaclab_camera_sensor(self._camera_sensor, label=str(camera_prim_path))
        self._opened = True

    def start_video(self) -> None:
        if self._writer is None:
            self._writer = _start_ffmpeg_writer(self.path, self.width, self.height, self.fps)

    def read_frame(self) -> np.ndarray | None:
        if self._camera_sensor is None:
            return None
        frame = _read_camera_rgb(self._camera_sensor, width=self.width, height=self.height)
        if frame is not None:
            self.last_frame = frame
        return frame

    def capture(self) -> None:
        if self._writer is None or self._writer.stdin is None or self._camera_sensor is None:
            return
        frame = self.read_frame()
        if frame is None:
            return
        self._writer.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self) -> dict[str, Any]:
        writer = self._writer
        self._writer = None
        if writer is not None:
            if writer.stdin is not None:
                writer.stdin.close()
            return_code = writer.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed for contact visualization video {self.path}: {return_code}")
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames": self.frames,
        }


class IsaacSimAdapter:
    """Real Isaac Sim adapter for stabilize and post-contact rollouts.

    The adapter writes debug USD/JSON artifacts when ``cfg.debug_dir`` is set.
    It returns ``is_real_physics=True`` only for this real adapter; mock adapters
    injected into ``IsaacPhysicsRunner`` should leave that flag false so strict
    final validation cannot mistake test data for real physics output.
    """

    is_real_physics = True

    def __init__(
        self,
        *,
        headless: bool = True,
        launch_config: Mapping[str, Any] | None = None,
        debug_log: Callable[[str], None] | None = None,
    ):
        self.headless = bool(headless)
        self.launch_config = {} if launch_config is None else dict(launch_config)
        self._debug_log = debug_log
        self._simulation_app = None
        self._app_launcher = None
        self._world = None
        self._modules: dict[str, Any] = {}
        self._dynamic_control = None
        self._enable_cameras = False
        self._camera_resolution = (1, 1)

    def _log_phase(self, message: str) -> None:
        if self._debug_log is not None:
            self._debug_log(message)

    def _log_elapsed(self, label: str, started: float) -> None:
        self._log_phase(f"{label} elapsed_s={time.monotonic() - started:.3f}")

    def _ensure_app(self) -> None:
        if self._simulation_app is not None:
            self._configure_camera_runtime_settings()
            return

        if self.headless:
            _force_headless_argv(enable_cameras=bool(self._enable_cameras))
            self._log_phase(f"forced headless argv enable_cameras={bool(self._enable_cameras)}")
            if not self.launch_config:
                try:
                    self._log_phase("before importing isaaclab.app.AppLauncher")
                    from isaaclab.app import AppLauncher
                except Exception as exc:
                    raise IsaacAdapterUnavailable(
                        "Contact generation requires isaaclab.app.AppLauncher in headless mode. "
                        f"Original exception: {type(exc).__name__}: {exc}"
                    ) from exc
                launcher_args = {
                    "headless": True,
                    "enable_cameras": bool(self._enable_cameras),
                    "kit_args": " ".join(
                        tuple(
                            arg
                            for arg in (
                                "--/app/window/enabled=false",
                                None if self._enable_cameras else "--/app/viewport/enabled=false",
                                "--/app/livestream/enabled=false",
                            )
                            if arg is not None
                        )
                    ),
                }
                self._log_phase(f"before AppLauncher args={launcher_args}")
                self._app_launcher = AppLauncher(launcher_args)
                self._simulation_app = self._app_launcher.app
                self._log_phase("after AppLauncher app creation")

        if self._simulation_app is None:
            try:
                self._log_phase("before importing isaacsim.SimulationApp")
                from isaacsim import SimulationApp
            except Exception as exc:  # pragma: no cover - exercised only on Isaac hosts.
                raise IsaacAdapterUnavailable(
                    "Isaac Sim is not importable. Run inside an Isaac Sim Python environment "
                    "or use a mock adapter for unit tests."
                ) from exc

            config = _default_launch_config(self.headless)
            config.update(self.launch_config)
            if self._enable_cameras:
                config.update(
                    {
                        "enable_cameras": True,
                        "disable_viewport_updates": False,
                        "width": int(self._camera_resolution[0]),
                        "height": int(self._camera_resolution[1]),
                        "window_width": int(self._camera_resolution[0]),
                        "window_height": int(self._camera_resolution[1]),
                    }
                )
            self._log_phase(f"before SimulationApp config={config}")
            try:
                self._simulation_app = SimulationApp(launch_config=config)
            except Exception as exc:  # pragma: no cover - exercised only on Isaac hosts.
                raise IsaacAdapterUnavailable(
                    "Isaac Sim SimulationApp startup failed inside contact adapter. "
                    f"Original exception: {type(exc).__name__}: {exc}. "
                    "Check Isaac Python environment, IsaacSim/IsaacLab version match, "
                    "PYTHONPATH, CUDA_VISIBLE_DEVICES, display/Vulkan setup, and stale Kit cache locks."
                ) from exc
            self._log_phase("after app creation")

        self._configure_camera_runtime_settings()

        # Isaac/Omniverse imports must occur after SimulationApp is ready.
        self._log_phase("before Omniverse imports")
        import omni.timeline
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

        try:
            from pxr import UsdShade
        except Exception:  # pragma: no cover - depends on installed Isaac version.
            UsdShade = None

        try:
            from pxr import PhysxSchema
        except Exception:  # pragma: no cover - depends on installed Isaac version.
            PhysxSchema = None

        try:
            from isaacsim.core.api import World
        except Exception as exc:  # pragma: no cover - Isaac host only.
            raise IsaacAdapterUnavailable(
                "Contact generation requires isaacsim.core.api.World. "
                f"Original exception: {type(exc).__name__}: {exc}"
            ) from exc

        self._modules = {
            "Gf": Gf,
            "UsdGeom": UsdGeom,
            "UsdLux": UsdLux,
            "UsdPhysics": UsdPhysics,
            "UsdShade": UsdShade,
            "PhysxSchema": PhysxSchema,
            "World": World,
            "omni_timeline": omni.timeline,
            "omni_usd": omni.usd,
        }
        self._log_phase("after Omniverse imports")

    def _configure_camera_runtime_settings(self) -> None:
        if not self._enable_cameras or self._simulation_app is None:
            return
        try:
            import carb
        except Exception as exc:  # pragma: no cover - Isaac host only.
            raise IsaacAdapterUnavailable(
                "Contact visualization requires IsaacLab camera support, but carb settings are unavailable. "
                f"Original exception: {type(exc).__name__}: {exc}"
            ) from exc
        settings = carb.settings.get_settings()
        settings.set_bool("/isaaclab/cameras_enabled", True)
        settings.set_bool("/app/window/enabled", False)
        settings.set_bool("/app/livestream/enabled", False)
        settings.set_bool("/app/viewport/enabled", True)
        self._log_phase(
            "camera runtime settings enabled "
            "/isaaclab/cameras_enabled=True /app/viewport/enabled=True"
        )

    def initialize(
        self,
        *,
        enable_cameras: bool = False,
        camera_resolution: tuple[int, int] = (1, 1),
    ) -> None:
        """Launch Isaac Sim without running a candidate, useful for smoke tests."""

        self._enable_cameras = bool(enable_cameras)
        self._camera_resolution = (
            max(1, int(camera_resolution[0])),
            max(1, int(camera_resolution[1])),
        )
        self._ensure_app()

    def close(self) -> None:
        if self._simulation_app is not None:
            self._simulation_app = None
            self._app_launcher = None
            self._world = None
            self._dynamic_control = None

    def _new_stage(self) -> Any:
        context = self._modules["omni_usd"].get_context()
        World = self._modules.get("World")
        if World is not None:
            clear_instance = getattr(World, "clear_instance", None)
            if callable(clear_instance):
                try:
                    clear_instance()
                except Exception:
                    pass
        self._world = None
        self._dynamic_control = None
        context.new_stage()
        stage = context.get_stage()
        self._modules["UsdGeom"].SetStageMetersPerUnit(stage, 1.0)
        self._modules["UsdGeom"].SetStageUpAxis(stage, self._modules["UsdGeom"].Tokens.z)
        try:
            self._world = World(stage_units_in_meters=1.0)
        except Exception as exc:
            raise IsaacAdapterUnavailable(
                "Isaac World creation failed; contact generation requires isaacsim.core.api.World."
            ) from exc
        return stage

    def _define_env_root(self, stage: Any, env_path: str) -> None:
        self._modules["UsdGeom"].Xform.Define(stage, env_path)

    def _define_visual_lighting(self, stage: Any) -> None:
        UsdLux = self._modules.get("UsdLux")
        if UsdLux is None:
            return
        try:
            dome = UsdLux.DomeLight.Define(stage, "/World/ContactVideoDomeLight")
            dome.CreateIntensityAttr(90.0)
        except Exception:
            pass
        try:
            distant = UsdLux.DistantLight.Define(stage, "/World/ContactVideoKeyLight")
            distant.CreateIntensityAttr(1600.0)
            distant.CreateAngleAttr(0.18)
        except Exception:
            pass
        try:
            rect = UsdLux.RectLight.Define(stage, "/World/ContactVideoSoftbox")
            rect.CreateIntensityAttr(450.0)
            rect.CreateWidthAttr(0.5)
            rect.CreateHeightAttr(0.35)
            self._set_matrix_xform(
                rect.GetPrim(),
                np.eye(3, dtype=np.float64),
                np.asarray([0.25, -0.35, 0.65], dtype=np.float64),
            )
        except Exception:
            pass

    def _set_matrix_xform(
        self,
        prim: Any,
        rotation: np.ndarray,
        translation: np.ndarray,
        scale: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        Gf = self._modules["Gf"]
        UsdGeom = self._modules["UsdGeom"]
        rot = np.asarray(rotation, dtype=np.float64)
        trans = np.asarray(translation, dtype=np.float64)
        scale_arr = _scale_array(scale)
        rs = rot * scale_arr.reshape(1, 3)
        matrix = Gf.Matrix4d(
            float(rs[0, 0]), float(rs[0, 1]), float(rs[0, 2]), 0.0,
            float(rs[1, 0]), float(rs[1, 1]), float(rs[1, 2]), 0.0,
            float(rs[2, 0]), float(rs[2, 1]), float(rs[2, 2]), 0.0,
            float(trans[0]), float(trans[1]), float(trans[2]), 1.0,
        )
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(matrix)

    def _get_matrix_pose(self, prim: Any) -> tuple[np.ndarray, np.ndarray]:
        UsdGeom = self._modules["UsdGeom"]
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        rotation = np.array([[float(matrix[i][j]) for j in range(3)] for i in range(3)], dtype=np.float64)
        translation = np.array([float(matrix[3][0]), float(matrix[3][1]), float(matrix[3][2])], dtype=np.float64)
        return rotation, translation

    def _runtime_control(self) -> Any:
        if self._dynamic_control is not None:
            return self._dynamic_control
        errors: list[str] = []
        for module_name in (
            "omni.isaac.dynamic_control._dynamic_control",
            "isaacsim.core.utils.dynamic_control",
        ):
            try:
                module = __import__(module_name, fromlist=["_"])
                acquire = getattr(module, "acquire_dynamic_control_interface", None)
                if not callable(acquire):
                    errors.append(f"{module_name}: missing acquire_dynamic_control_interface")
                    continue
                self._dynamic_control = acquire()
                return self._dynamic_control
            except Exception as exc:
                errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
                "Runtime rigid body state unavailable; deterministic contact generation refuses USD transform substitution. "
            "Could not acquire PhysX dynamic_control interface: "
            + "; ".join(errors)
        )

    def _runtime_vec_like(self, sample: Any, values: Sequence[float], name: str) -> Any:
        arr = _vec3_from_runtime(values, name)
        sample_type = type(sample)
        try:
            return sample_type(float(arr[0]), float(arr[1]), float(arr[2]))
        except Exception:
            pass
        try:
            result = sample_type()
            result.x = float(arr[0])
            result.y = float(arr[1])
            result.z = float(arr[2])
            return result
        except Exception as exc:
            raise RuntimeError(
                f"Runtime rigid body vector construction failed for {name}; "
                "deterministic contact generation refuses USD transform substitution."
            ) from exc

    def _runtime_quat_like(self, sample: Any, quat_xyzw: Sequence[float], name: str) -> Any:
        arr = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)
        if arr.shape != (4,) or not np.isfinite(arr).all():
            raise RuntimeError(f"Runtime rigid body quaternion for {name} must be finite xyzw, got {quat_xyzw!r}")
        arr = arr / max(float(np.linalg.norm(arr)), 1e-12)
        sample_type = type(sample)
        try:
            return sample_type(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
        except Exception:
            pass
        try:
            result = sample_type()
            result.x = float(arr[0])
            result.y = float(arr[1])
            result.z = float(arr[2])
            result.w = float(arr[3])
            return result
        except Exception as exc:
            raise RuntimeError(
                f"Runtime rigid body quaternion construction failed for {name}; "
                "deterministic contact generation refuses USD transform substitution."
            ) from exc

    def _runtime_pose_like(self, sample: Any, translation: Sequence[float], rotation: np.ndarray, name: str) -> Any:
        position = self._runtime_vec_like(getattr(sample, "p", None), translation, f"{name}.pose.p")
        quaternion = self._runtime_quat_like(
            getattr(sample, "r", None),
            _quat_xyzw_from_matrix(rotation),
            f"{name}.pose.r",
        )
        sample_type = type(sample)
        try:
            return sample_type(position, quaternion)
        except Exception:
            pass
        try:
            result = sample_type()
            result.p = position
            result.r = quaternion
            return result
        except Exception as exc:
            raise RuntimeError(
                f"Runtime rigid body pose construction failed for {name}; "
                "deterministic contact generation refuses USD transform substitution."
            ) from exc

    def _bind_runtime_handles(self, envs: Sequence[Mapping[str, Any]]) -> None:
        dc = self._runtime_control()
        for env in envs:
            for label, path_key, handle_key in (
                ("object", "object_path", "object_runtime_handle"),
                ("tool", "tool_path", "tool_runtime_handle"),
            ):
                path = str(env[path_key])
                handle = dc.get_rigid_body(path)
                if handle is None or int(handle) == 0:
                    raise RuntimeError(
                        f"Runtime rigid body state unavailable for {label} at {path}; "
                        "deterministic contact generation refuses USD transform substitution."
                    )
                env[handle_key] = handle

    def _set_runtime_body_state(
        self,
        env: Mapping[str, Any],
        body_name: str,
        *,
        rotation: np.ndarray,
        translation: np.ndarray,
        linear_velocity: Sequence[float] = (0.0, 0.0, 0.0),
        angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
        set_velocity: bool = True,
    ) -> None:
        dc = self._runtime_control()
        handle = env.get(f"{body_name}_runtime_handle")
        if handle is None or int(handle) == 0:
            raise RuntimeError(
                f"Runtime rigid body state unavailable for {body_name}; "
                "deterministic contact generation refuses USD transform substitution."
            )
        try:
            current_pose = dc.get_rigid_body_pose(handle)
            pose = self._runtime_pose_like(current_pose, translation, rotation, body_name)
            set_pose = getattr(dc, "set_rigid_body_pose", None)
            if not callable(set_pose):
                raise RuntimeError("dynamic_control pose setter API is unavailable")
            set_pose(handle, pose)
            if set_velocity:
                current_linear_velocity = dc.get_rigid_body_linear_velocity(handle)
                current_angular_velocity = dc.get_rigid_body_angular_velocity(handle)
                lin = self._runtime_vec_like(current_linear_velocity, linear_velocity, f"{body_name}.linear_velocity")
                ang = self._runtime_vec_like(current_angular_velocity, angular_velocity, f"{body_name}.angular_velocity")
                set_linear = getattr(dc, "set_rigid_body_linear_velocity", None)
                set_angular = getattr(dc, "set_rigid_body_angular_velocity", None)
                if not callable(set_linear) or not callable(set_angular):
                    raise RuntimeError("dynamic_control velocity setter API is unavailable")
                set_linear(handle, lin)
                set_angular(handle, ang)
        except Exception as exc:
            raise RuntimeError(
                f"Runtime rigid body deterministic initialization failed for {body_name}; "
                "deterministic contact generation refuses USD transform substitution. "
                f"PhysX runtime API failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _initialize_runtime_bodies(self, envs: Sequence[Mapping[str, Any]]) -> None:
        for env in envs:
            self._set_runtime_body_state(
                env,
                "object",
                rotation=env["initial_object_rotation_E"],
                translation=env["initial_object_center_world"],
            )
            self._set_runtime_body_state(
                env,
                "tool",
                rotation=env["initial_tool_rotation_E"],
                translation=env["initial_tool_center_world"],
                set_velocity=False,
            )

    def _get_runtime_body_state(self, env: Mapping[str, Any], body_name: str) -> _RuntimeBodyState:
        dc = self._runtime_control()
        handle_key = f"{body_name}_runtime_handle"
        handle = env.get(handle_key)
        if handle is None or int(handle) == 0:
            raise RuntimeError(
                f"Runtime rigid body state unavailable for {body_name}; "
                "deterministic contact generation refuses USD transform substitution."
            )
        try:
            pose = dc.get_rigid_body_pose(handle)
            linear_velocity = dc.get_rigid_body_linear_velocity(handle)
            angular_velocity = dc.get_rigid_body_angular_velocity(handle)
        except Exception as exc:
            raise RuntimeError(
                f"Runtime rigid body state unavailable for {body_name}; "
                "deterministic contact generation refuses USD transform substitution. "
                f"PhysX runtime API failed: {type(exc).__name__}: {exc}"
            ) from exc
        translation = _vec3_from_runtime(getattr(pose, "p", None), f"{body_name}.pose.p")
        quat_xyzw = _quat_xyzw_from_runtime(getattr(pose, "r", None), f"{body_name}.pose.r")
        rotation = _rotation_from_quat_xyzw(quat_xyzw)
        return _RuntimeBodyState(
            rotation=rotation,
            translation=translation,
            quaternion_wxyz=np.asarray(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                dtype=np.float64,
            ),
            linear_velocity=_vec3_from_runtime(linear_velocity, f"{body_name}.linear_velocity"),
            angular_velocity=_vec3_from_runtime(angular_velocity, f"{body_name}.angular_velocity"),
        )

    def _apply_physics_material(
        self,
        stage: Any,
        prim: Any,
        material_path: str,
        friction: float,
    ) -> dict[str, Any]:
        UsdPhysics = self._modules["UsdPhysics"]
        UsdShade = self._modules.get("UsdShade")
        if UsdShade is None:
            raise IsaacAdapterUnavailable("UsdShade is unavailable; cannot bind physics material")

        friction = float(friction)
        material = UsdShade.Material.Define(stage, material_path)
        material_prim = material.GetPrim()
        physics_material = UsdPhysics.MaterialAPI.Apply(material_prim)
        _create_and_set_attr(physics_material.CreateStaticFrictionAttr, friction)
        _create_and_set_attr(physics_material.CreateDynamicFrictionAttr, friction)
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        try:
            binding_api.Bind(material, materialPurpose="physics")
        except TypeError:
            try:
                binding_api.Bind(material, "physics")
            except TypeError:
                binding_api.Bind(material)
        return {
            "material_path": str(material_path),
            "static_friction": friction,
            "dynamic_friction": friction,
            "bound": True,
        }

    def _define_mesh_body(
        self,
        stage: Any,
        *,
        prim_path: str,
        mesh: _MeshData,
        rotation: np.ndarray,
        translation: np.ndarray,
        mass: float,
        friction: float,
        kinematic: bool,
    ) -> tuple[Any, dict[str, Any]]:
        Gf = self._modules["Gf"]
        UsdGeom = self._modules["UsdGeom"]
        UsdPhysics = self._modules["UsdPhysics"]
        PhysxSchema = self._modules["PhysxSchema"]

        usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
        usd_mesh.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in mesh.points])
        usd_mesh.CreateFaceVertexCountsAttr([3 for _ in mesh.faces])
        usd_mesh.CreateFaceVertexIndicesAttr([int(v) for face in mesh.faces for v in face])
        usd_mesh.CreateDoubleSidedAttr(True)
        usd_mesh.CreateDisplayColorAttr(
            [Gf.Vec3f(0.8, 0.28, 0.18) if kinematic else Gf.Vec3f(0.18, 0.42, 0.85)]
        )
        prim = usd_mesh.GetPrim()
        self._set_matrix_xform(prim, rotation, translation)

        UsdPhysics.CollisionAPI.Apply(prim)
        try:
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            try:
                attr = mesh_collision.CreateApproximationAttr()
            except Exception:
                attr = mesh_collision.GetApproximationAttr()
            attr.Set("convexHull")
        except Exception:
            pass
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid_body.CreateKinematicEnabledAttr(bool(kinematic))
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
        material = self._apply_physics_material(
            stage,
            prim,
            f"{prim_path}_PhysicsMaterial",
            friction,
        )
        if PhysxSchema is not None:
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        return prim, material

    def _define_ground(
        self,
        stage: Any,
        friction: float,
        *,
        prim_path: str = "/World/Ground",
        center: Sequence[float] = (0.0, 0.0, -0.05),
    ) -> tuple[Any, dict[str, Any]]:
        UsdGeom = self._modules["UsdGeom"]
        UsdPhysics = self._modules["UsdPhysics"]
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([self._modules["Gf"].Vec3f(0.55, 0.55, 0.55)])
        prim = cube.GetPrim()
        self._set_matrix_xform(
            prim,
            np.eye(3, dtype=np.float64),
            np.asarray(center, dtype=np.float64),
            scale=(4.0, 4.0, 0.1),
        )
        UsdPhysics.CollisionAPI.Apply(prim)
        material = self._apply_physics_material(stage, prim, f"{prim_path}_PhysicsMaterial", friction)
        return prim, material

    def _step(self, steps: int, *, reset: bool = False, force_render: bool = False) -> None:
        steps = max(0, int(steps))
        if self._world is not None:
            if reset:
                try:
                    self._world.reset()
                except Exception:
                    pass
            for _ in range(steps):
                self._world.step(render=bool(force_render) or not self.headless)
            return
        raise IsaacAdapterUnavailable(
            "Isaac World API is unavailable; contact generation refuses SimulationApp.update() substitution."
        )

    def _step_and_capture(
        self,
        steps: int,
        *,
        reset: bool = False,
        recorders: Sequence[_ContactVideoRecorder] = (),
    ) -> None:
        if not recorders:
            self._step(steps, reset=reset)
            return
        if reset:
            self._step(0, reset=True, force_render=True)
        for local_step in range(max(0, int(steps))):
            self._step(1, force_render=True)
            for recorder in recorders:
                recorder.capture()

    def _render_visual_frame(self) -> None:
        if self._world is not None:
            render = getattr(self._world, "render", None)
            if callable(render):
                render()
                return
        raise IsaacAdapterUnavailable(
            "Isaac World render API is unavailable; contact visualization refuses SimulationApp.update() substitution."
        )

    def _timeline_pose_state(self, envs: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, np.ndarray]]:
        state: dict[int, dict[str, np.ndarray]] = {}
        for env in envs:
            index = int(env["index"])
            object_state = self._get_runtime_body_state(env, "object")
            tool_state = self._get_runtime_body_state(env, "tool")
            state[index] = {
                "object_rotation": object_state.rotation,
                "object_translation": object_state.translation,
                "object_quaternion_wxyz": object_state.quaternion_wxyz,
                "object_linear_velocity": object_state.linear_velocity,
                "object_angular_velocity": object_state.angular_velocity,
                "tool_rotation": tool_state.rotation,
                "tool_translation": tool_state.translation,
                "tool_quaternion_wxyz": tool_state.quaternion_wxyz,
                "tool_linear_velocity": tool_state.linear_velocity,
                "tool_angular_velocity": tool_state.angular_velocity,
            }
        return state

    def _physics_dt(self) -> float | None:
        if self._world is not None:
            for attr in ("get_physics_dt", "get_rendering_dt"):
                getter = getattr(self._world, attr, None)
                if callable(getter):
                    try:
                        value = float(getter())
                    except Exception:
                        continue
                    if np.isfinite(value) and value > 0.0:
                        return value
        return None

    def _write_debug_artifacts(
        self,
        *,
        cfg: PhysicsRunConfig,
        candidate_index: int,
        stage: Any,
        payload: Mapping[str, Any],
    ) -> tuple[str | None, str | None]:
        if not cfg.debug_dir:
            return None, None
        debug_dir = Path(cfg.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        stage_path = debug_dir / f"isaac_contact_{candidate_index:06d}.usd"
        debug_json_path = debug_dir / f"isaac_contact_{candidate_index:06d}.json"
        try:
            stage.GetRootLayer().Export(str(stage_path))
        except Exception:
            stage_path = None
        write_json(debug_json_path, dict(payload))
        return str(stage_path) if stage_path is not None else None, str(debug_json_path)

    def _failure_result(
        self,
        *,
        status: str,
        candidate_index: int,
        stabilize_steps: int = 0,
        postcontact_steps: int = 0,
        stabilized: Mapping[str, Any] | None = None,
        post_tool_delta_pose9d_E: Any | None = None,
        post_tool_achieved_delta_pose9d_E: Any | None = None,
        post_object_delta_pose9d_E: Any | None = None,
        stabilized_in_contact: bool = False,
        stabilized_contact_count: int = 0,
        stabilized_contact_impulse_norm: float = 0.0,
        metrics: Mapping[str, Any] | None = None,
    ) -> IsaacCandidateResult:
        return IsaacCandidateResult(
            success=False,
            status=status,
            stabilize_steps=stabilize_steps,
            stabilized_in_contact=stabilized_in_contact,
            stabilized_contact_count=stabilized_contact_count,
            stabilized_contact_impulse_norm=stabilized_contact_impulse_norm,
            post_tool_delta_pose9d_E=post_tool_delta_pose9d_E,
            post_tool_achieved_delta_pose9d_E=post_tool_achieved_delta_pose9d_E,
            post_object_delta_pose9d_E=post_object_delta_pose9d_E,
            postcontact_steps=postcontact_steps,
            stabilized=stabilized,
            metrics={"candidate_index": int(candidate_index), **({} if metrics is None else dict(metrics))},
        )

    def _create_visual_recorders(
        self,
        *,
        cfg: PhysicsRunConfig,
        envs: Sequence[Mapping[str, Any]],
    ) -> dict[int, _ContactVideoRecorder]:
        if not cfg.visualization_enabled:
            return {}
        if not cfg.visualization_stabilization_picture and not cfg.visualization_postcontact_video:
            self._log_phase("visual recorders disabled")
            return {}
        if int(cfg.visualization_video_width) <= 0 or int(cfg.visualization_video_height) <= 0:
            raise ValueError("Contact visualization video dimensions must be positive")
        if int(cfg.visualization_video_fps) <= 0:
            raise ValueError("Contact visualization video_fps must be positive")
        max_count = max(
            int(cfg.visualization_postcontact_video_num) if cfg.visualization_postcontact_video else 0,
            int(cfg.visualization_stabilization_picture_num) if cfg.visualization_stabilization_picture else 0,
        )
        if max_count <= 0:
            raise ValueError("Contact visualization recorder count must be positive when visualization output is enabled")
        video_root = Path(cfg.visualization_video_dir or (Path(cfg.debug_dir or ".") / "videos"))
        recorders: dict[int, _ContactVideoRecorder] = {}
        for env in list(envs)[:max_count]:
            index = int(env["index"])
            recorders[index] = self._create_visual_recorder(cfg=cfg, env=env, video_root=video_root)
        self._log_phase(
            "visual recorders ready "
            f"count={len(recorders)} video_enabled={cfg.visualization_postcontact_video} "
            f"stabilization_picture={cfg.visualization_stabilization_picture} dir={video_root}"
        )
        return recorders

    def _create_visual_recorder(
        self,
        *,
        cfg: PhysicsRunConfig,
        env: Mapping[str, Any],
        video_root: Path,
    ) -> _ContactVideoRecorder:
        index = int(env["index"])
        offset = np.asarray(env["offset"], dtype=np.float64)
        base_pos = np.asarray(_tuple3(cfg.visualization_camera_pos, "visualization_camera_pos"))
        base_target = np.asarray(_tuple3(cfg.visualization_camera_target, "visualization_camera_target"))
        recorder = _ContactVideoRecorder(
            path=video_root / f"candidate_{index:06d}.mp4",
            width=int(cfg.visualization_video_width),
            height=int(cfg.visualization_video_height),
            fps=int(cfg.visualization_video_fps),
            camera_pos=tuple((base_pos + offset).tolist()),
            camera_target=tuple((base_target + offset).tolist()),
            debug_log=self._log_phase,
        )
        recorder.open(camera_prim_path=f"{env['env_path']}/ContactRecordCamera")
        return recorder

    def _start_postcontact_video_recorders(
        self,
        *,
        cfg: PhysicsRunConfig,
        recorders: Mapping[int, _ContactVideoRecorder],
        active_indices: set[int],
    ) -> dict[int, _ContactVideoRecorder]:
        if not cfg.visualization_enabled or not cfg.visualization_postcontact_video:
            self._log_phase("postcontact video disabled")
            return {}
        active: dict[int, _ContactVideoRecorder] = {}
        for index, recorder in recorders.items():
            if int(index) not in active_indices:
                continue
            recorder.start_video()
            active[int(index)] = recorder
        self._log_phase(
            "postcontact video recording enabled "
            f"count={len(active)} fps={cfg.visualization_video_fps}"
        )
        return active

    def _close_video_recorders(
        self,
        recorders: Mapping[int, _ContactVideoRecorder],
    ) -> dict[int, dict[str, Any]]:
        metadata: dict[int, dict[str, Any]] = {}
        for index, recorder in recorders.items():
            metadata[int(index)] = recorder.close()
            self._log_phase(
                "video recording saved "
                f"candidate={int(index)} path={metadata[int(index)]['path']} "
                f"frames={metadata[int(index)]['frames']}"
            )
        return metadata

    def _write_contact_picture_summary(
        self,
        *,
        cfg: PhysicsRunConfig,
        envs: Sequence[Mapping[str, Any]],
        recorders: dict[int, _ContactVideoRecorder],
        accepted_indices: Sequence[int],
    ) -> str | None:
        if not cfg.visualization_enabled:
            return None
        if not cfg.visualization_stabilization_picture:
            self._log_phase("stabilization picture summary disabled")
            return None
        pic_num = int(cfg.visualization_stabilization_picture_num)
        if pic_num <= 0:
            self._log_phase("stabilization picture summary skipped: visualization_stabilization_picture_num <= 0")
            return None
        env_by_index = {int(env["index"]): env for env in envs}
        selected = [int(index) for index in accepted_indices if int(index) in env_by_index][:pic_num]
        if not selected:
            self._log_phase("stabilization picture summary skipped: no accepted contact candidates")
            return None
        frames: list[np.ndarray] = []
        video_root = Path(cfg.visualization_video_dir or (Path(cfg.debug_dir or ".") / "videos"))
        for index in selected:
            recorder = recorders.get(index)
            if recorder is None:
                recorder = self._create_visual_recorder(
                    cfg=cfg,
                    env=env_by_index[index],
                    video_root=video_root,
                )
                recorders[index] = recorder
                self._log_phase(f"stabilization picture camera created for accepted candidate index={index}")
            self._render_visual_frame()
            frame = recorder.read_frame()
            if frame is None:
                self._log_phase(f"stabilization picture candidate skipped: no camera frame index={index}")
                continue
            frames.append(frame)
        if not frames:
            self._log_phase(
                "stabilization picture summary skipped: no camera frames available "
                f"accepted={selected}"
            )
            return None
        root = Path(cfg.visualization_picture_dir or cfg.visualization_video_dir or cfg.debug_dir or ".")
        path = root / "contact_cases.ppm"
        _write_ppm(path, _concat_frames_horizontally(frames))
        self._log_phase(
            "stabilization picture summary saved "
            f"path={path} count={len(frames)} selected={selected[:len(frames)]}"
        )
        return str(path)

    def _object_tool_unsigned_distance_min(
        self,
        *,
        object_mesh: _MeshData,
        tool_mesh: _MeshData,
        object_rotation: np.ndarray,
        object_translation: np.ndarray,
        tool_rotation: np.ndarray,
        tool_translation: np.ndarray,
    ) -> float:
        object_points = _transform_points(object_mesh.points, object_rotation, object_translation)
        tool_points = _transform_points(tool_mesh.points, tool_rotation, tool_translation)
        return _unsigned_pointcloud_distance_min(object_points, tool_points)

    def run_batch(
        self,
        *,
        candidates: Mapping[str, torch.Tensor],
        physical_props: Mapping[str, torch.Tensor],
        cfg: PhysicsRunConfig,
        commanded_tool_delta_pose9d_E: torch.Tensor,
    ) -> Sequence[IsaacCandidateResult]:
        n = int(candidates["tool_translation_E"].shape[0])
        if cfg.visualization_enabled and not self.headless:
            raise ValueError("Contact visualization uses headless cameras only; interactive windows are not supported.")
        if not cfg.object_mesh_path or not cfg.tool_mesh_path:
            return [
                self._failure_result(status="missing_asset_paths", candidate_index=index)
                for index in range(n)
            ]

        self._enable_cameras = bool(
            cfg.visualization_enabled
            and (cfg.visualization_stabilization_picture or cfg.visualization_postcontact_video)
        )
        self._log_phase(
            "batch visualization config "
            f"enabled={cfg.visualization_enabled} "
            f"stabilization_picture={cfg.visualization_stabilization_picture} "
            f"stabilization_picture_num={cfg.visualization_stabilization_picture_num} "
            f"postcontact_video={cfg.visualization_postcontact_video} "
            f"postcontact_video_num={cfg.visualization_postcontact_video_num} "
            f"video_dir={cfg.visualization_video_dir} picture_dir={cfg.visualization_picture_dir}"
        )
        self._camera_resolution = (
            max(1, int(cfg.visualization_video_width)),
            max(1, int(cfg.visualization_video_height)),
        )
        phase_start = time.monotonic()
        self._ensure_app()
        self._log_elapsed("batch app ready", phase_start)
        self._log_phase(f"batch stage create candidates={n}")
        phase_start = time.monotonic()
        stage = self._new_stage()
        if self._enable_cameras:
            self._define_visual_lighting(stage)
        self._log_elapsed("batch stage create", phase_start)
        self._log_phase("batch mesh load")
        phase_start = time.monotonic()
        object_mesh = _load_centered_mesh(cfg.object_mesh_path, cfg.object_scale)
        tool_mesh = _load_centered_mesh(cfg.tool_mesh_path, cfg.tool_scale_xyz)
        self._log_elapsed("batch mesh load", phase_start)
        spacing = float(cfg.env_spacing)
        if spacing <= 0.0:
            raise ValueError(f"Contact sim env_spacing must be positive, got {spacing}")
        self._log_phase(f"batch env define start candidates={n} spacing={spacing:.4f}")

        results: list[IsaacCandidateResult | None] = [None] * n
        envs: list[dict[str, Any]] = []
        phase_start = time.monotonic()
        for index in range(n):
            try:
                candidate = {key: candidates[key][index].detach().cpu() for key in candidates}
                props = {key: physical_props[key][index].detach().cpu() for key in physical_props}
                object_rotation_E = _tensor_np(candidate["object_rotation_E"], (3, 3), "object_rotation_E")
                object_center_E = _tensor_np(candidate["object_bbox_center_E"], (3,), "object_bbox_center_E")
                tool_translation_E = _tensor_np(candidate["tool_translation_E"], (3,), "tool_translation_E")
                tool_rotation_E = _tensor_np(candidate["tool_rotation_E"], (3, 3), "tool_rotation_E")
                contact_point_E = _tensor_np(candidate["contact_point_E"], (3,), "contact_point_E")
                if not all(
                    np.isfinite(np.asarray(value, dtype=np.float64)).all()
                    for value in (
                        object_rotation_E,
                        object_center_E,
                        tool_translation_E,
                        tool_rotation_E,
                        contact_point_E,
                    )
                ):
                    results[index] = self._failure_result(status="non_finite_input", candidate_index=index)
                    continue
                if not all(
                    np.isfinite(float(props[key]))
                    for key in (
                        "object_mass",
                        "tool_mass",
                        "object_friction",
                        "tool_friction",
                        "ground_friction",
                    )
                ):
                    results[index] = self._failure_result(status="non_finite_physical_props", candidate_index=index)
                    continue

                offset = _env_offset(index, n, spacing)
                env_path = f"/World/Env_{index:06d}"
                object_path = f"{env_path}/Object"
                tool_path = f"{env_path}/Tool"
                self._define_env_root(stage, env_path)

                object_center_world = object_center_E + offset
                tool_center_world = tool_translation_E + offset
                object_friction = float(props["object_friction"])
                tool_friction = float(props["tool_friction"])
                ground_friction = float(props["ground_friction"])

                _ground_prim, ground_material = self._define_ground(
                    stage,
                    ground_friction,
                    prim_path=f"{env_path}/Ground",
                    center=offset + np.array([0.0, 0.0, -0.05], dtype=np.float64),
                )
                object_prim, object_material = self._define_mesh_body(
                    stage,
                    prim_path=object_path,
                    mesh=object_mesh,
                    rotation=object_rotation_E,
                    translation=object_center_world,
                    mass=float(props["object_mass"]),
                    friction=object_friction,
                    kinematic=False,
                )
                tool_prim, tool_material = self._define_mesh_body(
                    stage,
                    prim_path=tool_path,
                    mesh=tool_mesh,
                    rotation=tool_rotation_E,
                    translation=tool_center_world,
                    mass=float(props["tool_mass"]),
                    friction=tool_friction,
                    kinematic=True,
                )
                envs.append(
                    {
                        "index": index,
                        "env_path": env_path,
                        "offset": offset,
                        "object_path": object_path,
                        "tool_path": tool_path,
                        "object_prim": object_prim,
                        "tool_prim": tool_prim,
                        "ground_path": f"{env_path}/Ground",
                        "initial_object_rotation_E": object_rotation_E,
                        "initial_object_center_world": object_center_world,
                        "initial_tool_rotation_E": tool_rotation_E,
                        "initial_tool_center_world": tool_center_world,
                        "contact_point_E": contact_point_E,
                        "commanded_delta": commanded_tool_delta_pose9d_E[index],
                        "materials": {
                            "object": {"mass": float(props["object_mass"]), **object_material},
                            "tool": {"mass": float(props["tool_mass"]), **tool_material},
                            "ground": ground_material,
                        },
                    }
                )
            except Exception as exc:
                results[index] = self._failure_result(
                    status=f"env_define_error:{type(exc).__name__}:{exc}",
                    candidate_index=index,
        )
        self._log_elapsed("batch env define", phase_start)

        if not envs:
            return [
                result
                if result is not None
                else self._failure_result(status="env_define_failed", candidate_index=index)
                for index, result in enumerate(results)
            ]

        timeline_records: list[dict[str, Any]] = []
        phase_start = time.monotonic()
        recorders = self._create_visual_recorders(cfg=cfg, envs=envs)
        self._log_elapsed("batch visual recorder create", phase_start)
        video_path_by_index = {index: str(recorder.path) for index, recorder in recorders.items()}
        self._step(0, reset=True, force_render=bool(recorders))
        self._bind_runtime_handles(envs)
        self._initialize_runtime_bodies(envs)
        if cfg.run_postcontact and int(cfg.t_stabilize) == 0:
            for env in envs:
                object_state = self._get_runtime_body_state(env, "object")
                tool_state = self._get_runtime_body_state(env, "tool")
                pose_errors = {
                    "object_translation": float(
                        np.linalg.norm(object_state.translation - env["initial_object_center_world"])
                    ),
                    "object_rotation": float(
                        np.max(np.abs(object_state.rotation - env["initial_object_rotation_E"]))
                    ),
                    "tool_translation": float(
                        np.linalg.norm(tool_state.translation - env["initial_tool_center_world"])
                    ),
                    "tool_rotation": float(
                        np.max(np.abs(tool_state.rotation - env["initial_tool_rotation_E"]))
                    ),
                }
                if any(value > 1e-4 for value in pose_errors.values()):
                    raise RuntimeError(
                        "Postcontact initial runtime pose does not match success-only stabilized env-frame pose: "
                        f"candidate={int(env['index'])} errors={pose_errors}"
                    )
                initial_distance = self._object_tool_unsigned_distance_min(
                    object_mesh=object_mesh,
                    tool_mesh=tool_mesh,
                    object_rotation=object_state.rotation,
                    object_translation=object_state.translation,
                    tool_rotation=tool_state.rotation,
                    tool_translation=tool_state.translation,
                )
                if initial_distance > float(cfg.unsigned_distance_accept_eps):
                    raise RuntimeError(
                        "Postcontact initial state is not near contact after loading success-only stabilized pose: "
                        f"candidate={int(env['index'])} unsigned_distance={initial_distance:.6g} "
                        f"eps={float(cfg.unsigned_distance_accept_eps):.6g}"
                    )
        self._log_phase(f"batch stabilize step active_envs={len(envs)} steps={cfg.t_stabilize}")
        phase_start = time.monotonic()
        self._step_and_capture(
            cfg.t_stabilize,
            recorders=tuple(recorders.values()),
        )
        self._log_elapsed("batch stabilize step", phase_start)

        post_envs: list[dict[str, Any]] = []
        for env in envs:
            index = int(env["index"])
            offset = env["offset"]
            object_path = env["object_path"]
            tool_path = env["tool_path"]
            object_state = self._get_runtime_body_state(env, "object")
            tool_state = self._get_runtime_body_state(env, "tool")
            stable_object_rotation_E = object_state.rotation
            stable_object_center_world = object_state.translation
            stable_tool_rotation_E = tool_state.rotation
            stable_tool_center_world = tool_state.translation
            stable_object_center_E = stable_object_center_world - offset
            stable_tool_center_E = stable_tool_center_world - offset
            stable_unsigned_distance_min = self._object_tool_unsigned_distance_min(
                object_mesh=object_mesh,
                tool_mesh=tool_mesh,
                object_rotation=stable_object_rotation_E,
                object_translation=stable_object_center_world,
                tool_rotation=stable_tool_rotation_E,
                tool_translation=stable_tool_center_world,
            )
            distance_accept_eps = float(cfg.unsigned_distance_accept_eps)
            distance_contact = stable_unsigned_distance_min <= distance_accept_eps
            contact_count = 1 if distance_contact else 0
            contact_force_norm = 0.0
            stabilized = {
                "object_rotation_E": torch.as_tensor(stable_object_rotation_E, dtype=torch.float32),
                "object_bbox_center_E": torch.as_tensor(stable_object_center_E, dtype=torch.float32),
                "tool_translation_E": torch.as_tensor(stable_tool_center_E, dtype=torch.float32),
                "tool_rotation_E": torch.as_tensor(stable_tool_rotation_E, dtype=torch.float32),
                "contact_point_E": torch.as_tensor(env["contact_point_E"], dtype=torch.float32),
            }
            debug_payload: dict[str, Any] = {
                "candidate_index": index,
                "env_path": f"/World/Env_{index:06d}",
                "stabilized_in_contact": contact_count > 0,
                "stabilized_contact_count": contact_count,
                "stabilized_contact_impulse_norm": contact_force_norm,
                "stabilized_contact_force_norm": contact_force_norm,
                "stabilized_unsigned_distance_min": stable_unsigned_distance_min,
                "unsigned_distance_accept_eps": distance_accept_eps,
                "unsigned_distance_method": "runtime_pose_vertex_cloud_min",
                "contact_source": "runtime_unsigned_distance",
                "stabilize_steps": int(cfg.t_stabilize),
                "pose_source": "physx_dynamic_control",
                "object_quaternion_wxyz": object_state.quaternion_wxyz.tolist(),
                "tool_quaternion_wxyz": tool_state.quaternion_wxyz.tolist(),
                "physics_materials": env["materials"],
            }
            if cfg.visualization_enabled:
                debug_payload["visualization_timeline"] = [
                    dict(record)
                    for record in timeline_records
                    if int(record.get("candidate_index", -1)) == index
                ]
            if index in video_path_by_index:
                debug_payload["video_path"] = video_path_by_index[index]
            stabilize_failure = (
                None
                if (distance_contact or not bool(cfg.require_stabilized_contact))
                else "stabilize_unsigned_distance_exceeded"
            )
            if stabilize_failure is not None:
                stage_path, debug_path = self._write_debug_artifacts(
                    cfg=cfg,
                    candidate_index=index,
                    stage=stage,
                    payload={**debug_payload, "status": stabilize_failure},
                )
                results[index] = IsaacCandidateResult(
                    success=False,
                    status=stabilize_failure,
                    stabilize_steps=int(cfg.t_stabilize),
                    stabilized_in_contact=False,
                    stabilized_contact_count=0,
                    stabilized_contact_impulse_norm=0.0,
                    stabilized_unsigned_distance_min=stable_unsigned_distance_min,
                    stabilized=stabilized,
                    stage_usd_path=stage_path,
                    debug_json_path=debug_path,
                    video_path=video_path_by_index.get(index),
                    debug_paths={"stage_usd_path": stage_path or "", "debug_json_path": debug_path or ""},
                    metrics=debug_payload,
                )
                continue

            env["picture_object_rotation"] = stable_object_rotation_E
            env["picture_object_translation"] = stable_object_center_world
            env["picture_tool_rotation"] = stable_tool_rotation_E
            env["picture_tool_translation"] = stable_tool_center_world
            if not cfg.run_postcontact:
                debug_payload.update({"postcontact_steps": 0, "status": "stabilized"})
                stage_path, debug_path = self._write_debug_artifacts(
                    cfg=cfg,
                    candidate_index=index,
                    stage=stage,
                    payload=debug_payload,
                )
                results[index] = IsaacCandidateResult(
                    success=True,
                    status="stabilized",
                    stabilize_steps=int(cfg.t_stabilize),
                    stabilized_in_contact=True,
                    stabilized_contact_count=contact_count,
                    stabilized_contact_impulse_norm=contact_force_norm,
                    stabilized_unsigned_distance_min=stable_unsigned_distance_min,
                    post_tool_delta_pose9d_E=env["commanded_delta"],
                    post_tool_achieved_delta_pose9d_E=torch.zeros(9, dtype=torch.float32),
                    post_object_delta_pose9d_E=torch.zeros(9, dtype=torch.float32),
                    postcontact_steps=0,
                    stabilized=stabilized,
                    stage_usd_path=stage_path,
                    debug_json_path=debug_path,
                    video_path=video_path_by_index.get(index),
                    debug_paths={"stage_usd_path": stage_path or "", "debug_json_path": debug_path or ""},
                    metrics=debug_payload,
                )
                continue

            post_envs.append(
                {
                    **env,
                    "stable_object_rotation_E": stable_object_rotation_E,
                    "stable_object_center_E": stable_object_center_E,
                    "stable_object_center_world": stable_object_center_world,
                    "stable_tool_rotation_E": stable_tool_rotation_E,
                    "stable_tool_translation_E": stable_tool_center_E,
                    "stabilized": stabilized,
                    "stabilized_contact_count": contact_count,
                    "stabilized_contact_impulse_norm": contact_force_norm,
                    "stabilized_unsigned_distance_min": stable_unsigned_distance_min,
                    "debug_payload": debug_payload,
                }
            )

        stabilization_picture_path = self._write_contact_picture_summary(
            cfg=cfg,
            envs=envs,
            recorders=recorders,
            accepted_indices=[
                int(env["index"])
                for env in envs
                if "picture_object_rotation" in env and results[int(env["index"])] is None
            ]
            + [
                index
                for index, result in enumerate(results)
                if result is not None and bool(result.success)
            ],
        )
        if stabilization_picture_path:
            for result in results:
                if result is not None and bool(result.success):
                    result.snapshot_paths = [*result.snapshot_paths, stabilization_picture_path]
                    result.metrics = {**dict(result.metrics), "stabilization_picture_path": stabilization_picture_path}

        post_steps = int(cfg.t_postcontact)
        if cfg.run_postcontact and post_steps <= 0:
            for env in post_envs:
                index = int(env["index"])
                debug_payload = dict(env["debug_payload"])
                debug_payload.update({"postcontact_steps": 0, "status": "complete"})
                stage_path, debug_path = self._write_debug_artifacts(
                    cfg=cfg,
                    candidate_index=index,
                    stage=stage,
                    payload=debug_payload,
                )
                results[index] = IsaacCandidateResult(
                    success=True,
                    status="complete",
                    stabilize_steps=int(cfg.t_stabilize),
                    stabilized_in_contact=True,
                    stabilized_contact_count=int(env["stabilized_contact_count"]),
                    stabilized_contact_impulse_norm=float(env["stabilized_contact_impulse_norm"]),
                    stabilized_unsigned_distance_min=float(env["stabilized_unsigned_distance_min"]),
                    stabilized=env["stabilized"],
                    stage_usd_path=stage_path,
                    debug_json_path=debug_path,
                    video_path=video_path_by_index.get(index),
                    debug_paths={"stage_usd_path": stage_path or "", "debug_json_path": debug_path or ""},
                    metrics=debug_payload,
                )
            post_envs = []

        if post_envs:
            self._log_phase(f"batch postcontact step active_envs={len(post_envs)} steps={post_steps}")
            phase_start = time.monotonic()
            post_state: list[dict[str, Any]] = []
            for env in post_envs:
                delta_t_E = env["commanded_delta"][:3].detach().cpu().numpy().astype(np.float64)
                delta_R_E = _rotation_from_pose9d(env["commanded_delta"])
                delta_axis, delta_angle = _axis_angle_from_rotation(delta_R_E)
                target_tool_translation_E = env["stable_tool_translation_E"] + delta_t_E
                target_tool_rotation_E = delta_R_E @ env["stable_tool_rotation_E"]
                env.update(
                    {
                        "delta_t_E": delta_t_E,
                        "delta_axis": delta_axis,
                        "delta_angle": delta_angle,
                        "target_tool_translation_E": target_tool_translation_E,
                        "target_tool_rotation_E": target_tool_rotation_E,
                    }
                )
                post_state.append(env)
            phase_start = time.monotonic()
            active_recorders = self._start_postcontact_video_recorders(
                cfg=cfg,
                recorders=recorders,
                active_indices={int(env["index"]) for env in post_state},
            )
            self._log_elapsed("batch postcontact video recorder start", phase_start)
            video_path_by_index = {index: str(recorder.path) for index, recorder in active_recorders.items()}
            if active_recorders:
                self._step(0, force_render=True)
            for step_index in range(post_steps):
                alpha = float(step_index + 1) / float(post_steps)
                for env in post_state:
                    interpolated_delta_R_E = _rotation_from_axis_angle(
                        env["delta_axis"],
                        env["delta_angle"] * alpha,
                    )
                    current_tool_translation_E = env["stable_tool_translation_E"] + alpha * env["delta_t_E"]
                    current_tool_rotation_E = interpolated_delta_R_E @ env["stable_tool_rotation_E"]
                    current_tool_center_world = current_tool_translation_E + env["offset"]
                    self._set_runtime_body_state(
                        env,
                        "tool",
                        rotation=current_tool_rotation_E,
                        translation=current_tool_center_world,
                        set_velocity=False,
                    )
                self._step_and_capture(
                    1,
                    recorders=tuple(active_recorders.values()),
                )
            self._log_elapsed("batch postcontact step", phase_start)

            for env in post_state:
                index = int(env["index"])
                object_state = self._get_runtime_body_state(env, "object")
                tool_state = self._get_runtime_body_state(env, "tool")
                final_object_rotation_E = object_state.rotation
                final_object_center_world = object_state.translation
                final_tool_rotation_E = tool_state.rotation
                final_tool_center_world = tool_state.translation
                final_object_center_E = final_object_center_world - env["offset"]
                final_tool_center_E = final_tool_center_world - env["offset"]
                delta_object_t_E = final_object_center_E - env["stable_object_center_E"]
                delta_object_R_E = final_object_rotation_E @ env["stable_object_rotation_E"].T
                achieved_delta_t_E = final_tool_center_E - env["stable_tool_translation_E"]
                achieved_delta_R_E = final_tool_rotation_E @ env["stable_tool_rotation_E"].T
                post_object_delta = _pose9d_from_transform(delta_object_t_E, delta_object_R_E)
                post_tool_achieved_delta = _pose9d_from_transform(achieved_delta_t_E, achieved_delta_R_E)
                debug_payload = dict(env["debug_payload"])
                debug_payload.update(
                    {
                        "postcontact_steps": post_steps,
                        "pose_source": "physx_dynamic_control",
                        "post_tool_delta_pose9d_E": env["commanded_delta"].detach().cpu().tolist(),
                        "post_tool_achieved_delta_pose9d_E": post_tool_achieved_delta.tolist(),
                        "post_object_delta_pose9d_E": post_object_delta.tolist(),
                        "post_object_rotation_E": final_object_rotation_E.tolist(),
                        "post_object_bbox_center_E": final_object_center_E.tolist(),
                        "post_tool_rotation_E": final_tool_rotation_E.tolist(),
                        "post_tool_translation_E": final_tool_center_E.tolist(),
                    }
                )
                if cfg.visualization_enabled:
                    debug_payload["visualization_timeline"] = [
                        dict(record)
                        for record in timeline_records
                        if int(record.get("candidate_index", -1)) == index
                    ]
                if index in video_path_by_index:
                    debug_payload["video_path"] = video_path_by_index[index]
                final_outputs_finite = (
                    np.isfinite(final_object_rotation_E).all()
                    and np.isfinite(final_object_center_E).all()
                    and np.isfinite(final_tool_rotation_E).all()
                    and np.isfinite(final_tool_center_E).all()
                    and bool(torch.isfinite(post_tool_achieved_delta).all())
                    and bool(torch.isfinite(post_object_delta).all())
                )
                post_failure = None if final_outputs_finite else "postcontact_non_finite_delta"
                status = post_failure or "complete"
                stage_path, debug_path = self._write_debug_artifacts(
                    cfg=cfg,
                    candidate_index=index,
                    stage=stage,
                    payload={**debug_payload, "status": status},
                )
                results[index] = IsaacCandidateResult(
                    success=post_failure is None,
                    status=status,
                    stabilize_steps=int(cfg.t_stabilize),
                    stabilized_in_contact=True,
                    stabilized_contact_count=int(env["stabilized_contact_count"]),
                    stabilized_contact_impulse_norm=float(env["stabilized_contact_impulse_norm"]),
                    stabilized_unsigned_distance_min=float(env["stabilized_unsigned_distance_min"]),
                    post_tool_delta_pose9d_E=env["commanded_delta"],
                    post_tool_achieved_delta_pose9d_E=post_tool_achieved_delta,
                    post_object_delta_pose9d_E=post_object_delta,
                    postcontact_steps=post_steps,
                    stabilized=env["stabilized"],
                    stage_usd_path=stage_path,
                    debug_json_path=debug_path,
                    video_path=video_path_by_index.get(index),
                    debug_paths={"stage_usd_path": stage_path or "", "debug_json_path": debug_path or ""},
                    metrics=debug_payload,
                )

        if stabilization_picture_path:
            for result in results:
                if result is not None and bool(result.success):
                    result.snapshot_paths = [*result.snapshot_paths, stabilization_picture_path]
                    result.metrics = {
                        **dict(result.metrics),
                        "stabilization_picture_path": stabilization_picture_path,
                    }

        phase_start = time.monotonic()
        video_metadata = self._close_video_recorders(recorders)
        self._log_elapsed("batch video recorder close", phase_start)
        for index, metadata in video_metadata.items():
            result = results[int(index)]
            if result is not None:
                result.metrics = {**dict(result.metrics), "video": dict(metadata)}
        self._log_phase("batch metrics read")
        return [
            result
            if result is not None
            else self._failure_result(status="batch_result_missing", candidate_index=index)
            for index, result in enumerate(results)
        ]

    def run_candidate(
        self,
        *,
        candidate: Mapping[str, torch.Tensor],
        physical_props: Mapping[str, torch.Tensor],
        cfg: PhysicsRunConfig,
        commanded_tool_delta_pose9d_E: torch.Tensor,
        candidate_index: int,
    ) -> IsaacCandidateResult:
        raise NotImplementedError(
            "Isaac contact simulation is batch-only: use run_batch so all candidates "
            "reside in separate envs in the same world and step together."
        )
