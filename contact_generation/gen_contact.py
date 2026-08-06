"""Geometry candidate generation for contact_pt_env_v1 datasets.

This module owns only the approximate geometric contact step:
object pose sampling/grounding, bbox-center canonicalization, tool/object anchor
sampling, B anchor pairs, shared M rotations per chunk, floor checks, and
penetration checks.  Torch/Kaolin/trimesh are imported lazily inside execution
paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from configs.config_contact_gen import (
    CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION,
    CONTACT_GEOMETRY_BBOX_TRANSLATION_NEAREST,
    CONTACT_GEOMETRY_INTERSECTING_ANCHORS,
    CONTACT_GEOMETRY_TANGENT_GAUSSIAN,
    PENETRATION_CHECK_BIDIRECTIONAL,
    PENETRATION_CHECK_TOOL_INTO_OBJECT,
    ROTATION_SELECTION_MOST_CAVITY_CENTERED,
    ROTATION_SELECTION_MOST_DOWNWARD,
    ROTATION_SELECTION_RANDOM_LEGAL,
    TOOL_SOURCE_OBJECTS,
    ContactGenCfg,
)
from utils.assets import (
    ToolAssetContractError,
    assert_adjusted_decomposed_mesh_path,
    load_tool_adjusted_entry,
    load_tool_contact_tip_mesh,
    load_tool_head_area,
    load_tool_kinematic_cloud,
    validate_tool_adjusted_entry,
)
from utils.io import hash_json, write_json
from utils.geometry.sdf import signed_distance_points_to_mesh
from utils.geometry.gripper_cavity import finger_hull_halfspaces


def _torch():
    import torch

    return torch


def _torch_f():
    import torch.nn.functional as F

    return F


def _geometry():
    from utils.geometry import (
        apply_pose_about_bbox_center,
        bbox_center_mesh,
        centralize_points_by_bbox,
        load_mesh_tensors,
        sample_surface_points_torch,
    )

    return (
        apply_pose_about_bbox_center,
        bbox_center_mesh,
        centralize_points_by_bbox,
        load_mesh_tensors,
        sample_surface_points_torch,
    )


def _kaolin():
    import kaolin
    import kaolin.metrics.trianglemesh
    import kaolin.ops.mesh

    return kaolin


@dataclass
class GeometryContactConfig:
    object_mesh_path: str
    tool_mesh_path: str
    tools_json_path: str
    object_id: str
    tool_id: str
    config_name: str
    config_hash: str
    output_path: str
    device: str
    seed: int
    B: int
    M: int
    K: int
    sdf_grid_res: int
    sdf_padding: float
    chunk_B: int
    tool_scale_xyz: tuple[float, float, float]
    object_scale_range: tuple[float, float]
    upright_threshold: float = 0.0
    rotation_selection: str = ROTATION_SELECTION_MOST_DOWNWARD
    tool_mesh_contract: str = "adjusted_decomposed_mesh"
    require_tool_tip_anchor: bool = False
    epsilon: float = 2e-3
    floor_eps: float = 0.0
    penetration_eps: float = 5e-4
    penetration_check_mode: str = PENETRATION_CHECK_TOOL_INTO_OBJECT
    contact_geometry_mode: str = CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION
    rejection_refill: bool = False
    rejection_max_rounds: int = 1
    tangent_translation_noise_std: float = 0.002
    tangent_rotation_noise_std_rad: float = 0.01
    rejection_apply_tangent_gaussian: bool = False
    precompute_convex_union_labels: bool = False
    precompute_mesh_sdf: bool = False
    visualization_enabled: bool = False

    @classmethod
    def from_contact_cfg(
        cls,
        *,
        contact_cfg: ContactGenCfg,
        object_mesh_path: str,
        tool_mesh_path: str,
        tools_json_path: str,
        object_id: str,
        tool_id: str,
        config_name: str,
        config_hash: str,
        output_path: str,
        device: str,
        seed: int,
        tool_scale_xyz: tuple[float, float, float],
    ) -> "GeometryContactConfig":
        return cls(
            object_mesh_path=object_mesh_path,
            tool_mesh_path=tool_mesh_path,
            tools_json_path=tools_json_path,
            object_id=object_id,
            tool_id=tool_id,
            config_name=config_name,
            config_hash=config_hash,
            output_path=output_path,
            device=device,
            seed=seed,
            B=contact_cfg.B,
            M=contact_cfg.M,
            K=contact_cfg.num_surface_pts,
            sdf_grid_res=contact_cfg.sdf_grid_res,
            sdf_padding=contact_cfg.sdf_padding,
            chunk_B=contact_cfg.chunk_B,
            tool_scale_xyz=tool_scale_xyz,
            object_scale_range=tuple(contact_cfg.object_scale_range),
            upright_threshold=contact_cfg.upright_threshold,
            rotation_selection=contact_cfg.rotation_selection,
            tool_mesh_contract=(
                "object_mesh"
                if contact_cfg.tool_source == TOOL_SOURCE_OBJECTS
                else "adjusted_decomposed_mesh"
            ),
            require_tool_tip_anchor=(
                bool(contact_cfg.require_tool_tip_anchor)
                and contact_cfg.tool_source != TOOL_SOURCE_OBJECTS
            ),
            epsilon=contact_cfg.epsilon,
            floor_eps=contact_cfg.floor_eps,
            penetration_eps=contact_cfg.penetration_eps,
            penetration_check_mode=contact_cfg.penetration_check_mode,
            contact_geometry_mode=contact_cfg.contact_geometry_mode,
            rejection_refill=bool(contact_cfg.rejection_refill),
            rejection_max_rounds=int(contact_cfg.rejection_max_rounds),
            tangent_translation_noise_std=float(
                contact_cfg.tangent_translation_noise_std
            ),
            tangent_rotation_noise_std_rad=float(
                contact_cfg.tangent_rotation_noise_std_rad
            ),
            rejection_apply_tangent_gaussian=bool(
                contact_cfg.rejection_apply_tangent_gaussian
            ),
            precompute_convex_union_labels=bool(
                contact_cfg.precompute_convex_union_labels
            ),
            precompute_mesh_sdf=bool(contact_cfg.precompute_mesh_sdf),
            visualization_enabled=bool(contact_cfg.visualization.enabled),
        )


@dataclass
class GeometryCandidateBatch:
    candidates: Mapping[str, Any]
    object_scale: float
    tool_scale_xyz: Any
    tool_head_area_aabb_norm: Any
    object_bbox_center_M: Any
    object_bbox_extent_M: Any
    tool_bbox_center_M: Any
    tool_bbox_extent_M: Any
    object_points_O: Any
    tool_points_T: Any
    contact_normal_E: Any
    tool_point_inside_object: Any | None
    object_point_inside_tool: Any | None
    tool_point_object_signed_sdf: Any | None
    object_point_tool_signed_sdf: Any | None
    source_candidate_index: Any
    debug_metrics: dict[str, Any]

    @property
    def num_candidates(self) -> int:
        return int(self.candidates["tool_translation_E"].shape[0])


def random_rotation_matrices(n: int, device: str):
    torch = _torch()
    if n <= 0:
        return torch.zeros(0, 3, 3, device=device)
    H = torch.randn(n, 3, 3, device=device)
    Q, R_ = torch.linalg.qr(H)
    signs = torch.sign(torch.diagonal(R_, dim1=-2, dim2=-1))
    Q = Q * signs.unsqueeze(-2)
    det = torch.det(Q)
    Q[det < 0] *= -1
    return Q


def sample_upright_rotations(n: int, device: str, upright_threshold: float):
    torch = _torch()
    collected = []
    needed = int(n)
    while needed > 0:
        cands = random_rotation_matrices(max(needed * 4, 16), device)
        good = cands[:, 2, 2] <= float(upright_threshold)
        cands = cands[good]
        if int(cands.shape[0]) == 0:
            continue
        take = min(int(cands.shape[0]), needed)
        collected.append(cands[:take])
        needed -= take
    return torch.cat(collected, dim=0)[:n]


def sample_object_pose_and_ground(vertices_O, faces):
    """Rotate object around bbox-centered object frame and ground it in env frame."""

    torch = _torch()
    apply_pose_about_bbox_center, bbox_center_mesh, *_ = _geometry()
    device = vertices_O.device
    rotation = random_rotation_matrices(1, str(device)).squeeze(0)
    bbox_center = bbox_center_mesh(vertices_O)[0]
    rotated = apply_pose_about_bbox_center(
        vertices_O,
        rotation,
        torch.zeros(3, dtype=vertices_O.dtype, device=device),
        bbox_center=bbox_center,
    )
    object_bbox_center_E = torch.zeros(3, dtype=vertices_O.dtype, device=device)
    object_bbox_center_E[2] = -rotated[:, 2].min()
    return rotated + object_bbox_center_E, rotation, object_bbox_center_E


def build_sdf_grid(obj_verts, obj_faces, grid_res: int, padding: float, device: str, chunk: int = 65536):
    """Build a signed SDF grid with Kaolin. Imported and executed lazily."""

    torch = _torch()
    kaolin = _kaolin()
    R = int(grid_res)
    bbox_min = obj_verts.min(dim=0).values - float(padding)
    bbox_max = obj_verts.max(dim=0).values + float(padding)

    xs = torch.linspace(bbox_min[0].item(), bbox_max[0].item(), R, device=device)
    ys = torch.linspace(bbox_min[1].item(), bbox_max[1].item(), R, device=device)
    zs = torch.linspace(bbox_min[2].item(), bbox_max[2].item(), R, device=device)
    xg, yg, zg = torch.meshgrid(xs, ys, zs, indexing="ij")
    pts = torch.stack([xg, yg, zg], dim=-1).reshape(-1, 3)
    n_pts = int(pts.shape[0])

    face_verts_1 = kaolin.ops.mesh.index_vertices_by_faces(obj_verts.unsqueeze(0), obj_faces)
    dist_vals = torch.empty(n_pts, dtype=torch.float32, device=device)
    inside_vals = torch.empty(n_pts, dtype=torch.bool, device=device)

    for start in range(0, n_pts, chunk):
        end = min(start + chunk, n_pts)
        q_pts = pts[start:end].unsqueeze(0)
        sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            q_pts.contiguous(), face_verts_1
        )
        dist_vals[start:end] = sq_dist.squeeze(0).clamp(min=0).sqrt()
        inside_vals[start:end] = kaolin.ops.mesh.check_sign(
            obj_verts.unsqueeze(0), obj_faces, q_pts
        ).squeeze(0)

    sdf_vals = torch.where(inside_vals, -dist_vals, dist_vals)
    sdf_dhw = sdf_vals.reshape(R, R, R).permute(2, 1, 0)
    return sdf_dhw.unsqueeze(0).unsqueeze(0), bbox_min, bbox_max


def query_sdf_grid(points_E, sdf_grid, bbox_min, bbox_max):
    F = _torch_f()
    span = bbox_max - bbox_min
    norm = 2.0 * (points_E - bbox_min) / span - 1.0
    grid = norm.view(1, 1, 1, -1, 3)
    out = F.grid_sample(
        sdf_grid,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return out.view(-1)


def _connected_face_components(faces):
    """Return face-index tensors for vertex-connected mesh components."""

    torch = _torch()
    faces_cpu = torch.as_tensor(faces, dtype=torch.long).detach().cpu()
    face_count = int(faces_cpu.shape[0])
    if face_count == 0:
        return []
    parent = list(range(face_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_face_by_vertex: dict[int, int] = {}
    for face_index, triangle in enumerate(faces_cpu.tolist()):
        for vertex_index in triangle:
            previous = first_face_by_vertex.setdefault(vertex_index, face_index)
            union(face_index, previous)

    groups: dict[int, list[int]] = {}
    for face_index in range(face_count):
        groups.setdefault(find(face_index), []).append(face_index)
    device = faces.device if isinstance(faces, torch.Tensor) else "cpu"
    return [
        torch.tensor(indices, dtype=torch.long, device=device)
        for _, indices in sorted(groups.items(), key=lambda item: min(item[1]))
    ]


def points_inside_convex_component_union(
    points,
    vertices,
    faces,
    *,
    chunk_size: int = 65536,
):
    """Point membership in the union of disconnected convex mesh parts."""

    torch = _torch()
    kaolin = _kaolin()
    query = torch.as_tensor(
        points, dtype=vertices.dtype, device=vertices.device
    )
    original_shape = query.shape[:-1]
    query = query.reshape(-1, 3).contiguous()
    inside_union = torch.zeros(
        query.shape[0], dtype=torch.bool, device=query.device
    )
    components = _connected_face_components(faces)
    if not components:
        raise RuntimeError("Convex-union labeling requires non-empty mesh faces")
    step = max(1, int(chunk_size))
    for face_indices in components:
        component_faces_global = faces.index_select(0, face_indices)
        vertex_indices, inverse = torch.unique(
            component_faces_global.reshape(-1),
            sorted=True,
            return_inverse=True,
        )
        component_vertices = vertices.index_select(0, vertex_indices)
        component_faces = inverse.reshape(-1, 3)
        for start in range(0, query.shape[0], step):
            end = min(start + step, query.shape[0])
            if bool(inside_union[start:end].all()):
                continue
            inside = kaolin.ops.mesh.check_sign(
                component_vertices.unsqueeze(0),
                component_faces,
                query[start:end].unsqueeze(0),
            ).squeeze(0)
            inside_union[start:end] |= inside
    return inside_union.reshape(original_shape)


def rejection_sample_candidates(
    P_tool_T,
    object_surface_E,
    cfg: GeometryContactConfig,
    *,
    sdf_grid,
    bbox_min,
    bbox_max,
    object_points_E=None,
    tool_sdf_grid=None,
    tool_bbox_min=None,
    tool_bbox_max=None,
    P_anchor,
    object_center_E=None,
    finger_cavity_halfspaces_T=None,
):
    """Return legal candidates produced from sampled surface-anchor pairs.

    Each chunk samples ``M`` random rotations once, then broadcasts those
    rotations across all anchor pairs in that chunk.  A candidate is accepted
    only when it stays above the floor and does not penetrate the object beyond
    ``cfg.penetration_eps``. Cavity-ranked generation oversamples anchor pairs,
    retains each pair's highest-capture legal orientation, then globally keeps
    the top ``B`` pair winners. Other modes retain the existing one-candidate
    rejection/refill behavior.
    """

    torch = _torch()
    device = cfg.device
    B = int(cfg.B)
    M = int(cfg.M)
    K = int(P_tool_T.shape[0])
    chunk_B = int(cfg.chunk_B)
    rank_by_cavity = (
        cfg.rotation_selection == ROTATION_SELECTION_MOST_CAVITY_CENTERED
    )
    if int(P_anchor.shape[0]) <= 0:
        raise ValueError("Nonpenetrating contact generation requires nonempty tool anchors")
    if rank_by_cavity and (
        cfg.penetration_check_mode != PENETRATION_CHECK_BIDIRECTIONAL
        or object_points_E is None
        or tool_sdf_grid is None
        or tool_bbox_min is None
        or tool_bbox_max is None
        or finger_cavity_halfspaces_T is None
    ):
        raise ValueError(
            "Cavity-ranked contact generation requires bidirectional SDF "
            "rejection, object surface points, a tool SDF grid, and exact "
            "finger-cavity halfspaces"
        )

    R_list = []
    t_list = []
    tool_anchor_list = []
    obj_anchor_list = []
    source_index_list = []
    min_sdf_list = []
    penetration_depth_list = []
    cavity_capture_fraction_list = []
    debug_chunks: list[dict[str, Any]] = []

    accepted_count = 0
    attempted_count = 0
    rounds_run = 0
    max_rounds = int(cfg.rejection_max_rounds) if cfg.rejection_refill else 1
    for round_index in range(max_rounds):
        remaining = B - accepted_count
        if remaining <= 0:
            break
        rounds_run = round_index + 1
        # Do not collapse the refill proposal batch to one or two anchors near
        # completion. Some geometry pairs have a low per-anchor legal rate;
        # repeatedly trying only ``remaining`` candidates made 499/500 tails
        # fail even though legal configurations still existed. Oversample a
        # modest batch and retain only the first candidates needed for exact B.
        if rank_by_cavity:
            round_attempts = 4 * B if round_index == 0 else max(B, 4 * remaining)
        else:
            round_attempts = (
                B
                if round_index == 0
                else max(remaining, min(B, 64))
            )
        round_offset = attempted_count

        for b_start in range(0, round_attempts, chunk_B):
            b_end = min(b_start + chunk_B, round_attempts)
            cb = b_end - b_start
            p_B = P_anchor[
                torch.randint(P_anchor.shape[0], (cb,), device=device)
            ]
            obj_idx = torch.randint(object_surface_E.shape[0], (cb,), device=device)
            p_A = object_surface_E[obj_idx]

            rotations = sample_upright_rotations(M, device, cfg.upright_threshold)
            base_translation = (
                p_A[:, None, :]
                - torch.einsum("mij,bj->bmi", rotations, p_B)
            )
            candidate_rotations = rotations.unsqueeze(0).expand(cb, -1, -1, -1)
            candidate_translation = base_translation
            if cfg.rejection_apply_tangent_gaussian:
                rotation_noise = torch.randn(
                    cb * M,
                    3,
                    dtype=P_tool_T.dtype,
                    device=P_tool_T.device,
                ) * float(cfg.tangent_rotation_noise_std_rad)
                noise_rotation = _axis_angle_vectors_to_matrices(
                    rotation_noise
                ).reshape(cb, M, 3, 3)
                candidate_rotations = torch.matmul(
                    noise_rotation,
                    candidate_rotations,
                )
                candidate_translation = candidate_translation + torch.randn(
                    cb,
                    M,
                    3,
                    dtype=P_tool_T.dtype,
                    device=P_tool_T.device,
                ) * float(cfg.tangent_translation_noise_std)
            points_E = (
                torch.einsum(
                    "bmij,kj->bmki",
                    candidate_rotations,
                    P_tool_T,
                )
                + candidate_translation[:, :, None, :]
            )

            floor_ok = points_E[..., 2].min(dim=-1).values >= -float(cfg.floor_eps)
            sdf = query_sdf_grid(
                points_E.reshape(-1, 3), sdf_grid, bbox_min, bbox_max
            ).reshape(cb, M, K)
            min_sdf = sdf.amin(dim=-1)
            reverse_sdf = None
            if cfg.penetration_check_mode == PENETRATION_CHECK_BIDIRECTIONAL:
                if (
                    object_points_E is None
                    or tool_sdf_grid is None
                    or tool_bbox_min is None
                    or tool_bbox_max is None
                ):
                    raise ValueError(
                        "Bidirectional penetration rejection requires object "
                        "surface points and a tool SDF grid"
                    )
                object_delta_E = (
                    object_points_E[None, None, :, :]
                    - candidate_translation[:, :, None, :]
                )
                object_points_T = torch.einsum(
                    "bmji,bmkj->bmki",
                    candidate_rotations,
                    object_delta_E,
                )
                reverse_sdf = query_sdf_grid(
                    object_points_T.reshape(-1, 3),
                    tool_sdf_grid,
                    tool_bbox_min,
                    tool_bbox_max,
                ).reshape(cb, M, int(object_points_E.shape[0]))
                min_sdf = torch.minimum(
                    min_sdf, reverse_sdf.amin(dim=-1)
                )
            elif (
                cfg.penetration_check_mode
                != PENETRATION_CHECK_TOOL_INTO_OBJECT
            ):
                raise ValueError(
                    "Unsupported penetration_check_mode "
                    f"{cfg.penetration_check_mode!r}"
                )
            penetration_depth = torch.clamp(-min_sdf, min=0.0)
            penetration_ok = penetration_depth <= float(cfg.penetration_eps)
            valid = floor_ok & penetration_ok
            chunk_min_z = points_E[..., 2].amin(dim=(-1, -2))
            downward_score = candidate_rotations[:, :, 2, 2]
            if cfg.rotation_selection == ROTATION_SELECTION_MOST_DOWNWARD:
                score = downward_score.clone()
            elif cfg.rotation_selection == ROTATION_SELECTION_RANDOM_LEGAL:
                score = torch.rand(cb, M, device=device)
            elif rank_by_cavity:
                inside_cavity = torch.ones(
                    object_points_T.shape[:-1],
                    dtype=torch.bool,
                    device=device,
                )
                for plane in finger_cavity_halfspaces_T:
                    plane_value = torch.einsum(
                        "bmki,i->bmk",
                        object_points_T,
                        plane[:3],
                    ) + plane[3]
                    inside_cavity &= plane_value <= 1.0e-7
                outside_gripper_material = reverse_sdf >= 0.0
                cavity_capture_fraction = (
                    inside_cavity & outside_gripper_material
                ).to(dtype=P_tool_T.dtype).mean(dim=-1)
                # argmin below chooses the maximum captured object-surface
                # fraction for each anchor pair.
                score = -cavity_capture_fraction
            else:
                cavity_capture_fraction = None
                raise ValueError(
                    "Unsupported rotation_selection "
                    f"'{cfg.rotation_selection}'. Expected "
                    f"'{ROTATION_SELECTION_MOST_DOWNWARD}' or "
                    f"'{ROTATION_SELECTION_RANDOM_LEGAL}' or "
                    f"'{ROTATION_SELECTION_MOST_CAVITY_CENTERED}'."
                )
            score[~valid] = float("inf")
            best_m = score.argmin(dim=-1)
            pair_valid = valid[torch.arange(cb, device=device), best_m]
            if not bool(pair_valid.any()):
                if cfg.visualization_enabled:
                    debug_chunks.append(
                        {
                            "round": int(round_index),
                            "b_start": int(b_start),
                            "b_end": int(b_end),
                            "valid_pairs": 0,
                            "min_tool_z": float(chunk_min_z.min().detach().cpu().item()),
                            "min_sdf": float(min_sdf.min().detach().cpu().item()),
                            "penetration_depth_max": float(
                                penetration_depth.max().detach().cpu().item()
                            ),
                        }
                    )
                continue

            vi = pair_valid.nonzero(as_tuple=False).squeeze(1)
            if not rank_by_cavity:
                vi = vi[: B - accepted_count]
            mi = best_m[vi]
            R_sel = candidate_rotations[vi, mi]
            p_B_sel = p_B[vi]
            p_A_sel = p_A[vi]
            t_sel = candidate_translation[vi, mi]
            min_sdf_sel = min_sdf[vi, mi]
            penetration_sel = penetration_depth[vi, mi]
            if rank_by_cavity:
                cavity_capture_fraction_list.append(
                    cavity_capture_fraction[vi, mi].detach().cpu()
                )
            downward_score_sel = downward_score[vi, mi]
            selected_source_indices = (
                torch.arange(cb, device=device)[vi] + round_offset + b_start
            )

            R_list.append(R_sel.detach().cpu())
            t_list.append(t_sel.detach().cpu())
            tool_anchor_list.append(p_B_sel.detach().cpu())
            obj_anchor_list.append(p_A_sel.detach().cpu())
            source_index_list.append(selected_source_indices.detach().cpu())
            min_sdf_list.append(min_sdf_sel.detach().cpu())
            penetration_depth_list.append(penetration_sel.detach().cpu())
            accepted_count += int(vi.numel())
            if cfg.visualization_enabled:
                debug_chunks.append(
                    {
                        "round": int(round_index),
                        "b_start": int(b_start),
                        "b_end": int(b_end),
                        "valid_pairs": int(pair_valid.sum().detach().cpu().item()),
                        "selected_rotation_indices": mi.detach().cpu().tolist(),
                        "selected_source_indices": selected_source_indices.detach().cpu().tolist(),
                        "selected_tool_anchors_T": p_B_sel.detach().cpu().tolist(),
                        "selected_object_anchors_E": p_A_sel.detach().cpu().tolist(),
                        "selected_downward_scores": downward_score_sel.detach().cpu().tolist(),
                        "min_tool_z": float(chunk_min_z.min().detach().cpu().item()),
                        "selected_min_sdf": min_sdf_sel.detach().cpu().tolist(),
                        "selected_penetration_depth": penetration_sel.detach().cpu().tolist(),
                        "chunk_min_sdf": float(min_sdf.min().detach().cpu().item()),
                        "chunk_penetration_depth_max": float(
                            penetration_depth.max().detach().cpu().item()
                        ),
                    }
                )
            if accepted_count >= B and not rank_by_cavity:
                break
        attempted_count += round_attempts

    incomplete = (
        accepted_count < B
        if rank_by_cavity
        else cfg.rejection_refill and accepted_count != B
    )
    if incomplete:
        raise RuntimeError(
            "Non-penetrating rejection sampler could not fill the required "
            f"candidate batch: accepted={accepted_count} required={B} "
            f"rounds={rounds_run} rotations_per_anchor={M}."
        )

    if not R_list:
        return {
            "tool_rotation_E": torch.zeros(0, 3, 3),
            "tool_translation_E": torch.zeros(0, 3),
            "contact_pt_tool_T": torch.zeros(0, 3),
            "contact_pt_obj_E": torch.zeros(0, 3),
            "source_candidate_index": torch.zeros(0, dtype=torch.int64),
            "initial_min_sdf": torch.zeros(0),
            "initial_penetration_depth": torch.zeros(0),
            "geometry_diagnostics": {
                "chunks": debug_chunks,
                "refill_rounds": rounds_run,
                "attempted_anchor_pairs": attempted_count,
            },
        }

    tool_rotation = torch.cat(R_list, dim=0)
    tool_translation = torch.cat(t_list, dim=0)
    tool_anchor = torch.cat(tool_anchor_list, dim=0)
    object_anchor = torch.cat(obj_anchor_list, dim=0)
    source_index = torch.cat(source_index_list, dim=0).to(dtype=torch.int64)
    selected_min_sdf = torch.cat(min_sdf_list, dim=0)
    selected_penetration = torch.cat(penetration_depth_list, dim=0)
    cavity_capture_fraction = None
    if rank_by_cavity:
        cavity_capture_fraction = torch.cat(cavity_capture_fraction_list, dim=0)
        top_indices = torch.topk(
            cavity_capture_fraction,
            k=B,
            largest=True,
            sorted=True,
        ).indices
        tool_rotation = tool_rotation[top_indices]
        tool_translation = tool_translation[top_indices]
        tool_anchor = tool_anchor[top_indices]
        object_anchor = object_anchor[top_indices]
        source_index = source_index[top_indices]
        selected_min_sdf = selected_min_sdf[top_indices]
        selected_penetration = selected_penetration[top_indices]
        cavity_capture_fraction = cavity_capture_fraction[top_indices]

    return {
        "tool_rotation_E": tool_rotation,
        "tool_translation_E": tool_translation,
        "contact_pt_tool_T": tool_anchor,
        "contact_pt_obj_E": object_anchor,
        "source_candidate_index": source_index,
        "initial_min_sdf": selected_min_sdf,
        "initial_penetration_depth": selected_penetration,
        "cavity_capture_fraction": cavity_capture_fraction,
        "geometry_diagnostics": {
            "chunks": debug_chunks,
            "refill_rounds": rounds_run,
            "attempted_anchor_pairs": attempted_count,
            "legal_anchor_pair_winners": accepted_count,
        },
    }


def intersecting_anchor_sample_candidates(
    P_tool_T,
    object_surface_E,
    cfg: GeometryContactConfig,
    *,
    P_anchor,
):
    """Align random surface anchors under unrestricted SO(3) rotations.

    There is deliberately no floor, stability, or penetration rejection.  The
    coincident surface anchors guarantee contact, while configurations whose
    remaining geometry intersects are retained as contact-positive examples.
    """

    torch = _torch()
    count = int(cfg.B)
    if int(P_anchor.shape[0]) <= 0:
        raise ValueError("Raw contact generation requires nonempty tool anchors")
    tool_indices = torch.randint(P_anchor.shape[0], (count,), device=cfg.device)
    object_indices = torch.randint(object_surface_E.shape[0], (count,), device=cfg.device)
    tool_anchors = P_anchor[tool_indices]
    object_anchors = object_surface_E[object_indices]
    rotations = random_rotation_matrices(count, cfg.device)
    rotated_anchors = torch.einsum("nij,nj->ni", rotations, tool_anchors)
    translations = object_anchors - rotated_anchors
    # Exact signed mesh distances are intentionally deferred to pretraining,
    # where they are the labels.  Avoid building/querying an SDF grid merely
    # for generation diagnostics.
    min_sdf = torch.full((count,), float("nan"), device=cfg.device)
    penetration_depth = torch.full((count,), float("nan"), device=cfg.device)
    return {
        "tool_rotation_E": rotations.detach().cpu(),
        "tool_translation_E": translations.detach().cpu(),
        "contact_pt_tool_T": tool_anchors.detach().cpu(),
        "contact_pt_obj_E": object_anchors.detach().cpu(),
        "source_candidate_index": torch.arange(count, dtype=torch.int64),
        "initial_min_sdf": min_sdf.detach().cpu(),
        "initial_penetration_depth": penetration_depth.detach().cpu(),
        "geometry_diagnostics": {
            "rejection": "none",
            "allows_intersection": True,
        },
    }


def _nearest_surface_pairs(tool_points_E, object_points_E, *, chunk_size: int = 64):
    torch = _torch()
    device = tool_points_E.device
    n = int(tool_points_E.shape[0])
    object_count = int(object_points_E.shape[0])
    min_dist = torch.empty(n, dtype=tool_points_E.dtype, device=device)
    tool_idx = torch.empty(n, dtype=torch.int64, device=device)
    object_idx = torch.empty(n, dtype=torch.int64, device=device)

    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        chunk = tool_points_E[start:end]
        object_chunk = object_points_E.unsqueeze(0).expand(end - start, -1, -1)
        distances = torch.cdist(chunk, object_chunk)
        flat_idx = distances.reshape(end - start, -1).argmin(dim=-1)
        tool_idx[start:end] = flat_idx // object_count
        object_idx[start:end] = flat_idx % object_count
        min_dist[start:end] = distances.reshape(end - start, -1)[
            torch.arange(end - start, device=device), flat_idx
        ]

    tool_points = tool_points_E[torch.arange(n, device=device), tool_idx]
    object_points = object_points_E[object_idx]
    return min_dist, tool_idx, object_idx, tool_points, object_points


def _axis_angle_vectors_to_matrices(axis_angle):
    """Convert a batch of rotation vectors to SO(3) matrices."""

    torch = _torch()
    count = int(axis_angle.shape[0])
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp_min(1e-12)
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        dim=-1,
    ).reshape(count, 3, 3)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand(
        count, -1, -1
    )
    sin = torch.sin(angle).reshape(count, 1, 1)
    cos = torch.cos(angle).reshape(count, 1, 1)
    return eye + sin * skew + (1.0 - cos) * torch.bmm(skew, skew)


def tangent_gaussian_sample_candidates(
    P_tool_T,
    object_points_E,
    cfg: GeometryContactConfig,
    *,
    P_anchor,
):
    """Paper-style UniCORN near-contact placement.

    A randomly oriented tool is initialized away from the object, translated
    by the shortest displacement between the explicitly supplied tool anchors
    and the sampled object surface, and then perturbed by a small zero-mean
    SE(3) Gaussian. ``P_anchor`` is mandatory so callers cannot silently fall
    back from fingertip anchors to the full tool. No floor, stability, or
    penetration rejection is applied after perturbation.
    """

    torch = _torch()
    count = int(cfg.B)
    device = cfg.device
    rotations = random_rotation_matrices(count, device)
    rotated_anchor = torch.einsum("bij,kj->bki", rotations, P_anchor)

    directions = torch.randn(
        count, 3, dtype=P_tool_T.dtype, device=P_tool_T.device
    )
    directions = directions / torch.linalg.norm(
        directions, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    object_center = 0.5 * (
        object_points_E.amin(dim=0) + object_points_E.amax(dim=0)
    )
    object_radius = torch.linalg.norm(
        object_points_E - object_center, dim=-1
    ).max()
    tool_radius = torch.linalg.norm(P_tool_T, dim=-1).max()
    separation = object_radius + tool_radius + max(float(cfg.sdf_padding), 0.01)
    initial_t = object_center.unsqueeze(0) + directions * separation
    initial_points = rotated_anchor + initial_t[:, None, :]

    _, tool_idx, _, nearest_tool_E, nearest_obj_E = _nearest_surface_pairs(
        initial_points,
        object_points_E,
        chunk_size=max(1, min(int(cfg.chunk_B), 64)),
    )
    tangent_delta = nearest_obj_E - nearest_tool_E
    tangent_t = initial_t + tangent_delta

    rotation_noise = torch.randn(
        count, 3, dtype=P_tool_T.dtype, device=P_tool_T.device
    ) * float(cfg.tangent_rotation_noise_std_rad)
    noise_rotation = _axis_angle_vectors_to_matrices(rotation_noise)
    final_rotations = torch.bmm(noise_rotation, rotations)
    translation_noise = torch.randn(
        count, 3, dtype=P_tool_T.dtype, device=P_tool_T.device
    ) * float(cfg.tangent_translation_noise_std)
    final_t = tangent_t + translation_noise
    # Contact labels are computed exactly by the pretraining dataset/model.
    # Building a dense SDF merely for generation diagnostics would dominate
    # the cost of the 500k-case paper dataset, so this no-rejection mode
    # intentionally records signed-distance diagnostics as unavailable.
    min_sdf = torch.full(
        (count,), float("nan"), dtype=P_tool_T.dtype, device=P_tool_T.device
    )
    penetration_depth = torch.full_like(min_sdf, float("nan"))
    return {
        "tool_rotation_E": final_rotations.detach().cpu(),
        "tool_translation_E": final_t.detach().cpu(),
        "contact_pt_tool_T": P_anchor[tool_idx].detach().cpu(),
        "contact_pt_obj_E": nearest_obj_E.detach().cpu(),
        "source_candidate_index": torch.arange(count, dtype=torch.int64),
        "initial_min_sdf": min_sdf.detach().cpu(),
        "initial_penetration_depth": penetration_depth.detach().cpu(),
        "geometry_diagnostics": {
            "placement": "shortest_sampled_surface_displacement_then_se3_gaussian",
            "anchor_source": (
                "contact_tip_mesh"
                if cfg.require_tool_tip_anchor
                else "full_tool_surface"
            ),
            "translation_noise_std": float(
                cfg.tangent_translation_noise_std
            ),
            "rotation_noise_std_rad": float(
                cfg.tangent_rotation_noise_std_rad
            ),
            "penetrating_fraction": None,
        },
    }


def _safe_contact_alpha(points_E, delta_E, cfg: GeometryContactConfig, *, sdf_grid, bbox_min, bbox_max):
    torch = _torch()
    device = points_E.device
    n = int(points_E.shape[0])
    low = torch.zeros(n, dtype=points_E.dtype, device=device)
    high = torch.ones(n, dtype=points_E.dtype, device=device)
    for _ in range(10):
        mid = (low + high) * 0.5
        moved = points_E + delta_E[:, None, :] * mid[:, None, None]
        floor_ok = moved[..., 2].min(dim=-1).values >= -float(cfg.floor_eps)
        sdf = query_sdf_grid(moved.reshape(-1, 3), sdf_grid, bbox_min, bbox_max).reshape(n, -1)
        min_sdf = sdf.amin(dim=-1)
        penetration_ok = torch.clamp(-min_sdf, min=0.0) <= float(cfg.penetration_eps)
        safe = floor_ok & penetration_ok
        low = torch.where(safe, mid, low)
        high = torch.where(safe, high, mid)
    return low


def bbox_translation_nearest_sample_candidates(
    P_tool_T,
    object_points_E,
    object_verts_E,
    cfg: GeometryContactConfig,
    *,
    sdf_grid,
    bbox_min,
    bbox_max,
):
    """Sample legal poses, then translate toward the nearest surface pair."""

    torch = _torch()
    device = cfg.device
    B = int(cfg.B)
    M = int(cfg.M)
    K = int(P_tool_T.shape[0])
    chunk_B = int(cfg.chunk_B)

    R_list = []
    t_list = []
    tool_anchor_list = []
    obj_anchor_list = []
    source_index_list = []
    min_sdf_list = []
    penetration_depth_list = []
    surface_distance_list = []
    contact_alpha_list = []
    debug_chunks: list[dict[str, Any]] = []

    translation_bias = float(cfg.sdf_padding)
    translation_min = object_verts_E.amin(dim=0) - translation_bias
    translation_max = object_verts_E.amax(dim=0) + translation_bias

    for b_start in range(0, B, chunk_B):
        b_end = min(b_start + chunk_B, B)
        cb = b_end - b_start
        translations = torch.rand(cb, 3, device=device) * (
            translation_max - translation_min
        ) + translation_min
        rotations = random_rotation_matrices(M, device)
        points_E = torch.einsum("mij,kj->mki", rotations, P_tool_T).unsqueeze(0) + translations[:, None, None, :]

        floor_ok = points_E[..., 2].min(dim=-1).values >= -float(cfg.floor_eps)
        sdf = query_sdf_grid(points_E.reshape(-1, 3), sdf_grid, bbox_min, bbox_max).reshape(cb, M, K)
        min_sdf = sdf.amin(dim=-1)
        penetration_depth = torch.clamp(-min_sdf, min=0.0)
        penetration_ok = penetration_depth <= float(cfg.penetration_eps)
        valid = floor_ok & penetration_ok
        downward_score = rotations[:, 2, 2].unsqueeze(0).expand(cb, -1)
        if cfg.rotation_selection == ROTATION_SELECTION_MOST_DOWNWARD:
            score = downward_score.clone()
        elif cfg.rotation_selection == ROTATION_SELECTION_RANDOM_LEGAL:
            score = torch.rand(cb, M, device=device)
        else:
            raise ValueError(
                "Unsupported rotation_selection "
                f"'{cfg.rotation_selection}'. Expected "
                f"'{ROTATION_SELECTION_MOST_DOWNWARD}' or '{ROTATION_SELECTION_RANDOM_LEGAL}'."
            )
        score[~valid] = float("inf")
        best_m = score.argmin(dim=-1)
        translation_valid = valid[torch.arange(cb, device=device), best_m]
        if not bool(translation_valid.any()):
            if cfg.visualization_enabled:
                debug_chunks.append(
                    {
                        "b_start": int(b_start),
                        "b_end": int(b_end),
                        "valid_translations": 0,
                        "translation_min": translation_min.detach().cpu().tolist(),
                        "translation_max": translation_max.detach().cpu().tolist(),
                        "chunk_min_sdf": float(min_sdf.min().detach().cpu().item()),
                        "chunk_penetration_depth_max": float(penetration_depth.max().detach().cpu().item()),
                    }
                )
            continue

        vi = translation_valid.nonzero(as_tuple=False).squeeze(1)
        mi = best_m[vi]
        selected_points_E = points_E[vi, mi]
        selected_R = rotations[mi]
        selected_t = translations[vi]
        nearest_dist, _, _, nearest_tool_E, nearest_obj_E = _nearest_surface_pairs(
            selected_points_E,
            object_points_E,
        )
        direction = nearest_obj_E - nearest_tool_E
        direction_norm = torch.linalg.norm(direction, dim=-1).clamp_min(1e-12)
        contact_delta = direction * ((nearest_dist / direction_norm).unsqueeze(-1))
        alpha = _safe_contact_alpha(
            selected_points_E,
            contact_delta,
            cfg,
            sdf_grid=sdf_grid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )
        final_t = selected_t + contact_delta * alpha.unsqueeze(-1)
        final_points_E = selected_points_E + contact_delta[:, None, :] * alpha[:, None, None]
        final_floor_ok = final_points_E[..., 2].min(dim=-1).values >= -float(cfg.floor_eps)
        final_sdf = query_sdf_grid(
            final_points_E.reshape(-1, 3), sdf_grid, bbox_min, bbox_max
        ).reshape(int(final_points_E.shape[0]), K)
        final_min_sdf = final_sdf.amin(dim=-1)
        final_penetration = torch.clamp(-final_min_sdf, min=0.0)
        final_penetration_ok = final_penetration <= float(cfg.penetration_eps)
        final_close_enough = final_min_sdf <= float(cfg.epsilon)
        accepted = final_floor_ok & final_penetration_ok & final_close_enough
        if not bool(accepted.any()):
            if cfg.visualization_enabled:
                debug_chunks.append(
                    {
                        "b_start": int(b_start),
                        "b_end": int(b_end),
                        "valid_translations": int(translation_valid.sum().detach().cpu().item()),
                        "accepted_contacts": 0,
                        "nearest_distance_min": float(nearest_dist.min().detach().cpu().item()),
                        "final_min_sdf_min": float(final_min_sdf.min().detach().cpu().item()),
                        "final_penetration_depth_max": float(final_penetration.max().detach().cpu().item()),
                    }
                )
            continue

        ai = accepted.nonzero(as_tuple=False).squeeze(1)
        accepted_points_E = final_points_E[ai]
        final_dist, final_tool_idx, _, _, final_obj_E = _nearest_surface_pairs(
            accepted_points_E,
            object_points_E,
        )

        R_list.append(selected_R[ai].detach().cpu())
        t_list.append(final_t[ai].detach().cpu())
        tool_anchor_list.append(P_tool_T[final_tool_idx].detach().cpu())
        obj_anchor_list.append(final_obj_E.detach().cpu())
        source_index_list.append((vi[ai] + b_start).detach().cpu())
        min_sdf_list.append(final_min_sdf[ai].detach().cpu())
        penetration_depth_list.append(final_penetration[ai].detach().cpu())
        surface_distance_list.append(final_dist.detach().cpu())
        contact_alpha_list.append(alpha[ai].detach().cpu())
        if cfg.visualization_enabled:
            debug_chunks.append(
                {
                    "b_start": int(b_start),
                    "b_end": int(b_end),
                    "valid_translations": int(translation_valid.sum().detach().cpu().item()),
                    "accepted_contacts": int(accepted.sum().detach().cpu().item()),
                    "translation_min": translation_min.detach().cpu().tolist(),
                    "translation_max": translation_max.detach().cpu().tolist(),
                    "nearest_distance": nearest_dist.detach().cpu().tolist(),
                    "accepted_surface_distance": final_dist.detach().cpu().tolist(),
                    "contact_alpha": alpha[ai].detach().cpu().tolist(),
                    "final_min_sdf": final_min_sdf[ai].detach().cpu().tolist(),
                    "final_penetration_depth": final_penetration[ai].detach().cpu().tolist(),
                }
            )

    if not R_list:
        return {
            "tool_rotation_E": torch.zeros(0, 3, 3),
            "tool_translation_E": torch.zeros(0, 3),
            "contact_pt_tool_T": torch.zeros(0, 3),
            "contact_pt_obj_E": torch.zeros(0, 3),
            "source_candidate_index": torch.zeros(0, dtype=torch.int64),
            "initial_min_sdf": torch.zeros(0),
            "initial_penetration_depth": torch.zeros(0),
            "geometry_diagnostics": {
                "chunks": debug_chunks,
                "translation_bbox_bias": translation_bias,
            },
        }

    return {
        "tool_rotation_E": torch.cat(R_list, dim=0),
        "tool_translation_E": torch.cat(t_list, dim=0),
        "contact_pt_tool_T": torch.cat(tool_anchor_list, dim=0),
        "contact_pt_obj_E": torch.cat(obj_anchor_list, dim=0),
        "source_candidate_index": torch.cat(source_index_list, dim=0).to(dtype=torch.int64),
        "initial_min_sdf": torch.cat(min_sdf_list, dim=0),
        "initial_penetration_depth": torch.cat(penetration_depth_list, dim=0),
        "geometry_diagnostics": {
            "chunks": debug_chunks,
            "translation_bbox_bias": translation_bias,
            "final_surface_distance": torch.cat(surface_distance_list, dim=0).tolist(),
            "contact_alpha": torch.cat(contact_alpha_list, dim=0).tolist(),
        },
    }


def _candidate_fields_from_geometry(geometry, R_obj, object_bbox_center_E):
    torch = _torch()
    R_tool_E = geometry["tool_rotation_E"].to(R_obj.device)
    t_tool_E = geometry["tool_translation_E"].to(R_obj.device)
    contact_pt_obj_E = geometry["contact_pt_obj_E"].to(R_obj.device)
    n = int(R_tool_E.shape[0])
    return {
        "object_rotation_E": R_obj.unsqueeze(0).expand(n, -1, -1).detach().cpu(),
        "object_bbox_center_E": object_bbox_center_E.unsqueeze(0).expand(n, -1).detach().cpu(),
        "tool_translation_E": t_tool_E.detach().cpu(),
        "tool_rotation_E": R_tool_E.detach().cpu(),
        "contact_point_E": contact_pt_obj_E.detach().cpu(),
    }


def generate_contact_candidates(cfg: GeometryContactConfig) -> GeometryCandidateBatch:
    """Generate approximate contact candidates for one object/tool/object-pose."""

    _log(
        "[GEOMETRY] start "
        f"tool={cfg.tool_id} object={cfg.object_id} B={cfg.B} M={cfg.M} device={cfg.device}"
    )
    torch = _torch()
    (
        _apply_pose_about_bbox_center,
        bbox_center_mesh,
        centralize_points_by_bbox,
        load_mesh_tensors,
        sample_surface_points_torch,
    ) = _geometry()
    torch.manual_seed(int(cfg.seed))
    tool_id = cfg.tool_id or Path(cfg.tool_mesh_path).stem
    is_gripper = False
    if cfg.tool_mesh_contract == "adjusted_decomposed_mesh":
        tool_id = cfg.tool_id or assert_adjusted_decomposed_mesh_path(cfg.tool_mesh_path)
        assert_adjusted_decomposed_mesh_path(cfg.tool_mesh_path, tool_id)
        if not cfg.tools_json_path or not Path(cfg.tools_json_path).exists():
            raise ToolAssetContractError(f"tools_adjusted.json is required: {cfg.tools_json_path}")
        adjusted_entry = load_tool_adjusted_entry(cfg.tools_json_path, tool_id)
        validate_tool_adjusted_entry(adjusted_entry, tool_id)
        is_gripper = (
            "source_generated_gripper_id" in adjusted_entry
            or "source_one_dof_gripper_id" in adjusted_entry
        )
        if (
            is_gripper
            and cfg.contact_geometry_mode
            in {
                CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION,
                CONTACT_GEOMETRY_INTERSECTING_ANCHORS,
            }
            and not cfg.require_tool_tip_anchor
        ):
            raise ToolAssetContractError(
                f"Gripper '{tool_id}' requires explicit fingertip contact anchors"
            )
    elif cfg.tool_mesh_contract != "object_mesh":
        raise ValueError(
            "GeometryContactConfig.tool_mesh_contract must be "
            "adjusted_decomposed_mesh or object_mesh"
        )

    _log(
        "[GEOMETRY] loading meshes "
        f"object_mesh={cfg.object_mesh_path} tool_mesh={cfg.tool_mesh_path}"
    )
    obj_verts_raw, obj_faces = load_mesh_tensors(cfg.object_mesh_path, cfg.device)
    tool_verts_raw, tool_faces = load_mesh_tensors(cfg.tool_mesh_path, cfg.device)
    tool_scale_xyz = torch.tensor(cfg.tool_scale_xyz, dtype=torch.float32, device=cfg.device)
    tool_verts_M = tool_verts_raw * tool_scale_xyz.unsqueeze(0)
    object_scale = torch.empty(1, device=cfg.device).uniform_(
        cfg.object_scale_range[0], cfg.object_scale_range[1]
    ).item()
    obj_verts_M = obj_verts_raw * float(object_scale)
    _log(f"[GEOMETRY] sampled object_scale={object_scale:.6f}")

    object_bbox_center, object_bbox_extent = bbox_center_mesh(obj_verts_M)
    tool_bbox_center, tool_bbox_extent = bbox_center_mesh(tool_verts_M)
    obj_verts_O, _, _ = centralize_points_by_bbox(
        obj_verts_M,
        bbox_center=object_bbox_center,
        bbox_extent=object_bbox_extent,
    )
    tool_verts_T, _, _ = centralize_points_by_bbox(
        tool_verts_M,
        bbox_center=tool_bbox_center,
        bbox_extent=tool_bbox_extent,
    )

    object_points_M = sample_surface_points_torch(obj_verts_M, obj_faces, cfg.K)
    object_points_O, _, _ = centralize_points_by_bbox(
        object_points_M,
        bbox_center=object_bbox_center,
        bbox_extent=object_bbox_extent,
    )
    if is_gripper:
        if cfg.K != 512:
            raise ValueError(
                "Cached gripper contact clouds require K=512, "
                f"got K={cfg.K}"
            )
        _, cached_cloud = load_tool_kinematic_cloud(
            cfg.tools_json_path, tool_id
        )
        tool_points_M = cached_cloud.to(
            device=tool_verts_M.device, dtype=tool_verts_M.dtype
        )
    else:
        tool_points_M = sample_surface_points_torch(
            tool_verts_M, tool_faces, cfg.K
        )
    tool_points_T, _, _ = centralize_points_by_bbox(
        tool_points_M,
        bbox_center=tool_bbox_center,
        bbox_extent=tool_bbox_extent,
    )
    _log(f"[GEOMETRY] sampled surface points object={cfg.K} tool={cfg.K}")

    strict_tip_points_T = None
    if cfg.require_tool_tip_anchor:
        tip_mesh_path = load_tool_contact_tip_mesh(
            cfg.tools_json_path,
            cfg.tool_mesh_path,
            tool_id=tool_id,
        )
        _log(
            "[GEOMETRY] loading strict fingertip anchor mesh "
            f"tip_mesh={tip_mesh_path}"
        )
        tip_verts_raw, tip_faces = load_mesh_tensors(
            str(tip_mesh_path), cfg.device
        )
        tip_verts_M = tip_verts_raw * tool_scale_xyz.unsqueeze(0)
        tip_points_M = sample_surface_points_torch(
            tip_verts_M, tip_faces, cfg.K
        )
        strict_tip_points_T = tip_points_M - tool_bbox_center.unsqueeze(0)
        _log(
            "[GEOMETRY] sampled strict fingertip anchors "
            f"count={strict_tip_points_T.shape[0]}"
        )
    finger_cavity_halfspaces_T = None
    if (
        cfg.rotation_selection
        == ROTATION_SELECTION_MOST_CAVITY_CENTERED
    ):
        if cfg.tool_mesh_contract != "adjusted_decomposed_mesh":
            raise ValueError(
                "Cavity-centered rotation selection requires an exact "
                "adjusted generated-gripper mesh"
            )
        cavity_equations = finger_hull_halfspaces(
            cfg.tool_mesh_path,
            scale_xyz=tuple(float(value) for value in cfg.tool_scale_xyz),
            bbox_center=tuple(
                float(value)
                for value in tool_bbox_center.detach().cpu().tolist()
            ),
        )
        finger_cavity_halfspaces_T = torch.as_tensor(
            cavity_equations,
            dtype=tool_points_T.dtype,
            device=tool_points_T.device,
        )
        _log(
            "[GEOMETRY] loaded exact finger-cavity hull "
            f"planes={finger_cavity_halfspaces_T.shape[0]}"
        )

    _log("[GEOMETRY] sampling object pose and grounding")
    object_verts_E, R_obj, object_bbox_center_E = sample_object_pose_and_ground(obj_verts_O, obj_faces)
    object_points_E = torch.einsum("ij,kj->ki", R_obj, object_points_O) + object_bbox_center_E
    sdf_grid = bbox_min = bbox_max = None
    tool_sdf_grid = tool_bbox_min = tool_bbox_max = None
    if cfg.contact_geometry_mode not in {
        CONTACT_GEOMETRY_INTERSECTING_ANCHORS,
        CONTACT_GEOMETRY_TANGENT_GAUSSIAN,
    }:
        _log(
            "[GEOMETRY] building object SDF "
            f"grid_res={cfg.sdf_grid_res} padding={cfg.sdf_padding}"
        )
        sdf_grid, bbox_min, bbox_max = build_sdf_grid(
            object_verts_E,
            obj_faces,
            cfg.sdf_grid_res,
            cfg.sdf_padding,
            cfg.device,
        )
        if (
            cfg.contact_geometry_mode
            == CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION
            and cfg.penetration_check_mode
            == PENETRATION_CHECK_BIDIRECTIONAL
        ):
            _log(
                "[GEOMETRY] building tool SDF for bidirectional "
                f"penetration rejection grid_res={cfg.sdf_grid_res} "
                f"padding={cfg.sdf_padding}"
            )
            (
                tool_sdf_grid,
                tool_bbox_min,
                tool_bbox_max,
            ) = build_sdf_grid(
                tool_verts_T,
                tool_faces,
                cfg.sdf_grid_res,
                cfg.sdf_padding,
                cfg.device,
            )
    head_area_tensor = torch.tensor(
        load_tool_head_area(
            cfg.tools_json_path,
            cfg.tool_mesh_path,
            tool_id=tool_id,
        )
        if cfg.tool_mesh_contract == "adjusted_decomposed_mesh"
        else ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        dtype=torch.float32,
    )

    if cfg.contact_geometry_mode == CONTACT_GEOMETRY_TANGENT_GAUSSIAN:
        contact_anchors = (
            strict_tip_points_T
            if cfg.require_tool_tip_anchor
            else tool_points_T
        )
        _log(
            "[GEOMETRY] paper-style tangent-plus-Gaussian candidates "
            f"count={cfg.B} translation_std={cfg.tangent_translation_noise_std:g} "
            f"rotation_std_rad={cfg.tangent_rotation_noise_std_rad:g} "
            f"anchors={'contact_tip_mesh' if cfg.require_tool_tip_anchor else 'full_tool_surface'}"
        )
        geometry = tangent_gaussian_sample_candidates(
            tool_points_T,
            object_points_E,
            cfg,
            P_anchor=contact_anchors,
        )
    elif cfg.contact_geometry_mode in {
        CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION,
        CONTACT_GEOMETRY_INTERSECTING_ANCHORS,
    }:
        _log(f"[GEOMETRY] sampling object surface anchors count={max(cfg.B * 4, 16384)}")
        object_surface_E = sample_surface_points_torch(object_verts_E, obj_faces, max(cfg.B * 4, 16384))
        if cfg.require_tool_tip_anchor:
            contact_anchors = strict_tip_points_T
            _log(
                "[GEOMETRY] using strict contact_tip_mesh anchors; "
                "full-surface fallback is disabled"
            )
        else:
            contact_anchors = tool_points_T
        if cfg.contact_geometry_mode == CONTACT_GEOMETRY_INTERSECTING_ANCHORS:
            _log(
                "[GEOMETRY] intersecting anchor candidates "
                f"anchor_pairs={cfg.B} rejection=none"
            )
            geometry = intersecting_anchor_sample_candidates(
                tool_points_T,
                object_surface_E,
                cfg,
                P_anchor=contact_anchors,
            )
        else:
            _log(
                "[GEOMETRY] rejection sampling candidates "
                f"anchor_pairs={cfg.B} rotations_per_pair={cfg.M}"
            )
            geometry = rejection_sample_candidates(
                tool_points_T,
                object_surface_E,
                cfg,
                sdf_grid=sdf_grid,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                object_points_E=object_points_E,
                tool_sdf_grid=tool_sdf_grid,
                tool_bbox_min=tool_bbox_min,
                tool_bbox_max=tool_bbox_max,
                P_anchor=contact_anchors,
                object_center_E=object_bbox_center_E,
                finger_cavity_halfspaces_T=finger_cavity_halfspaces_T,
            )
    elif cfg.contact_geometry_mode == CONTACT_GEOMETRY_BBOX_TRANSLATION_NEAREST:
        _log(
            "[GEOMETRY] bbox-translation nearest-surface candidates "
            f"translations={cfg.B} rotations_per_translation={cfg.M}"
        )
        geometry = bbox_translation_nearest_sample_candidates(
            tool_points_T,
            object_points_E,
            object_verts_E,
            cfg,
            sdf_grid=sdf_grid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )
    else:
        raise ValueError(
            "Unsupported contact_geometry_mode "
            f"'{cfg.contact_geometry_mode}'. Expected "
            f"'{CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION}' or "
            f"'{CONTACT_GEOMETRY_BBOX_TRANSLATION_NEAREST}' or "
            f"'{CONTACT_GEOMETRY_INTERSECTING_ANCHORS}' or "
            f"'{CONTACT_GEOMETRY_TANGENT_GAUSSIAN}'."
        )
    candidates = _candidate_fields_from_geometry(geometry, R_obj, object_bbox_center_E)
    n = int(candidates["tool_translation_E"].shape[0])
    _log(f"[GEOMETRY] generated candidates={n}")
    object_min_z_E = float(object_verts_E[:, 2].min().detach().cpu().item())
    tool_min_z_if_accepted = None
    if n > 0:
        R_tool_E = geometry["tool_rotation_E"].to(tool_verts_T.device)
        t_tool_E = geometry["tool_translation_E"].to(tool_verts_T.device)
        tool_points_E = torch.einsum("nij,kj->nki", R_tool_E, tool_verts_T) + t_tool_E[:, None, :]
        tool_min_z_if_accepted = float(tool_points_E[..., 2].min().detach().cpu().item())
    initial_min_sdf = geometry["initial_min_sdf"].detach().cpu()
    initial_penetration_depth = geometry["initial_penetration_depth"].detach().cpu()
    initial_min_sdf_value = float(initial_min_sdf.min().item()) if int(initial_min_sdf.numel()) > 0 else None
    initial_penetration_depth_max = (
        float(initial_penetration_depth.max().item())
        if int(initial_penetration_depth.numel()) > 0
        else None
    )
    tool_point_inside_object = None
    object_point_inside_tool = None
    tool_point_object_signed_sdf = None
    object_point_tool_signed_sdf = None
    if cfg.precompute_convex_union_labels or cfg.precompute_mesh_sdf:
        if n <= 0:
            raise RuntimeError(
                "Cannot precompute contact supervision for an empty candidate batch"
            )
        R_tool_E = geometry["tool_rotation_E"].to(tool_points_T.device)
        t_tool_E = geometry["tool_translation_E"].to(tool_points_T.device)
        sampled_tool_points_E = (
            torch.einsum("nij,kj->nki", R_tool_E, tool_points_T)
            + t_tool_E[:, None, :]
        )
        tool_queries_O = torch.matmul(
            sampled_tool_points_E - object_bbox_center_E.reshape(1, 1, 3),
            R_obj,
        )
        object_queries_T = torch.matmul(
            object_points_E.unsqueeze(0) - t_tool_E[:, None, :],
            R_tool_E,
        )
    if cfg.precompute_convex_union_labels:
        _log(
            "[GEOMETRY] precomputing paper point-in-convex-union labels "
            f"candidates={n} points_per_cloud={cfg.K}"
        )
        tool_point_inside_object = points_inside_convex_component_union(
            tool_queries_O,
            obj_verts_O,
            obj_faces,
        ).detach().cpu()
        object_point_inside_tool = points_inside_convex_component_union(
            object_queries_T,
            tool_verts_T,
            tool_faces,
        ).detach().cpu()
        _log(
            "[GEOMETRY] convex-union labels complete "
            f"tool_positive_fraction="
            f"{tool_point_inside_object.float().mean().item():.6f} "
            f"object_positive_fraction="
            f"{object_point_inside_tool.float().mean().item():.6f}"
        )
    if cfg.precompute_mesh_sdf:
        _log(
            "[GEOMETRY] precomputing mutual signed mesh SDF "
            f"candidates={n} points_per_cloud={cfg.K}"
        )
        tool_point_object_signed_sdf = signed_distance_points_to_mesh(
            tool_queries_O.reshape(-1, 3),
            obj_verts_O,
            obj_faces,
            chunk_size=8192,
        ).reshape(n, cfg.K).detach().cpu()
        object_point_tool_signed_sdf = signed_distance_points_to_mesh(
            object_queries_T.reshape(-1, 3),
            tool_verts_T,
            tool_faces,
            chunk_size=8192,
        ).reshape(n, cfg.K).detach().cpu()
        if (
            not bool(tool_point_object_signed_sdf.isfinite().all())
            or not bool(object_point_tool_signed_sdf.isfinite().all())
        ):
            raise RuntimeError("Precomputed mutual signed mesh SDF is non-finite")
        _log("[GEOMETRY] mutual signed mesh SDF complete")

    geometry_diagnostics: dict[str, Any] = {
        "enabled": bool(cfg.visualization_enabled),
        "object_id": cfg.object_id,
        "tool_id": cfg.tool_id,
        "object_mesh_path": str(cfg.object_mesh_path),
        "tool_mesh_path": str(cfg.tool_mesh_path),
        "seed": int(cfg.seed),
        "B": int(cfg.B),
        "M": int(cfg.M),
        "chunk_B": int(cfg.chunk_B),
        "K": int(cfg.K),
        "object_scale": float(object_scale),
        "tool_scale_xyz": tuple(float(x) for x in cfg.tool_scale_xyz),
        "object_bbox_center_M": object_bbox_center.detach().cpu().tolist(),
        "object_bbox_extent_M": object_bbox_extent.detach().cpu().tolist(),
        "tool_bbox_center_M": tool_bbox_center.detach().cpu().tolist(),
        "tool_bbox_extent_M": tool_bbox_extent.detach().cpu().tolist(),
        "object_bbox_center_E": object_bbox_center_E.detach().cpu().tolist(),
        "object_min_z_E": object_min_z_E,
        "accepted_tool_min_z_E": tool_min_z_if_accepted,
        "initial_min_sdf": initial_min_sdf.tolist(),
        "initial_min_sdf_min": initial_min_sdf_value,
        "initial_penetration_depth": initial_penetration_depth.tolist(),
        "initial_penetration_depth_max": initial_penetration_depth_max,
        "candidate_count_after_geometry_filter": n,
        "precomputed_convex_union_labels": bool(
            cfg.precompute_convex_union_labels
        ),
        "precomputed_mesh_sdf": bool(cfg.precompute_mesh_sdf),
        "tool_point_positive_count": (
            int(tool_point_inside_object.sum().item())
            if tool_point_inside_object is not None
            else None
        ),
        "object_point_positive_count": (
            int(object_point_inside_tool.sum().item())
            if object_point_inside_tool is not None
            else None
        ),
        "source_candidate_index": geometry["source_candidate_index"].detach().cpu().tolist(),
        "cavity_capture_fraction": (
            geometry["cavity_capture_fraction"].detach().cpu().tolist()
            if geometry.get("cavity_capture_fraction") is not None
            else None
        ),
        "geometry_filter": (
            "none_intersection_allowed"
            if cfg.contact_geometry_mode == CONTACT_GEOMETRY_INTERSECTING_ANCHORS
            else (
                "paper_tangent_gaussian_no_rejection"
                if cfg.contact_geometry_mode == CONTACT_GEOMETRY_TANGENT_GAUSSIAN
                else "penetration_floor_upright"
            )
        ),
        "contact_geometry_mode": cfg.contact_geometry_mode,
        "rotation_selection": cfg.rotation_selection,
        "tool_mesh_contract": cfg.tool_mesh_contract,
        "require_tool_tip_anchor": bool(cfg.require_tool_tip_anchor),
        "contact_anchor_source": (
            "contact_tip_mesh"
            if cfg.require_tool_tip_anchor
            else "full_tool_surface"
        ),
    }
    if cfg.visualization_enabled:
        geometry_diagnostics["chunks"] = geometry.get("geometry_diagnostics", {}).get("chunks", [])
        geometry_diagnostics["chosen_tool_anchors_T"] = geometry["contact_pt_tool_T"].detach().cpu().tolist()
        geometry_diagnostics["chosen_object_anchors_E"] = geometry["contact_pt_obj_E"].detach().cpu().tolist()
        _log(
            "[GEOMETRY-DIAG] "
            f"tool={cfg.tool_id} object={cfg.object_id} seed={cfg.seed} "
            f"B={cfg.B} M={cfg.M} chunk_B={cfg.chunk_B} K={cfg.K} "
            f"object_scale={object_scale:.6f} object_min_z_E={object_min_z_E:.6f} "
            f"accepted_tool_min_z_E={tool_min_z_if_accepted} "
            f"initial_min_sdf_min={initial_min_sdf_value} "
            f"initial_penetration_depth_max={initial_penetration_depth_max} candidates={n} "
            f"candidate_artifact={candidate_debug_path_for(cfg.output_path)}"
        )

    return GeometryCandidateBatch(
        candidates=candidates,
        object_scale=float(object_scale),
        tool_scale_xyz=tool_scale_xyz.detach().cpu(),
        tool_head_area_aabb_norm=head_area_tensor.detach().cpu(),
        object_bbox_center_M=object_bbox_center.detach().cpu(),
        object_bbox_extent_M=object_bbox_extent.detach().cpu(),
        tool_bbox_center_M=tool_bbox_center.detach().cpu(),
        tool_bbox_extent_M=tool_bbox_extent.detach().cpu(),
        object_points_O=object_points_O.detach().cpu(),
        tool_points_T=tool_points_T.detach().cpu(),
        contact_normal_E=torch.zeros(n, 3, dtype=torch.float32),
        tool_point_inside_object=tool_point_inside_object,
        object_point_inside_tool=object_point_inside_tool,
        tool_point_object_signed_sdf=tool_point_object_signed_sdf,
        object_point_tool_signed_sdf=object_point_tool_signed_sdf,
        source_candidate_index=geometry["source_candidate_index"].detach().cpu(),
        debug_metrics={
            "algorithm": cfg.contact_geometry_mode,
            "contact_geometry_mode": cfg.contact_geometry_mode,
            "rotation_selection": cfg.rotation_selection,
            "num_anchor_pairs": int(cfg.B),
            "rotations_per_pair": int(cfg.M),
            "penetration_eps": float(cfg.penetration_eps),
            "tangent_translation_noise_std": float(
                cfg.tangent_translation_noise_std
            ),
            "tangent_rotation_noise_std_rad": float(
                cfg.tangent_rotation_noise_std_rad
            ),
            "tool_mesh_contract": cfg.tool_mesh_contract,
            "require_tool_tip_anchor": bool(cfg.require_tool_tip_anchor),
            "geometry_diagnostics": geometry_diagnostics,
        },
    )


def candidate_debug_path_for(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_suffix(path.suffix + ".candidate.pt")


def candidate_manifest_path_for(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_suffix(path.suffix + ".candidate.manifest.json")


def save_candidate_artifact(
    output_path: str | Path,
    cfg: GeometryContactConfig,
    batch: GeometryCandidateBatch,
) -> Path:
    torch = _torch()
    path = candidate_debug_path_for(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "contact_candidate_v1",
        "generator": "contact_generation.gen_contact",
        "config_name": cfg.config_name,
        "config_hash": cfg.config_hash,
        "candidate_artifact_path": str(path),
        "object_id": cfg.object_id,
        "tool_id": cfg.tool_id,
        "object_mesh_path": str(Path(cfg.object_mesh_path).resolve()),
        "tool_mesh_path": str(Path(cfg.tool_mesh_path).resolve()),
        "object_scale": float(batch.object_scale),
        "tool_scale_xyz": batch.tool_scale_xyz.detach().cpu(),
        "tool_head_area_aabb_norm": batch.tool_head_area_aabb_norm.detach().cpu(),
        "object_bbox_center_M": batch.object_bbox_center_M.detach().cpu(),
        "object_bbox_extent_M": batch.object_bbox_extent_M.detach().cpu(),
        "tool_bbox_center_M": batch.tool_bbox_center_M.detach().cpu(),
        "tool_bbox_extent_M": batch.tool_bbox_extent_M.detach().cpu(),
        "object_points_O": batch.object_points_O.detach().cpu(),
        "tool_points_T": batch.tool_points_T.detach().cpu(),
        "contact_normal_E": batch.contact_normal_E.detach().cpu(),
        "tool_point_inside_object": (
            batch.tool_point_inside_object.detach().cpu()
            if batch.tool_point_inside_object is not None
            else None
        ),
        "object_point_inside_tool": (
            batch.object_point_inside_tool.detach().cpu()
            if batch.object_point_inside_tool is not None
            else None
        ),
        "tool_point_object_signed_sdf": (
            batch.tool_point_object_signed_sdf.detach().cpu()
            if batch.tool_point_object_signed_sdf is not None
            else None
        ),
        "object_point_tool_signed_sdf": (
            batch.object_point_tool_signed_sdf.detach().cpu()
            if batch.object_point_tool_signed_sdf is not None
            else None
        ),
        "object_point_sample_seed": int(cfg.seed),
        "tool_point_sample_seed": int(cfg.seed),
        "num_candidates": batch.num_candidates,
        "candidates": {key: value.detach().cpu() for key, value in batch.candidates.items()},
        "source_candidate_index": batch.source_candidate_index,
        "debug_metrics": dict(batch.debug_metrics),
    }
    torch.save(payload, path)
    write_json(
        candidate_manifest_path_for(output_path),
        {
            "schema_version": "contact_candidate_manifest_v1",
            "status": "candidate_generated",
            "candidate_artifact_path": str(path),
            "num_candidates": batch.num_candidates,
            "precomputed_convex_union_labels": (
                batch.tool_point_inside_object is not None
                and batch.object_point_inside_tool is not None
            ),
            "precomputed_mesh_sdf": (
                batch.tool_point_object_signed_sdf is not None
                and batch.object_point_tool_signed_sdf is not None
            ),
            "tool_point_label_shape": (
                list(batch.tool_point_inside_object.shape)
                if batch.tool_point_inside_object is not None
                else None
            ),
            "object_point_label_shape": (
                list(batch.object_point_inside_tool.shape)
                if batch.object_point_inside_tool is not None
                else None
            ),
            "tool_point_sdf_shape": (
                list(batch.tool_point_object_signed_sdf.shape)
                if batch.tool_point_object_signed_sdf is not None
                else None
            ),
            "object_point_sdf_shape": (
                list(batch.object_point_tool_signed_sdf.shape)
                if batch.object_point_tool_signed_sdf is not None
                else None
            ),
            "config_name": cfg.config_name,
            "config_hash": cfg.config_hash,
            "candidate_hash": hash_json(
                {
                    "object_id": cfg.object_id,
                    "tool_id": cfg.tool_id,
                    "B": cfg.B,
                    "M": cfg.M,
                }
            ),
        },
    )
    return path


def load_candidate_artifact(output_path: str | Path) -> dict[str, Any]:
    torch = _torch()
    path = candidate_debug_path_for(output_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Geometry candidate artifact does not exist: {path}. "
            "Run contact_generation.gen_contact first."
        )
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema_version") != "contact_candidate_v1":
        raise ValueError(f"Invalid contact candidate artifact: {path}")
    return payload


def run_geometry_contact_pair(cfg: GeometryContactConfig) -> int:
    _log(
        "[GEOMETRY-PAIR-START] "
        f"tool={cfg.tool_id} object={cfg.object_id} output={cfg.output_path}"
    )
    candidates = generate_contact_candidates(cfg)
    _log(
        "[GEOMETRY-PAIR-DONE] "
        f"tool={cfg.tool_id} object={cfg.object_id} candidates={candidates.num_candidates}"
    )
    _log(f"[GEOMETRY-SAVE] output={candidate_debug_path_for(cfg.output_path)}")
    save_candidate_artifact(cfg.output_path, cfg, candidates)
    return candidates.num_candidates


def _log(message: str) -> None:
    print(f"[contact_generation.geometry] {message}", flush=True)
