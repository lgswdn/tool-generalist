"""Geometry candidate generation for contact_pt_env_v1 datasets.

This module owns only the approximate geometric contact step:
object pose sampling/grounding, bbox-center canonicalization, tool/object anchor
sampling, B anchor pairs, shared M rotations per chunk, floor checks, and
penetration checks.  Torch/Kaolin/trimesh are imported lazily inside execution
paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from configs.config_contact_gen import (
    ROTATION_SELECTION_MOST_DOWNWARD,
    ROTATION_SELECTION_RANDOM_LEGAL,
    TOOL_SOURCE_OBJECTS,
    ContactGenCfg,
)
from utils.assets import (
    ToolAssetContractError,
    assert_adjusted_decomposed_mesh_path,
    compute_head_bounds,
    load_tool_head_area,
    split_head_body,
)
from utils.io import hash_json, write_json


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
    contact_mode_prob: Mapping[str, float] = field(default_factory=lambda: {"head": 0.7, "body": 0.3})
    upright_threshold: float = 0.0
    rotation_selection: str = ROTATION_SELECTION_MOST_DOWNWARD
    tool_mesh_contract: str = "adjusted_decomposed_mesh"
    use_tool_head_area: bool = True
    epsilon: float = 2e-3
    floor_eps: float = 0.0
    penetration_eps: float = 5e-4
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
            contact_mode_prob=dict(contact_cfg.contact_mode_prob),
            upright_threshold=contact_cfg.upright_threshold,
            rotation_selection=contact_cfg.rotation_selection,
            tool_mesh_contract=(
                "object_mesh"
                if contact_cfg.tool_source == TOOL_SOURCE_OBJECTS
                else "adjusted_decomposed_mesh"
            ),
            use_tool_head_area=contact_cfg.tool_source != TOOL_SOURCE_OBJECTS,
            epsilon=contact_cfg.epsilon,
            floor_eps=contact_cfg.floor_eps,
            penetration_eps=contact_cfg.penetration_eps,
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
    source_candidate_index: Any
    debug_metrics: dict[str, Any]

    @property
    def num_candidates(self) -> int:
        return int(self.candidates["tool_translation_E"].shape[0])


def head_contact_probability(contact_mode_prob: Mapping[str, float]) -> float:
    total = sum(float(value) for value in contact_mode_prob.values())
    if total <= 0:
        raise ValueError("ContactGenCfg.contact_mode_prob must have positive total mass")
    return float(contact_mode_prob.get("head", 0.0)) / total


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


def _sample_tool_anchors_per_pair(P_tool, P_head, P_body, probability: float, count: int):
    """Sample one tool anchor per pair with independent head/body Bernoulli draws."""

    torch = _torch()
    device = P_tool.device
    if P_head is None or P_body is None:
        return P_tool[torch.randint(P_tool.shape[0], (count,), device=device)]

    choose_head = torch.rand(count, device=device) < float(probability)
    anchors = torch.empty(count, 3, dtype=P_tool.dtype, device=device)
    if bool(choose_head.any()):
        n_head = int(choose_head.sum().item())
        anchors[choose_head] = P_head[torch.randint(P_head.shape[0], (n_head,), device=device)]
    if bool((~choose_head).any()):
        n_body = int((~choose_head).sum().item())
        anchors[~choose_head] = P_body[torch.randint(P_body.shape[0], (n_body,), device=device)]
    return anchors


def rejection_sample_candidates(
    P_tool_T,
    object_surface_E,
    cfg: GeometryContactConfig,
    *,
    sdf_grid,
    bbox_min,
    bbox_max,
    P_head=None,
    P_body=None,
):
    """Return one legal candidate per accepted anchor pair.

    Each chunk samples ``M`` random rotations once, then broadcasts those
    rotations across all anchor pairs in that chunk.  A candidate is accepted
    only when it stays above the floor and does not penetrate the object beyond
    ``cfg.penetration_eps``.  Rotation selection is controlled by
    ``cfg.rotation_selection``.
    """

    torch = _torch()
    device = cfg.device
    B = int(cfg.B)
    M = int(cfg.M)
    K = int(P_tool_T.shape[0])
    chunk_B = int(cfg.chunk_B)
    head_prob = head_contact_probability(cfg.contact_mode_prob)

    R_list = []
    t_list = []
    tool_anchor_list = []
    obj_anchor_list = []
    source_index_list = []
    min_sdf_list = []
    penetration_depth_list = []
    debug_chunks: list[dict[str, Any]] = []

    for b_start in range(0, B, chunk_B):
        b_end = min(b_start + chunk_B, B)
        cb = b_end - b_start
        p_B = _sample_tool_anchors_per_pair(P_tool_T, P_head, P_body, head_prob, cb)
        obj_idx = torch.randint(object_surface_E.shape[0], (cb,), device=device)
        p_A = object_surface_E[obj_idx]

        shifted = P_tool_T.unsqueeze(0) - p_B.unsqueeze(1)
        rotations = sample_upright_rotations(M, device, cfg.upright_threshold)
        points_E = torch.einsum("mij,bkj->bmki", rotations, shifted) + p_A[:, None, None, :]

        floor_ok = points_E[..., 2].min(dim=-1).values >= -float(cfg.floor_eps)
        sdf = query_sdf_grid(points_E.reshape(-1, 3), sdf_grid, bbox_min, bbox_max).reshape(cb, M, K)
        min_sdf = sdf.amin(dim=-1)
        penetration_depth = torch.clamp(-min_sdf, min=0.0)
        penetration_ok = penetration_depth <= float(cfg.penetration_eps)
        valid = floor_ok & penetration_ok
        chunk_min_z = points_E[..., 2].amin(dim=(-1, -2))
        downward_score = rotations[:, 2, 2].unsqueeze(0).expand(cb, -1).clone()
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
        pair_valid = valid[torch.arange(cb, device=device), best_m]
        if not bool(pair_valid.any()):
            if cfg.visualization_enabled:
                debug_chunks.append(
                    {
                        "b_start": int(b_start),
                        "b_end": int(b_end),
                        "valid_pairs": 0,
                        "min_tool_z": float(chunk_min_z.min().detach().cpu().item()),
                        "min_sdf": float(min_sdf.min().detach().cpu().item()),
                        "penetration_depth_max": float(penetration_depth.max().detach().cpu().item()),
                    }
                )
            continue

        vi = pair_valid.nonzero(as_tuple=False).squeeze(1)
        mi = best_m[vi]
        R_sel = rotations[mi]
        p_B_sel = p_B[vi]
        p_A_sel = p_A[vi]
        t_sel = p_A_sel - torch.einsum("nij,nj->ni", R_sel, p_B_sel)
        min_sdf_sel = min_sdf[vi, mi]
        penetration_sel = penetration_depth[vi, mi]
        downward_score_sel = downward_score[vi, mi]

        R_list.append(R_sel.detach().cpu())
        t_list.append(t_sel.detach().cpu())
        tool_anchor_list.append(p_B_sel.detach().cpu())
        obj_anchor_list.append(p_A_sel.detach().cpu())
        source_index_list.append((torch.arange(cb, device=device)[vi] + b_start).detach().cpu())
        min_sdf_list.append(min_sdf_sel.detach().cpu())
        penetration_depth_list.append(penetration_sel.detach().cpu())
        if cfg.visualization_enabled:
            debug_chunks.append(
                {
                    "b_start": int(b_start),
                    "b_end": int(b_end),
                    "valid_pairs": int(pair_valid.sum().detach().cpu().item()),
                    "selected_rotation_indices": mi.detach().cpu().tolist(),
                    "selected_source_indices": (torch.arange(cb, device=device)[vi] + b_start).detach().cpu().tolist(),
                    "selected_tool_anchors_T": p_B_sel.detach().cpu().tolist(),
                    "selected_object_anchors_E": p_A_sel.detach().cpu().tolist(),
                    "selected_downward_scores": downward_score_sel.detach().cpu().tolist(),
                    "min_tool_z": float(chunk_min_z.min().detach().cpu().item()),
                    "selected_min_sdf": min_sdf_sel.detach().cpu().tolist(),
                    "selected_penetration_depth": penetration_sel.detach().cpu().tolist(),
                    "chunk_min_sdf": float(min_sdf.min().detach().cpu().item()),
                    "chunk_penetration_depth_max": float(penetration_depth.max().detach().cpu().item()),
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
            "geometry_diagnostics": {"chunks": debug_chunks},
        }

    return {
        "tool_rotation_E": torch.cat(R_list, dim=0),
        "tool_translation_E": torch.cat(t_list, dim=0),
        "contact_pt_tool_T": torch.cat(tool_anchor_list, dim=0),
        "contact_pt_obj_E": torch.cat(obj_anchor_list, dim=0),
        "source_candidate_index": torch.cat(source_index_list, dim=0).to(dtype=torch.int64),
        "initial_min_sdf": torch.cat(min_sdf_list, dim=0),
        "initial_penetration_depth": torch.cat(penetration_depth_list, dim=0),
        "geometry_diagnostics": {"chunks": debug_chunks},
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
    if cfg.tool_mesh_contract == "adjusted_decomposed_mesh":
        tool_id = cfg.tool_id or assert_adjusted_decomposed_mesh_path(cfg.tool_mesh_path)
        assert_adjusted_decomposed_mesh_path(cfg.tool_mesh_path, tool_id)
        if cfg.use_tool_head_area and (
            not cfg.tools_json_path or not Path(cfg.tools_json_path).exists()
        ):
            raise ToolAssetContractError(f"tools_adjusted.json is required: {cfg.tools_json_path}")
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
    tool_points_M = sample_surface_points_torch(tool_verts_M, tool_faces, cfg.K)
    tool_points_T, _, _ = centralize_points_by_bbox(
        tool_points_M,
        bbox_center=tool_bbox_center,
        bbox_extent=tool_bbox_extent,
    )
    _log(f"[GEOMETRY] sampled surface points object={cfg.K} tool={cfg.K}")

    _log("[GEOMETRY] sampling object pose and grounding")
    object_verts_E, R_obj, object_bbox_center_E = sample_object_pose_and_ground(obj_verts_O, obj_faces)
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
    _log(f"[GEOMETRY] sampling object surface anchors count={max(cfg.B * 4, 16384)}")
    object_surface_E = sample_surface_points_torch(object_verts_E, obj_faces, max(cfg.B * 4, 16384))

    if cfg.use_tool_head_area:
        _log("[GEOMETRY] loading tool head area and splitting head/body points")
        head_area = load_tool_head_area(cfg.tools_json_path, cfg.tool_mesh_path, tool_id=tool_id)
        head_area_tensor = torch.tensor(
            head_area if head_area is not None else ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            dtype=torch.float32,
        )
        head_bounds = compute_head_bounds(tool_verts_T, head_area)
        P_head, P_body = split_head_body(tool_points_T, head_bounds)
    else:
        _log("[GEOMETRY] tool head area disabled; using full tool surface for anchors")
        head_area_tensor = torch.tensor(
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            dtype=torch.float32,
        )
        P_head = P_body = None

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
        P_head=P_head,
        P_body=P_body,
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
        "source_candidate_index": geometry["source_candidate_index"].detach().cpu().tolist(),
        "geometry_filter": "penetration_floor_upright",
        "rotation_selection": cfg.rotation_selection,
        "tool_mesh_contract": cfg.tool_mesh_contract,
        "use_tool_head_area": bool(cfg.use_tool_head_area),
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
        source_candidate_index=geometry["source_candidate_index"].detach().cpu(),
        debug_metrics={
            "algorithm": "bbox_centered_anchor_pair_rejection",
            "rotation_selection": cfg.rotation_selection,
            "num_anchor_pairs": int(cfg.B),
            "rotations_per_pair": int(cfg.M),
            "penetration_eps": float(cfg.penetration_eps),
            "head_contact_probability": head_contact_probability(cfg.contact_mode_prob),
            "tool_mesh_contract": cfg.tool_mesh_contract,
            "use_tool_head_area": bool(cfg.use_tool_head_area),
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
