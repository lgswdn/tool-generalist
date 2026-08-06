#!/usr/bin/env python3
"""Analyze contact dataset quality with tool-point signed SDF to object mesh.

For each contact pose pair in contact_pt_env_v1 files, this script computes the
fraction of sampled tool points whose signed distance to the posed object mesh is
below a threshold.  SDF labels are computed the same way as pretraining: query
points are transformed into the centered object mesh frame, then evaluated with
the signed kaolin backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from utils.contact.schema import ContactSchemaError, load_and_validate_contact_pt
from utils.geometry.mesh_io import load_mesh_vertices_faces, scale_vertices
from utils.geometry.sdf import _signed_distance_points_to_prepared_mesh


BLOCKED_PT_SUFFIXES = (
    ".candidate.pt",
    ".physics_debug.pt",
    ".stabilized_success.pt",
    ".stabilized.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Root contact dataset/artifact directory.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help="SDF threshold in dataset units, usually meters. Defaults to 0.005.",
    )
    parser.add_argument(
        "--metric",
        choices=("signed_le", "abs_le"),
        default="signed_le",
        help=(
            "signed_le counts sdf <= threshold, including penetrating points. "
            "abs_le counts abs(sdf) <= threshold."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for SDF queries. Defaults to cuda.",
    )
    parser.add_argument(
        "--sdf-chunk-size",
        type=int,
        default=65536,
        help="Number of query points per kaolin SDF call chunk.",
    )
    parser.add_argument(
        "--contact-chunk-size",
        type=int,
        default=256,
        help="Number of contact pose pairs processed together per file.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of .pt files, useful for smoke tests. 0 means all.",
    )
    parser.add_argument(
        "--max-contacts-per-file",
        type=int,
        default=0,
        help="Optional cap on contact pose pairs analyzed per .pt file. 0 means all.",
    )
    parser.add_argument(
        "--tool-mesh-contract",
        choices=("auto", "adjusted_decomposed_mesh", "object_mesh"),
        default="auto",
        help=(
            "Tool mesh contract used by schema validation. Defaults to auto, "
            "which honors a payload tool_mesh_contract field when present and "
            "falls back to object_mesh if only the adjusted tool asset path "
            "contract fails; use adjusted_decomposed_mesh for strict tool-asset validation."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSONL output path for per-contact metrics. Overridden by --output-dir.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional JSON output path for dataset/file summary metrics. Overridden by --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional report directory. When set, writes all reports there with "
            "filenames based on the contact dataset hash, i.e. the data_dir basename."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N files. Use 0 to disable.",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Skip invalid/incomplete contact files instead of failing.",
    )
    return parser.parse_args()


def iter_contact_pt_files(data_dir: str | Path, *, max_files: int = 0) -> Iterable[Path]:
    root = Path(data_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"data dir does not exist: {root}")
    emitted = 0
    for path in root.rglob("*.pt"):
        if any(str(path).endswith(suffix) for suffix in BLOCKED_PT_SUFFIXES):
            continue
        yield path
        emitted += 1
        if int(max_files) > 0 and emitted >= int(max_files):
            return


def load_centered_object_mesh(data: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, faces = load_mesh_vertices_faces(data["object_mesh_path"], process=False)
    scaled = scale_vertices(vertices, float(data["object_scale"]))
    centered = scaled - torch.as_tensor(data["object_bbox_center_M"], dtype=torch.float64).cpu().numpy()
    return (
        torch.as_tensor(centered, dtype=torch.float32, device=device).contiguous(),
        torch.as_tensor(faces, dtype=torch.long, device=device).contiguous(),
    )


def tensor_on_device(data: dict[str, Any], key: str, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(data[key], dtype=torch.float32, device=device).contiguous()


def load_metric_contact_pt(pt_path: Path, *, tool_mesh_contract: str) -> dict[str, Any]:
    contract = str(tool_mesh_contract)
    try:
        return dict(
            load_and_validate_contact_pt(
                pt_path,
                allow_mock=False,
                require_real_physics=False,
                require_complete=True,
                tool_mesh_contract=contract,
            )
        )
    except ContactSchemaError as exc:
        if contract != "auto" or "tool mesh must be" not in str(exc):
            raise
        return dict(
            load_and_validate_contact_pt(
                pt_path,
                allow_mock=False,
                require_real_physics=False,
                require_complete=True,
                tool_mesh_contract="object_mesh",
            )
        )


def analyze_file(
    pt_path: Path,
    *,
    threshold: float,
    metric: str,
    device: torch.device,
    sdf_chunk_size: int,
    contact_chunk_size: int,
    max_contacts_per_file: int = 0,
    tool_mesh_contract: str = "auto",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = load_metric_contact_pt(pt_path, tool_mesh_contract=str(tool_mesh_contract))
    n_full = int(data["num_contacts"])
    n = n_full
    if int(max_contacts_per_file) > 0:
        n = min(n, int(max_contacts_per_file))
    if n <= 0:
        return [], {
            "pt_path": str(pt_path),
            "object_id": str(data.get("object_id", "")),
            "tool_id": str(data.get("tool_id", "")),
            "num_contacts_in_file": n_full,
            "num_contacts_analyzed": 0,
        }

    object_v, object_f = load_centered_object_mesh(data, device)
    object_face_vertices = object_v[object_f].unsqueeze(0).contiguous()
    tool_points_T = tensor_on_device(data, "tool_points_T", device)
    object_R = tensor_on_device(data, "object_rotation_E", device)
    object_t = tensor_on_device(data, "object_bbox_center_E", device)
    tool_R = tensor_on_device(data, "tool_rotation_E", device)
    tool_t = tensor_on_device(data, "tool_translation_E", device)
    threshold_f = float(threshold)

    records: list[dict[str, Any]] = []
    ratio_values: list[float] = []
    inside_values: list[float] = []
    min_values: list[float] = []

    step = max(1, int(contact_chunk_size))
    for start in range(0, n, step):
        end = min(n, start + step)
        c = end - start
        tool_points_E = torch.matmul(tool_points_T.unsqueeze(0), tool_R[start:end].transpose(1, 2))
        tool_points_E = tool_points_E + tool_t[start:end].reshape(c, 1, 3)
        q_tool_obj = torch.matmul(
            tool_points_E - object_t[start:end].reshape(c, 1, 3),
            object_R[start:end],
        )
        sdf = _signed_distance_points_to_prepared_mesh(
            q_tool_obj.reshape(c * tool_points_T.shape[0], 3),
            mesh_v=object_v,
            mesh_f=object_f,
            face_vertices=object_face_vertices,
            chunk_size=int(sdf_chunk_size),
            backend="kaolin",
        ).reshape(c, tool_points_T.shape[0])

        if metric == "abs_le":
            close_mask = sdf.abs() <= threshold_f
        else:
            close_mask = sdf <= threshold_f
        ratios = close_mask.float().mean(dim=1)
        inside_ratios = (sdf < 0.0).float().mean(dim=1)
        mins = sdf.min(dim=1).values
        means = sdf.mean(dim=1)
        quantiles = torch.quantile(sdf, torch.tensor([0.01, 0.05, 0.5, 0.95], device=device), dim=1)

        for local_i in range(c):
            contact_index = start + local_i
            ratio = float(ratios[local_i].detach().cpu())
            inside_ratio = float(inside_ratios[local_i].detach().cpu())
            min_sdf = float(mins[local_i].detach().cpu())
            ratio_values.append(ratio)
            inside_values.append(inside_ratio)
            min_values.append(min_sdf)
            records.append(
                {
                    "pt_path": str(pt_path),
                    "object_id": str(data["object_id"]),
                    "tool_id": str(data["tool_id"]),
                    "contact_index": int(contact_index),
                    "num_tool_points": int(tool_points_T.shape[0]),
                    "threshold": threshold_f,
                    "metric": metric,
                    "ratio": ratio,
                    "ratio_inside": inside_ratio,
                    "num_inside": int((sdf[local_i] < 0.0).sum().detach().cpu()),
                    "min_sdf": min_sdf,
                    "mean_sdf": float(means[local_i].detach().cpu()),
                    "p01_sdf": float(quantiles[0, local_i].detach().cpu()),
                    "p05_sdf": float(quantiles[1, local_i].detach().cpu()),
                    "p50_sdf": float(quantiles[2, local_i].detach().cpu()),
                    "p95_sdf": float(quantiles[3, local_i].detach().cpu()),
                }
            )

    file_summary = summarize_values(
        ratio_values,
        prefix="ratio",
        extra={
            "pt_path": str(pt_path),
            "object_id": str(data["object_id"]),
            "tool_id": str(data["tool_id"]),
            "num_contacts_in_file": n_full,
            "num_contacts_analyzed": n,
            "threshold": threshold_f,
            "metric": metric,
            "ratio_inside_mean": mean_or_none(inside_values),
            "min_sdf_min": min(min_values) if min_values else None,
        },
    )
    return records, file_summary


def mean_or_none(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def summarize_values(values: list[float], *, prefix: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = dict(extra or {})
    if not values:
        summary.update(
            {
                f"{prefix}_count": 0,
                f"{prefix}_mean": None,
                f"{prefix}_min": None,
                f"{prefix}_p05": None,
                f"{prefix}_p50": None,
                f"{prefix}_p95": None,
                f"{prefix}_max": None,
            }
        )
        return summary
    tensor = torch.tensor(values, dtype=torch.float64)
    qs = torch.quantile(tensor, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    summary.update(
        {
            f"{prefix}_count": int(tensor.numel()),
            f"{prefix}_mean": float(tensor.mean()),
            f"{prefix}_min": float(tensor.min()),
            f"{prefix}_p05": float(qs[0]),
            f"{prefix}_p50": float(qs[1]),
            f"{prefix}_p95": float(qs[2]),
            f"{prefix}_max": float(tensor.max()),
        }
    )
    return summary


def dataset_hash_from_path(data_dir: str | Path) -> str:
    name = Path(data_dir).expanduser().resolve().name
    if not name:
        raise ValueError(f"Could not infer dataset hash from data_dir={data_dir}")
    return name


def resolve_report_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        dataset_hash = dataset_hash_from_path(args.data_dir)
        return (
            output_dir / f"{dataset_hash}.contact_sdf_quality.jsonl",
            output_dir / f"{dataset_hash}.contact_sdf_quality.summary.json",
        )
    output_path = Path(args.output).expanduser() if args.output else None
    summary_path = Path(args.summary_output).expanduser() if args.summary_output else None
    return output_path, summary_path


def main() -> int:
    args = parse_args()
    if float(args.threshold) < 0.0:
        raise ValueError("--threshold must be non-negative")
    if int(args.sdf_chunk_size) <= 0:
        raise ValueError("--sdf-chunk-size must be positive")
    if int(args.contact_chunk_size) <= 0:
        raise ValueError("--contact-chunk-size must be positive")
    if int(args.max_files) < 0:
        raise ValueError("--max-files must be non-negative")
    if int(args.max_contacts_per_file) < 0:
        raise ValueError("--max-contacts-per-file must be non-negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested --device cuda but torch.cuda.is_available() is false")

    print(
        "[analyze_contact_sdf_quality] scanning "
        f"data_dir={Path(args.data_dir).expanduser()} "
        f"max_files={int(args.max_files)} threshold={float(args.threshold)} "
        f"metric={args.metric} device={device}",
        flush=True,
    )

    num_contacts = 0
    ratio_values: list[float] = []
    inside_values: list[float] = []
    min_sdf_values: list[float] = []
    file_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    progress_every = int(args.progress_every)
    file_i = 0
    output_path, summary_path = resolve_report_paths(args)

    output_f = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_f = output_path.open("w", encoding="utf-8")
    try:
        for file_i, pt_path in enumerate(iter_contact_pt_files(args.data_dir, max_files=int(args.max_files)), start=1):
            try:
                records, file_summary = analyze_file(
                    pt_path,
                    threshold=float(args.threshold),
                    metric=str(args.metric),
                    device=device,
                    sdf_chunk_size=int(args.sdf_chunk_size),
                contact_chunk_size=int(args.contact_chunk_size),
                max_contacts_per_file=int(args.max_contacts_per_file),
                tool_mesh_contract=str(args.tool_mesh_contract),
            )
            except (ContactSchemaError, RuntimeError, ValueError, FileNotFoundError) as exc:
                if not bool(args.allow_invalid):
                    raise
                skipped.append({"pt_path": str(pt_path), "error": str(exc)})
                continue
            for record in records:
                num_contacts += 1
                ratio_values.append(float(record["ratio"]))
                inside_values.append(float(record["ratio_inside"]))
                min_sdf_values.append(float(record["min_sdf"]))
                if output_f is not None:
                    output_f.write(json.dumps(record, sort_keys=True) + "\n")
            file_summaries.append(file_summary)
            if progress_every > 0 and file_i % progress_every == 0:
                print(
                    "[analyze_contact_sdf_quality] progress "
                    f"files_seen={file_i} contacts={num_contacts} skipped={len(skipped)}",
                    flush=True,
                )
    finally:
        if output_f is not None:
            output_f.close()
    if file_i == 0:
        raise RuntimeError(f"No training .pt files found under {args.data_dir}")

    summary = summarize_values(
        ratio_values,
        prefix="ratio",
        extra={
            "data_dir": str(Path(args.data_dir).expanduser()),
            "num_files_seen": file_i,
            "num_files_analyzed": len(file_summaries),
            "num_contacts": num_contacts,
            "num_skipped": len(skipped),
            "threshold": float(args.threshold),
            "metric": str(args.metric),
            "device": str(device),
            "ratio_inside_mean": mean_or_none(inside_values),
            "min_sdf_min": min(min_sdf_values) if min_sdf_values else None,
            "file_summaries": file_summaries,
            "skipped": skipped,
        },
    )

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k not in {"file_summaries", "skipped"}}, indent=2, sort_keys=True))
    if output_path is not None:
        print(f"[analyze_contact_sdf_quality] wrote per-contact JSONL: {output_path}", flush=True)
    if summary_path is not None:
        print(f"[analyze_contact_sdf_quality] wrote summary JSON: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
