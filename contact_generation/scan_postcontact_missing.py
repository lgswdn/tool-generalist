#!/usr/bin/env python3
"""Scan a contact artifact for stabilized contacts missing post-contact output.

This is a lightweight diagnostic: it loads only the experiment config and JSON
manifests.  It does not import Isaac, run generation, or load tensor payloads.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.artifacts.resolver import resolve_artifacts
from utils.config.loader import load_exp_cfg
from utils.io import read_json


@dataclass(frozen=True)
class ScanRow:
    stabilized_artifact: Path
    final_artifact: Path
    status: str
    num_stabilized: int
    num_contacts: int
    reason: str


def _contact_artifact_dir(config: str | Path) -> Path:
    cfg = load_exp_cfg(config)
    for ref in resolve_artifacts(cfg).stages:
        if ref.stage == "contact_gen":
            return ref.directory
    raise RuntimeError(f"Config has no contact_gen artifact stage: {config}")


def _final_artifact_from_stabilized(path: Path) -> Path:
    suffix = ".stabilized_success.pt"
    text = str(path)
    if not text.endswith(suffix):
        raise ValueError(f"Not a stabilized-success artifact: {path}")
    return Path(text[: -len(suffix)])


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return payload


def _scan_one(stabilized_path: Path) -> ScanRow:
    stabilized_manifest = _read_manifest(
        Path(str(stabilized_path).removesuffix(".pt") + ".manifest.json")
    )
    final_path = _final_artifact_from_stabilized(stabilized_path)
    final_manifest = _read_manifest(Path(str(final_path) + ".manifest.json"))

    num_stabilized = int(stabilized_manifest.get("num_stabilized", 0) or 0)
    num_contacts = int(final_manifest.get("num_contacts", 0) or 0)
    final_status = str(final_manifest.get("status", "missing_manifest"))

    if num_stabilized <= 0:
        reason = "no_stabilized_successes"
        status = "skipped"
    elif not final_path.exists():
        reason = "missing_final_pt"
        status = "missing"
    elif not final_manifest:
        reason = "missing_final_manifest"
        status = "missing"
    elif final_status != "complete":
        reason = f"final_manifest_status={final_status}"
        status = "incomplete"
    elif num_contacts <= 0:
        reason = "final_manifest_num_contacts_zero"
        status = "incomplete"
    else:
        reason = "complete"
        status = "complete"

    return ScanRow(
        stabilized_artifact=stabilized_path,
        final_artifact=final_path,
        status=status,
        num_stabilized=num_stabilized,
        num_contacts=num_contacts,
        reason=reason,
    )


def scan_postcontact_missing(config: str | Path) -> list[ScanRow]:
    artifact_dir = _contact_artifact_dir(config)
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Contact artifact directory does not exist: {artifact_dir}")
    stabilized_paths = sorted(artifact_dir.rglob("*.stabilized_success.pt"))
    return [_scan_one(path) for path in stabilized_paths]


def _print_summary(rows: list[ScanRow], *, config: str | Path) -> None:
    artifact_dir = _contact_artifact_dir(config)
    complete = [row for row in rows if row.status == "complete"]
    missing = [row for row in rows if row.status in {"missing", "incomplete"}]
    skipped = [row for row in rows if row.status == "skipped"]
    print(f"[postcontact-scan] config={config}", flush=True)
    print(f"[postcontact-scan] artifact_dir={artifact_dir}", flush=True)
    print(
        "[postcontact-scan] "
        f"stabilized_files={len(rows)} complete_postcontact={len(complete)} "
        f"missing_or_incomplete={len(missing)} skipped_no_stabilized={len(skipped)}",
        flush=True,
    )
    print(
        "[postcontact-scan] "
        f"stabilized_success_total={sum(row.num_stabilized for row in rows)} "
        f"postcontact_contacts_total={sum(row.num_contacts for row in complete)}",
        flush=True,
    )
    if missing:
        print("[postcontact-scan] missing_or_incomplete_examples:", flush=True)
        for row in missing[:20]:
            rel = row.final_artifact
            try:
                rel = row.final_artifact.relative_to(artifact_dir)
            except ValueError:
                pass
            print(
                "  "
                f"status={row.status} reason={row.reason} "
                f"stabilized={row.num_stabilized} contacts={row.num_contacts} output={rel}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment config Python file/module exposing EXP_CFG")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = scan_postcontact_missing(args.config)
    _print_summary(rows, config=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
