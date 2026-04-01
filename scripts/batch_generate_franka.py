#!/usr/bin/env python3
"""Batch-generate Franka USDs with tools mounted from a tools directory.

This script discovers tool USDs under a tools root (default: ./static/objects_usd)
and invokes generate_franka.py once per tool.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


def _safe_name(name: str) -> str:
	return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "tool"


def _discover_tool_usds(tools_root: Path) -> list[tuple[str, Path]]:
	"""Discover one primary tool USD per immediate child folder.

	Rules:
	1) If folder has <folder_name>.usd at its top-level, use it.
	2) Else if exactly one top-level *.usd exists, use it.
	3) Else skip this folder (ambiguous or missing).
	"""
	results: list[tuple[str, Path]] = []

	# Also accept top-level usd files directly under tools_root.
	for usd in sorted(tools_root.glob("*.usd")):
		results.append((usd.stem, usd.resolve()))

	for child in sorted(tools_root.iterdir()):
		if not child.is_dir():
			continue

		same_name = child / f"{child.name}.usd"
		if same_name.is_file():
			results.append((child.name, same_name.resolve()))
			continue

		root_usds = sorted(child.glob("*.usd"))
		if len(root_usds) == 1:
			results.append((child.name, root_usds[0].resolve()))

	return results


def main() -> None:
	parser = argparse.ArgumentParser(description="Batch generate Franka USDs for all tools in a directory.")
	parser.add_argument(
		"--tools-root",
		type=str,
		default="/mnt/afs/wangyuze/ToolGeneralist/static/objects_usd",
		help="Directory containing tool USD folders/files.",
	)
	parser.add_argument(
		"--isaaclab-sh",
		type=str,
		default="/mnt/afs/wangyuze/ToolGeneralist/IsaacLab-2.2.0/isaaclab.sh",
		help="Path to isaaclab.sh launcher.",
	)
	parser.add_argument(
		"--generator-script",
		type=str,
		default="/mnt/afs/wangyuze/ToolGeneralist/tool-generalist/scripts/generate_franka.py",
		help="Path to generate_franka.py.",
	)
	parser.add_argument(
		"--output-base",
		type=str,
		default="/mnt/afs/wangyuze/ToolGeneralist/static/franka/generated/Robots",
		help="Shared output folder. All generated USDs are written here with one copied Franka base.",
	)
	parser.add_argument("--headless", action="store_true", default=True)
	parser.add_argument("--no-headless", action="store_true", default=False)
	parser.add_argument("--mirror-tool-assets", action="store_true", default=False)
	parser.add_argument("--no-mirror-tool-assets", action="store_true", default=False)
	parser.add_argument("--enable-gravity", action="store_true", default=True)
	parser.add_argument("--disable-gravity", action="store_true", default=False)
	parser.add_argument("--overwrite", action="store_true", default=True)
	parser.add_argument("--no-overwrite", action="store_true", default=False)
	parser.add_argument("--cuda-visible-devices", type=str, default="")
	parser.add_argument("--dry-run", action="store_true", default=False)
	args = parser.parse_args()

	tools_root = Path(args.tools_root).expanduser().resolve()
	isaaclab_sh = Path(args.isaaclab_sh).expanduser().resolve()
	generator_script = Path(args.generator_script).expanduser().resolve()
	output_base = Path(args.output_base).expanduser().resolve()

	if not tools_root.is_dir():
		raise FileNotFoundError(f"tools-root not found: {tools_root}")
	if not isaaclab_sh.is_file():
		raise FileNotFoundError(f"isaaclab.sh not found: {isaaclab_sh}")
	if not generator_script.is_file():
		raise FileNotFoundError(f"generator-script not found: {generator_script}")

	tool_items = _discover_tool_usds(tools_root)
	if len(tool_items) == 0:
		raise RuntimeError(f"No tool USD discovered under: {tools_root}")

	headless = args.headless and not args.no_headless
	mirror = args.mirror_tool_assets and not args.no_mirror_tool_assets
	enable_gravity = args.enable_gravity and not args.disable_gravity
	overwrite = args.overwrite and not args.no_overwrite

	print(f"[INFO] Discovered {len(tool_items)} tools")
	for n, p in tool_items:
		print(f"  - {n}: {p}")

	success = 0
	failures: list[tuple[str, int]] = []

	for tool_name, tool_usd in tool_items:
		safe_tool_name = _safe_name(tool_name)
		output_root = output_base
		output_usd = f"panda_instanceable_{safe_tool_name}.usd"

		cmd = [
			str(isaaclab_sh),
			"-p",
			str(generator_script),
			"--tool-usd",
			str(tool_usd),
			"--output-root",
			str(output_root),
			"--output-usd",
			output_usd,
		]
		if mirror:
			cmd.append("--mirror-tool-assets")
		if enable_gravity:
			cmd.append("--enable-gravity")
		if headless:
			cmd.append("--headless")
		if overwrite and success == 0 and len(failures) == 0:
			cmd.append("--overwrite")
		else:
			cmd.append("--reuse-output-root")

		env = os.environ.copy()
		if args.cuda_visible_devices.strip():
			env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices.strip()

		print("\n[RUN]", " ".join(cmd))
		if args.dry_run:
			continue

		proc = subprocess.run(cmd, env=env)
		if proc.returncode == 0:
			success += 1
		else:
			failures.append((tool_name, proc.returncode))

	print("\n[SUMMARY]")
	print(f"  success: {success}")
	print(f"  failed : {len(failures)}")
	for name, code in failures:
		print(f"    - {name}: exit code {code}")

	if len(failures) > 0:
		raise SystemExit(1)


if __name__ == "__main__":
	main()
