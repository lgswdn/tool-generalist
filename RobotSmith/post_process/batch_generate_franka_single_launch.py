#!/usr/bin/env python3
"""Batch-generate Franka USDs with different tools in a single IsaacLab launch.

This script merges the behavior of:
- generate_franka.py (single-tool generation logic)
- batch_generate_franka.py (tool discovery + batch loop)

Key difference:
- IsaacLab is launched only once, then all tool USDs are processed in one run.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher


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


parser = argparse.ArgumentParser(
	description="Batch-generate Franka USDs for many tools in one IsaacLab process."
)

AppLauncher.add_app_launcher_args(parser)

# Batch discovery options
parser.add_argument(
	"--tools-root",
	type=str,
	default="/mnt/afs/wangyuze/ToolGeneralist/static/objects_usd",
	help="Directory containing tool USD folders/files.",
)
parser.add_argument(
	"--output-usd-prefix",
	type=str,
	default="panda_instanceable_",
	help="Prefix of generated USD name: <prefix><tool_name>.usd",
)
parser.add_argument(
	"--fail-fast",
	action="store_true",
	default=False,
	help="Stop immediately when one tool generation fails.",
)
parser.add_argument("--dry-run", action="store_true", default=False)

# Franka source/output options (from generate_franka.py)
parser.add_argument(
	"--src-root",
	type=str,
	default="/mnt/afs/wangyuze/ToolGeneralist/static/franka/Robots/FrankaEmika",
	help="Source Franka asset root that contains panda_instanceable.usd, Props/, Materials/.",
)
parser.add_argument(
	"--src-usd",
	type=str,
	default="panda_instanceable.usd",
	help="Source robot USD file name under src-root.",
)
parser.add_argument(
	"--output-root",
	type=str,
	default="/mnt/afs/wangyuze/ToolGeneralist/static/franka/generated/Robots",
	help="Output asset root. Entire src-root is copied here once.",
)
parser.add_argument(
	"--overwrite",
	action="store_true",
	default=True,
	help="Delete output-root and recreate from src-root before generation.",
)
parser.add_argument(
	"--no-overwrite",
	action="store_true",
	default=False,
	help="Disable overwrite behavior.",
)
parser.add_argument(
	"--reuse-output-root",
	action="store_true",
	default=False,
	help="Reuse existing output-root and merge missing base assets.",
)

# Tool reference and variants
parser.add_argument(
	"--tool-root-prim",
	type=str,
	default="/root",
	help="Tool prim path to reference inside tool USD.",
)
parser.add_argument(
	"--mirror-tool-assets",
	action="store_true",
	default=False,
	help="Copy each tool USD directory under output-root/ToolAssets for relative refs.",
)
parser.add_argument(
	"--tool-variant-physics",
	type=str,
	default="PhysX",
	help="Variant selection for tool variant set 'Physics' if present.",
)
parser.add_argument(
	"--tool-variant-sensor",
	type=str,
	default="None",
	help="Variant selection for tool variant set 'Sensor' if present.",
)
parser.add_argument(
	"--tool-variant-robot",
	type=str,
	default="None",
	help="Variant selection for tool variant set 'Robot' if present.",
)
parser.add_argument(
	"--tool-rb-prim",
	type=str,
	default="",
	help="Optional rigid-body prim under tool mount. Absolute or relative (e.g. 'base').",
)
parser.add_argument(
	"--tool-collider-scope",
	type=str,
	default="colliders",
	help="Prefer collision meshes from this child scope under tool root.",
)

# Attachment / transform
parser.add_argument(
	"--attach-link-name",
	type=str,
	default="panda_link7",
	help="Franka link name to attach tool.",
)
parser.add_argument("--tool-mount-name", type=str, default="tool_mount", help="Prim name for tool mount xform.")
parser.add_argument("--joint-name", type=str, default="tool_weld_joint", help="Fixed joint prim name.")
parser.add_argument(
	"--tool-pos",
	type=str,
	default="0.08799998,-4.9709342e-8,0.926",
	help="Tool mount translation xyz.",
)
parser.add_argument(
	"--tool-rot",
	type=str,
	default="-1.4551854e-11,0.9238795,0.38268346,-4.6566123e-10",
	help="Tool mount quaternion wxyz.",
)
parser.add_argument(
	"--tool-scale",
	type=str,
	default="0.1,0.1,0.1",
	help="Tool mount scale xyz.",
)
parser.add_argument(
	"--strip-gripper-mode",
	type=str,
	choices=["remove", "deactivate"],
	default="remove",
	help="How to strip gripper from panda.usda-known prims.",
)

# Physics
parser.add_argument("--mass-kg", type=float, default=0.2)
parser.add_argument("--enable-gravity", action="store_true", default=True)
parser.add_argument("--disable-gravity", action="store_true", default=False)
parser.add_argument("--max-linear-velocity", type=float, default=1000.0)
parser.add_argument("--max-angular-velocity", type=float, default=1000.0)
parser.add_argument("--max-depenetration-velocity", type=float, default=5.0)
parser.add_argument("--contact-offset", type=float, default=0.005)
parser.add_argument("--rest-offset", type=float, default=0.0)

args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("Isaac Lab Startup (single launch batch mode)")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

# Known prim paths from local Panda USDA layout.
PANDA_ROOT_PATH = "/panda"
PANDA_HAND_JOINT_PATH = "/panda/panda_link7/panda_hand_joint"
PANDA_HAND_PATH = "/panda/panda_hand"
PANDA_LEFT_FINGER_PATH = "/panda/panda_leftfinger"
PANDA_RIGHT_FINGER_PATH = "/panda/panda_rightfinger"


def _parse_vec3(text: str) -> tuple[float, float, float]:
	vals = [float(x.strip()) for x in text.split(",")]
	if len(vals) != 3:
		raise ValueError(f"Expected 3 values, got: {text}")
	return vals[0], vals[1], vals[2]


def _parse_quat_wxyz(text: str) -> tuple[float, float, float, float]:
	vals = [float(x.strip()) for x in text.split(",")]
	if len(vals) != 4:
		raise ValueError(f"Expected 4 values, got: {text}")
	return vals[0], vals[1], vals[2], vals[3]


def _find_first_prim_with_name(stage: Usd.Stage, name: str):
	for prim in stage.TraverseAll():
		if prim.GetName() == name:
			return prim
	return None


def _deactivate_prims_by_name(stage: Usd.Stage, names: set[str]) -> list[str]:
	deactivated = []
	for prim in stage.TraverseAll():
		if prim.GetName() in names and prim.IsActive():
			prim.SetActive(False)
			deactivated.append(prim.GetPath().pathString)
	return deactivated


def _remove_or_deactivate_prim(stage: Usd.Stage, prim_path: str, remove: bool) -> bool:
	prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
	if not prim.IsValid():
		return False
	if remove:
		stage.RemovePrim(prim.GetPath())
	else:
		prim.SetActive(False)
	return True


def _strip_known_gripper(stage: Usd.Stage, remove: bool = True) -> list[str]:
	actions = []
	for prim_path in [
		PANDA_HAND_JOINT_PATH,
		PANDA_HAND_PATH,
		PANDA_LEFT_FINGER_PATH,
		PANDA_RIGHT_FINGER_PATH,
	]:
		ok = _remove_or_deactivate_prim(stage, prim_path, remove=remove)
		if ok:
			actions.append(("removed" if remove else "deactivated") + f": {prim_path}")
	return actions


def _collect_mesh_prims(root_prim) -> list:
	meshes = []
	for prim in Usd.PrimRange(root_prim):
		if prim.IsInstanceProxy():
			continue
		if prim.IsA(UsdGeom.Mesh):
			meshes.append(prim)
	return meshes


def _collect_collision_prims(root_prim, preferred_scope_name: str = "colliders") -> tuple[list, str]:
	if preferred_scope_name:
		scope_path = root_prim.GetPath().AppendChild(preferred_scope_name)
		scope_prim = root_prim.GetStage().GetPrimAtPath(scope_path)
		if scope_prim.IsValid():
			meshes = _collect_mesh_prims(scope_prim)
			if len(meshes) > 0:
				return meshes, f"scope:{preferred_scope_name}"

	return _collect_mesh_prims(root_prim), "scope:<all-meshes>"


def _apply_variant_if_exists(prim, variant_set_name: str, variant_name: str) -> bool:
	vs_api = prim.GetVariantSets()
	if not vs_api.HasVariantSet(variant_set_name):
		return False
	vs = vs_api.GetVariantSet(variant_set_name)
	if variant_name not in set(vs.GetVariantNames()):
		return False
	vs.SetVariantSelection(variant_name)
	return True


def _mirror_tool_assets_if_requested(tool_usd: Path, out_root: Path, mirror: bool) -> Path:
	if not mirror:
		return tool_usd

	tool_src_dir = tool_usd.parent
	tool_dst_dir = out_root / "ToolAssets" / tool_src_dir.name
	if tool_dst_dir.exists():
		shutil.rmtree(tool_dst_dir)
	shutil.copytree(tool_src_dir, tool_dst_dir)
	return tool_dst_dir / tool_usd.name


def _get_authored_reference_path(referenced_tool_usd: Path, out_usd: Path, prefer_relative: bool) -> str:
	if not prefer_relative:
		return str(referenced_tool_usd)

	rel = os.path.relpath(referenced_tool_usd, start=out_usd.parent)
	return rel.replace("\\", "/")


def _resolve_tool_rigid_body_prim(stage: Usd.Stage, tool_mount_prim, tool_rb_hint: str | None):
	if tool_rb_hint:
		if tool_rb_hint.startswith("/"):
			rb_path = Sdf.Path(tool_rb_hint)
		else:
			rb_path = tool_mount_prim.GetPath().AppendPath(Sdf.Path(tool_rb_hint))
		rb_prim = stage.GetPrimAtPath(rb_path)
		if not rb_prim.IsValid():
			raise RuntimeError(f"tool rigid body prim not found: {rb_path}")
		return rb_prim

	for prim in Usd.PrimRange(tool_mount_prim):
		if prim.IsInstanceProxy():
			continue
		if prim.HasAPI(UsdPhysics.RigidBodyAPI):
			return prim

	for child_name in ["link_coacd_convex_piece_0", "link", "base", "tool", "body"]:
		cand = stage.GetPrimAtPath(tool_mount_prim.GetPath().AppendChild(child_name))
		if cand.IsValid() and not cand.IsInstanceProxy():
			return cand

	return tool_mount_prim


def _apply_tool_physics(
	rb_prim,
	mesh_prims: list,
	mass_kg: float,
	enable_gravity: bool,
	max_linear_velocity: float,
	max_angular_velocity: float,
	max_depenetration_velocity: float,
	contact_offset: float,
	rest_offset: float,
):
	rb_api = UsdPhysics.RigidBodyAPI.Apply(rb_prim)
	rb_api.CreateRigidBodyEnabledAttr().Set(True)
	rb_api.CreateKinematicEnabledAttr().Set(False)

	mass_api = UsdPhysics.MassAPI.Apply(rb_prim)
	mass_api.CreateMassAttr().Set(mass_kg)

	for mesh_prim in mesh_prims:
		col_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
		col_api.CreateCollisionEnabledAttr().Set(True)

	physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(rb_prim)
	physx_rb.CreateDisableGravityAttr().Set(not enable_gravity)
	physx_rb.CreateMaxLinearVelocityAttr().Set(max_linear_velocity)
	physx_rb.CreateMaxAngularVelocityAttr().Set(max_angular_velocity)
	physx_rb.CreateMaxDepenetrationVelocityAttr().Set(max_depenetration_velocity)

	for mesh_prim in mesh_prims:
		physx_col = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
		physx_col.CreateContactOffsetAttr().Set(contact_offset)
		physx_col.CreateRestOffsetAttr().Set(rest_offset)


def _create_fixed_joint(
	stage: Usd.Stage,
	parent_link_prim,
	child_rb_prim,
	joint_name: str,
	local_pos0: tuple[float, float, float],
	local_rot0_wxyz: tuple[float, float, float, float],
	local_pos1: tuple[float, float, float],
	local_rot1_wxyz: tuple[float, float, float, float],
):
	joint_path = parent_link_prim.GetPath().AppendChild(joint_name)
	fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
	fixed_joint.CreateBody0Rel().SetTargets([parent_link_prim.GetPath()])
	fixed_joint.CreateBody1Rel().SetTargets([child_rb_prim.GetPath()])

	fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
	fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))

	w0, x0, y0, z0 = local_rot0_wxyz
	w1, x1, y1, z1 = local_rot1_wxyz
	fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(w0, x0, y0, z0))
	fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(w1, x1, y1, z1))


def _prepare_output_root(src_root: Path, out_root: Path, overwrite: bool, reuse_output_root: bool) -> None:
	if out_root.exists():
		if overwrite:
			shutil.rmtree(out_root)
			shutil.copytree(src_root, out_root)
			print(f"[INFO] Recreated output root: {out_root}")
		elif reuse_output_root:
			shutil.copytree(src_root, out_root, dirs_exist_ok=True)
			print(f"[INFO] Reused output root (merged base assets): {out_root}")
		else:
			raise FileExistsError(
				f"output-root already exists: {out_root}. Use --overwrite or --reuse-output-root."
			)
	else:
		shutil.copytree(src_root, out_root)
		print(f"[INFO] Created output root: {out_root}")


def _generate_one(stage_path: Path, tool_usd: Path, tool_name: str, output_root: Path) -> None:
	stage = Usd.Stage.Open(str(stage_path))
	if stage is None:
		raise RuntimeError(f"Failed to open stage: {stage_path}")

	referenced_tool_usd = _mirror_tool_assets_if_requested(tool_usd, output_root, args.mirror_tool_assets)
	authored_tool_ref_path = _get_authored_reference_path(
		referenced_tool_usd=referenced_tool_usd,
		out_usd=stage_path,
		prefer_relative=args.mirror_tool_assets,
	)

	if not stage.GetPrimAtPath(Sdf.Path(PANDA_ROOT_PATH)).IsValid():
		raise RuntimeError(f"Expected panda root not found: {PANDA_ROOT_PATH}")

	strip_actions = _strip_known_gripper(stage, remove=(args.strip_gripper_mode == "remove"))
	if len(strip_actions) == 0:
		to_deactivate = {
			"panda_hand",
			"panda_leftfinger",
			"panda_rightfinger",
			"panda_finger_joint1",
			"panda_finger_joint2",
			"panda_hand_joint",
		}
		for p in _deactivate_prims_by_name(stage, to_deactivate):
			strip_actions.append(f"deactivated: {p}")

	link_prim = _find_first_prim_with_name(stage, args.attach_link_name)
	if link_prim is None:
		raise RuntimeError(f"Attach link not found by name: {args.attach_link_name}")

	panda_root_prim = stage.GetPrimAtPath(Sdf.Path(PANDA_ROOT_PATH))
	tool_mount_path = panda_root_prim.GetPath().AppendChild(args.tool_mount_name)
	tool_mount_prim = stage.DefinePrim(tool_mount_path, "Xform")

	tool_root_prim = args.tool_root_prim.strip() or "/root"
	tool_mount_prim.GetReferences().AddReference(authored_tool_ref_path, Sdf.Path(tool_root_prim))

	variant_changes = []
	for vs_name, vs_value in [
		("Physics", args.tool_variant_physics),
		("Sensor", args.tool_variant_sensor),
		("Robot", args.tool_variant_robot),
	]:
		if vs_value and _apply_variant_if_exists(tool_mount_prim, vs_name, vs_value):
			variant_changes.append(f"{vs_name}={vs_value}")

	pos = _parse_vec3(args.tool_pos)
	rot = _parse_quat_wxyz(args.tool_rot)
	scale = _parse_vec3(args.tool_scale)
	xform = UsdGeom.Xformable(tool_mount_prim)
	xform.ClearXformOpOrder()
	xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
	xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(rot[0], rot[1], rot[2], rot[3]))
	xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))

	rb_hint = args.tool_rb_prim.strip() or None
	tool_rb_prim = _resolve_tool_rigid_body_prim(stage, tool_mount_prim, rb_hint)
	mesh_prims, collision_source = _collect_collision_prims(tool_mount_prim, args.tool_collider_scope)

	enable_gravity = args.enable_gravity and not args.disable_gravity
	_apply_tool_physics(
		rb_prim=tool_rb_prim,
		mesh_prims=mesh_prims,
		mass_kg=args.mass_kg,
		enable_gravity=enable_gravity,
		max_linear_velocity=args.max_linear_velocity,
		max_angular_velocity=args.max_angular_velocity,
		max_depenetration_velocity=args.max_depenetration_velocity,
		contact_offset=args.contact_offset,
		rest_offset=args.rest_offset,
	)

	_create_fixed_joint(
		stage=stage,
		parent_link_prim=link_prim,
		child_rb_prim=tool_rb_prim,
		joint_name=args.joint_name,
		local_pos0=(0.0, 0.0, 0.107),
		local_rot0_wxyz=(0.9238795, 0.0, 0.0, -0.38268346),
		local_pos1=(0.0, 0.0, 0.0),
		local_rot1_wxyz=(1.0, 0.0, 0.0, 0.0),
	)

	stage.GetRootLayer().Save()

	print(f"[DONE] {tool_name}")
	print(f"  output usd   : {stage_path}")
	print(f"  tool usd     : {tool_usd}")
	print(f"  tool ref usd : {referenced_tool_usd}")
	print(f"  attach link  : {link_prim.GetPath()}")
	print(f"  tool mount   : {tool_mount_path}")
	print(f"  tool rb      : {tool_rb_prim.GetPath()}")
	print(f"  collision src: {collision_source}")
	print(f"  collision n  : {len(mesh_prims)}")
	if len(variant_changes) > 0:
		print(f"  variants     : {', '.join(variant_changes)}")
	print(f"  stripped n   : {len(strip_actions)}")


def main() -> None:
	tools_root = Path(args.tools_root).expanduser().resolve()
	src_root = Path(args.src_root).expanduser().resolve()
	out_root = Path(args.output_root).expanduser().resolve()
	src_usd = src_root / args.src_usd

	if not tools_root.is_dir():
		raise FileNotFoundError(f"tools-root not found: {tools_root}")
	if not src_root.is_dir():
		raise FileNotFoundError(f"src-root not found: {src_root}")
	if not src_usd.is_file():
		raise FileNotFoundError(f"src-usd not found: {src_usd}")

	tool_items = _discover_tool_usds(tools_root)
	if len(tool_items) == 0:
		raise RuntimeError(f"No tool USD discovered under: {tools_root}")

	overwrite = args.overwrite and not args.no_overwrite
	reuse_output_root = args.reuse_output_root or not overwrite

	print(f"[INFO] Discovered {len(tool_items)} tools")
	for n, p in tool_items:
		print(f"  - {n}: {p}")

	_prepare_output_root(src_root, out_root, overwrite=overwrite, reuse_output_root=reuse_output_root)

	success = 0
	failures: list[tuple[str, str]] = []

	for tool_name, tool_usd in tool_items:
		safe_tool_name = _safe_name(tool_name)
		output_usd_name = f"{args.output_usd_prefix}{safe_tool_name}.usd"
		output_usd_path = out_root / output_usd_name

		print("\n[RUN]", tool_name)
		print(f"  source usd -> {src_usd}")
		print(f"  output usd -> {output_usd_path}")

		if args.dry_run:
			continue

		try:
			shutil.copy2(out_root / args.src_usd, output_usd_path)
			_generate_one(output_usd_path, tool_usd, tool_name, out_root)
			success += 1
		except Exception as e:  # noqa: BLE001
			failures.append((tool_name, str(e)))
			print(f"[ERROR] {tool_name}: {e}")
			if args.fail_fast:
				break

	print("\n[SUMMARY]")
	print(f"  success: {success}")
	print(f"  failed : {len(failures)}")
	for name, reason in failures:
		print(f"    - {name}: {reason}")

	if len(failures) > 0:
		raise SystemExit(1)


if __name__ == "__main__":
	print("MAIN")
	#try:
	main()
	#finally:
	simulation_app.close()
