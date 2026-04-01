#!/usr/bin/env python3
"""Generate a Franka USD with removed gripper and welded custom tool.

This script performs the full pipeline:
1) Copy Franka source asset tree (to keep relative references valid)
2) Copy source USD into a new output USD
3) Deactivate gripper/finger prims and finger joints
4) Add custom tool as a reference under the attach link (default: panda_link8)
5) Apply rigid-body / collision / mass (and PhysX, if available) properties
6) Create a UsdPhysics.FixedJoint between link8 and the tool rigid body

Run inside Isaac Lab/Isaac Sim Python environment, for example:
  ./IsaacLab-2.2.0/isaaclab.sh -p ./IsaacLab-scripts/generate_franka.py -- \
	  --tool-usd /abs/path/to/tool.usd \
	  --output-root /mnt/afs/wangyuze/ToolGeneralist/static/franka/generated
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate Franka USD with custom welded tool.")

AppLauncher.add_app_launcher_args(parser)

parser.add_argument(
	"--src-root",
	type=str,
	default="/mnt/afs/wangyuze/ToolGeneralist/static/franka/Robots/FrankaEmika",
	help="Source Franka asset root that contains panda_instanceable.usd, Props/, Materials/",
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
	default="/mnt/afs/wangyuze/ToolGeneralist/static/franka/generated/Robots/FrankaEmika",
	help="Output asset root. Entire src-root is copied here.",
)
parser.add_argument(
	"--output-usd",
	type=str,
	default="panda_instanceable_tool.usd",
	help="Output robot USD file name under output-root.",
)
parser.add_argument("--tool-usd", type=str, required=True, help="Absolute path to tool USD.")
parser.add_argument(
	"--tool-root-prim",
	type=str,
	default="/root",
	help="Tool prim path to reference inside tool USD. For provided fork.usda use /root.",
)
parser.add_argument(
	"--mirror-tool-assets",
	action="store_true",
	default=False,
	help="Copy tool USD directory under output-root/ToolAssets to keep relative payloads valid.",
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
	help="Prefer collision meshes from this child scope under tool root (base.usda uses 'colliders').",
)
parser.add_argument(
	"--attach-link-name",
	type=str,
	default="panda_link7",
	help="Franka link name to attach tool. For provided panda.usda, use panda_link7.",
)
parser.add_argument("--tool-mount-name", type=str, default="tool_mount", help="Prim name for tool mount xform.")
parser.add_argument("--joint-name", type=str, default="tool_weld_joint", help="Fixed joint prim name.")

parser.add_argument("--tool-pos", type=str, default="0,0,0", help="Tool mount translation xyz.")
parser.add_argument("--tool-rot", type=str, default="1,0,0,0", help="Tool mount quaternion wxyz.")
parser.add_argument("--tool-scale", type=str, default="1,1,1", help="Tool mount scale xyz.")
parser.add_argument(
	"--strip-gripper-mode",
	type=str,
	choices=["remove", "deactivate"],
	default="remove",
	help="How to strip gripper from panda.usda-known prims.",
)

parser.add_argument("--mass-kg", type=float, default=0.2)
parser.add_argument("--enable-gravity", action="store_true", default=False)
parser.add_argument("--max-linear-velocity", type=float, default=1000.0)
parser.add_argument("--max-angular-velocity", type=float, default=1000.0)
parser.add_argument("--max-depenetration-velocity", type=float, default=5.0)
parser.add_argument("--contact-offset", type=float, default=0.005)
parser.add_argument("--rest-offset", type=float, default=0.0)

parser.add_argument("--overwrite", action="store_true", default=False)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("Isaac Lab Startup")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

# Known prim paths from local Panda USDA layout.
PANDA_ROOT_PATH = "/panda"
PANDA_LINK7_PATH = "/panda/panda_link7"
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
	"""Remove/deactivate gripper subtree using known panda.usda structure.

	Order matters: remove joint first, then rigid bodies.
	"""
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
	"""Collect collision prims with preference for a dedicated collider scope.

	For the provided tool base.usda, collision geometry lives under /root/colliders.
	We first try that scope, and fallback to all meshes under the tool mount.
	Returns: (prims, source_tag)
	"""
	if preferred_scope_name:
		scope_path = root_prim.GetPath().AppendChild(preferred_scope_name)
		scope_prim = root_prim.GetStage().GetPrimAtPath(scope_path)
		if scope_prim.IsValid():
			meshes = _collect_mesh_prims(scope_prim)
			if len(meshes) > 0:
				return meshes, f"scope:{preferred_scope_name}"

	return _collect_mesh_prims(root_prim), "scope:<all-meshes>"


def _apply_variant_if_exists(prim, variant_set_name: str, variant_name: str) -> bool:
	"""Set variant selection on prim if the variant set and variant exist."""
	vs_api = prim.GetVariantSets()
	if not vs_api.HasVariantSet(variant_set_name):
		return False
	vs = vs_api.GetVariantSet(variant_set_name)
	if variant_name not in set(vs.GetVariantNames()):
		return False
	vs.SetVariantSelection(variant_name)
	return True


def _mirror_tool_assets_if_requested(tool_usd: Path, out_root: Path, mirror: bool) -> Path:
	"""Optionally mirror tool folder under output root and return usd path to reference.

	This helps with exported tools that use relative payloads/references (e.g. configuration/*.usd).
	"""
	if not mirror:
		return tool_usd

	tool_src_dir = tool_usd.parent
	tool_dst_dir = out_root / "ToolAssets" / tool_src_dir.name
	if tool_dst_dir.exists():
		shutil.rmtree(tool_dst_dir)
	shutil.copytree(tool_src_dir, tool_dst_dir)
	return tool_dst_dir / tool_usd.name


def _get_authored_reference_path(referenced_tool_usd: Path, out_usd: Path, prefer_relative: bool) -> str:
	"""Compute the path string authored into AddReference().

	When prefer_relative is True (used with mirrored tool assets), author a path
	relative to the output USD layer for portability.
	"""
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

	# Auto-find first rigid body under tool mount
	for prim in Usd.PrimRange(tool_mount_prim):
		if prim.IsInstanceProxy():
			continue
		if prim.HasAPI(UsdPhysics.RigidBodyAPI):
			return prim

	# Heuristic for exported tool base.usda: prefer root link prim if present.
	for child_name in ["link_coacd_convex_piece_0", "link", "base", "tool", "body"]:
		cand = stage.GetPrimAtPath(tool_mount_prim.GetPath().AppendChild(child_name))
		if cand.IsValid() and not cand.IsInstanceProxy():
			return cand

	# Fallback: use mount prim itself
	return tool_mount_prim


def _apply_tool_physics(
	tool_mount_prim,
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
	# Ensure rigid body and mass API
	rb_api = UsdPhysics.RigidBodyAPI.Apply(rb_prim)
	rb_api.CreateRigidBodyEnabledAttr().Set(True)
	rb_api.CreateKinematicEnabledAttr().Set(False)

	mass_api = UsdPhysics.MassAPI.Apply(rb_prim)
	mass_api.CreateMassAttr().Set(mass_kg)

	# Ensure collision API on all meshes under tool mount
	for mesh_prim in mesh_prims:
		col_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
		col_api.CreateCollisionEnabledAttr().Set(True)

	# PhysX-specific parameters (if available)
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
	return fixed_joint


def main():
	src_root = Path(args.src_root).resolve()
	src_usd = src_root / args.src_usd
	out_root = Path(args.output_root).resolve()
	out_usd = out_root / args.output_usd
	tool_usd = Path(args.tool_usd).resolve()

	if not src_root.is_dir():
		raise FileNotFoundError(f"src-root not found: {src_root}")
	if not src_usd.is_file():
		raise FileNotFoundError(f"src-usd not found: {src_usd}")
	if not tool_usd.is_file():
		raise FileNotFoundError(f"tool-usd not found: {tool_usd}")

	# 1) Copy source tree
	if out_root.exists():
		if not args.overwrite:
			raise FileExistsError(f"output-root already exists: {out_root}. Use --overwrite to replace it.")
		shutil.rmtree(out_root)
	shutil.copytree(src_root, out_root)
	print("Source tree copied.")

	# 2) Copy source USD to a new output USD for editing
	shutil.copy2(out_root / args.src_usd, out_usd)
	print("Source USD copied to output USD.")

	# 3) Open stage
	stage = Usd.Stage.Open(str(out_usd))
	if stage is None:
		raise RuntimeError(f"Failed to open stage: {out_usd}")
	print("Stage opened.")

	# 3.1) Optional mirror for tool assets with relative payload/references.
	referenced_tool_usd = _mirror_tool_assets_if_requested(tool_usd, out_root, args.mirror_tool_assets)
	print(f"Tool USD path to reference: {referenced_tool_usd}")
	authored_tool_ref_path = _get_authored_reference_path(
		referenced_tool_usd=referenced_tool_usd,
		out_usd=out_usd,
		prefer_relative=args.mirror_tool_assets,
	)
	print(f"Authored tool reference path: {authored_tool_ref_path}")

	# 4) Strip gripper using known panda.usda paths.
	if not stage.GetPrimAtPath(Sdf.Path(PANDA_ROOT_PATH)).IsValid():
		raise RuntimeError(f"Expected panda root not found: {PANDA_ROOT_PATH}")
	strip_actions = _strip_known_gripper(stage, remove=(args.strip_gripper_mode == "remove"))
	print(f"Gripper removed, strip actions: {strip_actions}")

	# Fallback by name if known strip found nothing.
	if len(strip_actions) == 0:
		print("Fallback to deactivate prims by name")
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

	# 5) Find attachment link
	link_prim = _find_first_prim_with_name(stage, args.attach_link_name)
	if link_prim is None:
		raise RuntimeError(f"Attach link not found by name: {args.attach_link_name}")
	print(f"Attach link found: {link_prim.GetPath()}")
	
	# 6) Add tool reference under attach link
	tool_mount_path = link_prim.GetPath().AppendChild(args.tool_mount_name)
	tool_mount_prim = stage.DefinePrim(tool_mount_path, "Xform")
	tool_root_prim = args.tool_root_prim.strip() or "/root"
	tool_mount_prim.GetReferences().AddReference(authored_tool_ref_path, Sdf.Path(tool_root_prim))
	print(f"Tool reference added under {tool_mount_path}, source prim: {tool_root_prim}")

	# For variant-based tools (like provided fork.usda), select safe defaults.
	variant_changes = []
	for vs_name, vs_value in [
		("Physics", args.tool_variant_physics),
		("Sensor", args.tool_variant_sensor),
		("Robot", args.tool_variant_robot),
	]:
		if vs_value:
			if _apply_variant_if_exists(tool_mount_prim, vs_name, vs_value):
				variant_changes.append(f"{vs_name}={vs_value}")

	# Set mount transform
	pos = _parse_vec3(args.tool_pos)
	rot = _parse_quat_wxyz(args.tool_rot)
	scale = _parse_vec3(args.tool_scale)
	xform = UsdGeom.Xformable(tool_mount_prim)
	xform.ClearXformOpOrder()
	xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
	xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(rot[0], rot[1], rot[2], rot[3]))
	xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))
	print(f"Transform applied to tool mount: " + f"pos={pos}, rot={rot}, scale={scale}")
	
	# 7) Resolve tool rigid body prim and apply physics
	rb_hint = args.tool_rb_prim.strip() or None
	tool_rb_prim = _resolve_tool_rigid_body_prim(stage, tool_mount_prim, rb_hint)
	mesh_prims, collision_source = _collect_collision_prims(tool_mount_prim, args.tool_collider_scope)
	_apply_tool_physics(
		tool_mount_prim=tool_mount_prim,
		rb_prim=tool_rb_prim,
		mesh_prims=mesh_prims,
		mass_kg=args.mass_kg,
		enable_gravity=args.enable_gravity,
		max_linear_velocity=args.max_linear_velocity,
		max_angular_velocity=args.max_angular_velocity,
		max_depenetration_velocity=args.max_depenetration_velocity,
		contact_offset=args.contact_offset,
		rest_offset=args.rest_offset,
	)
	print("Physics applied")

	# 8) Create fixed joint
	_create_fixed_joint(
		stage=stage,
		parent_link_prim=link_prim,
		child_rb_prim=tool_rb_prim,
		joint_name=args.joint_name,
		# Defaults match original panda_hand_joint in the provided panda.usda.
		local_pos0=(0.0, 0.0, 0.107),
		local_rot0_wxyz=(0.9238795, 0.0, 0.0, -0.38268346),
		local_pos1=(0.0, 0.0, 0.0),
		local_rot1_wxyz=(1.0, 0.0, 0.0, 0.0),
	)

	# 9) Save
	stage.GetRootLayer().Save()

	print("[DONE] Generated Franka tool USD")
	print(f"  source root : {src_root}")
	print(f"  output root : {out_root}")
	print(f"  output usd  : {out_usd}")
	print(f"  tool usd    : {tool_usd}")
	print(f"  tool ref usd: {referenced_tool_usd}")
	print(f"  tool prim   : {tool_root_prim}")
	print(f"  attach link : {link_prim.GetPath()}")
	print(f"  tool mount  : {tool_mount_path}")
	print(f"  tool rb     : {tool_rb_prim.GetPath()}")
	print(f"  collision src: {collision_source}")
	print(f"  collision prims: {len(mesh_prims)}")
	if len(variant_changes) > 0:
		print(f"  tool variants: {', '.join(variant_changes)}")
	else:
		print("  tool variants: <none applied>")
	print(f"  stripped    : {len(strip_actions)}")
	for p in strip_actions:
		print(f"    - {p}")

if __name__ == "__main__":
	main()
	simulation_app.close()
