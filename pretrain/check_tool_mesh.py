"""check_tool_mesh.py — Check tool mesh density and face extent.

Usage:  python check_tool_mesh.py --data-dir tmp_data/
"""
import argparse, torch, trimesh
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="tmp_data")
    args = parser.parse_args()

    files = sorted(Path(args.data_dir).rglob("*.pt"))
    if not files: print("No .pt files"); return

    d = torch.load(str(files[0]), map_location="cpu", weights_only=False)
    tool_mesh_path = d["tool_mesh_path"]
    print(f"Tool mesh: {tool_mesh_path}")

    mesh = trimesh.load(tool_mesh_path, force="mesh", process=False)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.int64)

    print(f"  Vertices: {verts.shape[0]}")
    print(f"  Faces:    {faces.shape[0]}")
    print(f"  Verts range: [{verts.min():.4f}, {verts.max():.4f}]")
    print(f"  Verts centroid: {verts.mean(0).numpy()}")

    # Check face edge lengths
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e01 = (v1 - v0).norm(dim=1)
    e02 = (v2 - v0).norm(dim=1)
    e12 = (v2 - v1).norm(dim=1)
    all_edges = torch.cat([e01, e02, e12])
    print(f"\n  Edge lengths: mean={all_edges.mean():.5f}  max={all_edges.max():.5f}  min={all_edges.min():.5f}")

    # Check face areas
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    areas = cross.norm(dim=1) * 0.5
    print(f"  Face areas:  mean={areas.mean():.6f}  max={areas.max():.6f}")
    print(f"  Total surface area: {areas.sum():.4f}")

    # Compare tool_pts_canonical vs mesh verts
    P_tool = d["tool_pts_canonical"]
    print(f"\n  tool_pts_canonical: {P_tool.shape}")
    print(f"  tool_pts range: [{P_tool.min():.4f}, {P_tool.max():.4f}]")

    # Check if canonical pts are consistent with mesh verts
    dists = torch.cdist(P_tool.unsqueeze(0), verts.unsqueeze(0)).squeeze(0)
    nn_to_mesh = dists.min(dim=1).values
    print(f"  NN dist (canonical pts → mesh verts): mean={nn_to_mesh.mean():.5f}  max={nn_to_mesh.max():.5f}")

    # Now the key test: recompute obj SDF with kaolin and compare
    # Load obj points, transform to tool frame, compute distance
    ci = 0
    P_obj = d["obj_pts_canonical"]
    R_obj = d["object_rotation"]
    z_shift = d["obj_z_shift"]
    R_tool = d["tool_rotations"][ci]
    t_tool = d["tool_translations"][ci]

    # obj canonical → world
    obj_world = P_obj @ R_obj.T
    obj_world = obj_world.clone()
    obj_world[:, 2] -= z_shift

    # world → tool canonical
    obj_in_tool = (obj_world - t_tool) @ R_tool  # (Q, 3)

    print(f"\n  obj_in_tool_frame range: [{obj_in_tool.min():.4f}, {obj_in_tool.max():.4f}]")
    print(f"  obj_in_tool_frame centroid: {obj_in_tool.mean(0).numpy()}")

    # NN from obj_in_tool to mesh verts (not faces)
    dists2 = torch.cdist(obj_in_tool.unsqueeze(0), verts.unsqueeze(0)).squeeze(0)
    nn_to_mesh_verts = dists2.min(dim=1).values
    print(f"  NN dist (obj_in_tool → mesh verts): mean={nn_to_mesh_verts.mean():.5f}  max={nn_to_mesh_verts.max():.5f}")

    stored_sdf = d["obj_pts_sdf"][ci]
    print(f"  stored obj_sdf:                     mean={stored_sdf.abs().mean():.5f}  max={stored_sdf.abs().max():.5f}")
    print(f"  ratio (stored / nn_to_mesh_verts):  mean={(stored_sdf.abs() / nn_to_mesh_verts.clamp(min=1e-6)).mean():.4f}")

if __name__ == "__main__":
    main()
