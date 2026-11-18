"""Compare the closed clasp mesh and nominal YAM grasp in station coordinates."""

import numpy as np
from pxr import Usd, UsdGeom

import demo
import geometry
from scenekit import USD
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP


def triangles(stage, link, mesh_name):
    mesh = UsdGeom.Mesh.Get(stage, f"/RootNode/Geometry/{link}/{mesh_name}")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    points = np.array([geometry.place(transform.Transform(p)) for p in mesh.GetPointsAttr().Get()])
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not set(counts).issubset({3, 4}):
        raise ValueError("Expected triangles and quads in the toolbox meshes")
    starts = np.r_[0, counts.cumsum()[:-1]]
    return points[np.r_[indices[starts[:, None] + [0, 1, 2]], indices[starts[counts == 4, None] + [0, 2, 3]]]]


def section(mesh, origin, direction):
    """Signed intersections with a line through the mesh, in metres."""
    a, b, c = mesh.transpose(1, 0, 2)
    e1, e2 = b - a, c - a
    cross = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, cross)
    valid = abs(det) > 1e-12
    a, e1, e2, cross, det = (x[valid] for x in (a, e1, e2, cross, det))
    offset = origin - a
    u = np.einsum("ij,ij->i", offset, cross) / det
    v_cross = np.cross(offset, e1)
    v = (v_cross @ direction) / det
    t = np.einsum("ij,ij->i", e2, v_cross) / det
    return np.unique(t[(u >= 0) & (v >= 0) & (u + v <= 1)].round(8))


def main():
    stage = Usd.Stage.Open(str(USD))
    case = triangles(stage, "box_01_obj_00", "box_01_mesh_00")
    for side in ("left", "right"):
        clip = triangles(stage, f"{side}_clip", "lock_01_mesh_00")
        arm = demo.Arm(side)
        for _ in demo.grasp_clip(arm, geometry):
            pass
        tcp, rotation = ik.tcp_pose(arm.q, arm.base)
        pad_axis, approach = rotation[:, 0], rotation[:, 2]
        contact = geometry.clip_tab(side, 0)
        gap = -2 * FINGER_OPEN * arm.grip
        tip = tcp + PAD_TIP_PAST_TCP * approach

        print(f"{side} clasp bounds (m): {clip.min((0, 1)).round(6)} to {clip.max((0, 1)).round(6)}")
        nominal_axis = demo.clip_rotation(side, demo.CLIP_APPROACH_PITCH)[:, 0]
        offsets = section(clip, contact, nominal_axis)
        print(f"  mesh contact faces (m): {(contact + offsets[:, None] * nominal_axis).round(6).tolist()}")
        print(f"  mesh width between faces (mm): {np.ptp(offsets) * 1000:.2f}")
        print(f"  target TCP (m): {arm.pos.round(6)}, URDF FK TCP (m): {tcp.round(6)}")
        print(f"  pad-tip midpoint (m): {tip.round(6)}, commanded closing gap (mm): {gap * 1000:.2f}")

        # Test the two pad centrelines for a straight insertion. The real pads have finite width.
        half_gap = -FINGER_OPEN * demo.CLIP_INSERT_OPENING
        for sign in (-1, 1):
            lane = contact + sign * half_gap * nominal_axis
            for name, mesh in (("clip", clip), ("case", case)):
                hits = section(mesh, lane, approach)
                hits = hits[(hits >= -0.055) & (hits <= PAD_TIP_PAST_TCP - demo.CLIP_PAD_CONTACT)]
                if hits.size:
                    print(f"  insertion lane {sign:+d} meets {name} at approach offsets (mm): "
                          f"{(hits * 1000).round(2).tolist()}")


if __name__ == "__main__":
    main()
