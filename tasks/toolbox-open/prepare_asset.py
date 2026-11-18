"""Build the Isaac Lab articulation as a small USD layer over the supplied asset."""

from pathlib import Path
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

HERE = Path(__file__).resolve().parent
SOURCE = next((HERE / "assets").glob("storage_toolbox*/simready_usd/sm*.usd"))
OUTPUT = HERE / "assets/toolbox_open.usdc"
CLIP_Y = 0.155


def split_mesh(mesh, sign):
    """Keep the faces for one clip and their face-varying normals and UVs."""
    points = np.asarray(mesh.GetPointsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    keep_faces = sign * points[indices[starts], 1] > 0
    keep_corners = np.repeat(keep_faces, counts)
    used, remapped = np.unique(indices[keep_corners], return_inverse=True)

    def subset(values, interpolation):
        selection = {"faceVarying": np.flatnonzero(keep_corners),
                     "uniform": np.flatnonzero(keep_faces), "vertex": used, "varying": used}
        if interpolation not in selection:
            return values
        return type(values)([values[int(i)] for i in selection[interpolation]])

    normals = mesh.GetNormalsAttr().Get()
    if normals:
        mesh.GetNormalsAttr().Set(subset(normals, mesh.GetNormalsInterpolation()))
    for primvar in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        if primvar.IsIndexed():
            primvar.SetIndices(subset(primvar.GetIndices(), primvar.GetInterpolation()))
        elif primvar.HasValue():
            primvar.Set(subset(primvar.Get(), primvar.GetInterpolation()))
    shifted = points[used].copy()
    shifted[:, 1] -= sign * CLIP_Y
    mesh.GetPointsAttr().Set([Gf.Vec3f(*map(float, p)) for p in shifted])
    mesh.GetExtentAttr().Set([Gf.Vec3f(*map(float, p)) for p in (shifted.min(0), shifted.max(0))])
    mesh.GetFaceVertexCountsAttr().Set(counts[keep_faces].tolist())
    mesh.GetFaceVertexIndicesAttr().Set(remapped.tolist())


def cube(stage, parent, name, center, size):
    shape = UsdGeom.Cube.Define(stage, parent.GetPath().AppendChild(name))
    shape.CreateSizeAttr(1)
    shape.AddTranslateOp().Set(Gf.Vec3d(*center))
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    shape.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return shape.GetPrim()


def filter_pair(first, second):
    UsdPhysics.FilteredPairsAPI.Apply(first).CreateFilteredPairsRel().AddTarget(second.GetPath())


def main():
    # Build in memory so a failed conversion cannot truncate the prior asset.
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/RootNode", "Xform")
    root.GetReferences().AddReference(str(SOURCE))
    root.GetVariantSets().GetVariantSet("PhysX").SetVariantSelection("Enabled")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    geometry = root.GetPath().AppendChild("Geometry")
    joints = root.GetPath().AppendChild("Joints")
    base, lid, handle, old_clip = [
        stage.GetPrimAtPath(geometry.AppendChild(f"{name}_01_obj_00"))
        for name in ("box", "top", "hand", "lock")
    ]
    for body in (base, lid, handle, old_clip):
        body.SetInstanceable(False)
    for name, parent, child in (("top", base, lid), ("hand", lid, handle)):
        hinge = UsdPhysics.RevoluteJoint.Get(stage, joints.AppendChild(f"joint_{name}_01_joint_00"))
        hinge.CreateBody0Rel().SetTargets([parent.GetPath()])
        hinge.CreateBody1Rel().SetTargets([child.GetPath()])
    # Isaac Lab fixes the base after spawning, using the actual world pose.
    UsdPhysics.ArticulationRootAPI.Apply(base)
    stage.GetPrimAtPath(joints.AppendChild("joint_lock_01_joint_00")).SetActive(False)
    old_clip.SetActive(False)

    base_mesh = base.GetChild("box_01_mesh_00")
    filter_pair(base, lid)
    filter_pair(base, handle)
    for sign, side in ((-1, "left"), (1, "right")):
        clip = stage.DefinePrim(geometry.AppendChild(f"{side}_clip"), "Xform")
        # Reference the original geometry and material network directly, independently of the disabled link.
        instances = SOURCE.parent / "payloads/instances_physx.usda"
        source_prim = Sdf.Path("/Instances/lock_01_mesh_00")
        clip.GetReferences().AddReference(str(instances), source_prim)
        clip.SetInstanceable(False)
        UsdGeom.Xformable(clip).AddTranslateOp().Set(Gf.Vec3d(0.1438092, sign * CLIP_Y, 0.1714074))
        UsdPhysics.RigidBodyAPI.Apply(clip)
        UsdPhysics.MassAPI.Apply(clip).CreateMassAttr(0.025)
        mesh = UsdGeom.Mesh(clip.GetChild("lock_01_mesh_00"))
        split_mesh(mesh, sign)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")

        hinge = UsdPhysics.RevoluteJoint.Define(stage, joints.AppendChild(f"{side}_clip_joint"))
        hinge.CreateAxisAttr("Y")
        hinge.CreateBody0Rel().SetTargets([lid.GetPath()])
        hinge.CreateBody1Rel().SetTargets([clip.GetPath()])
        hinge.CreateLocalPos0Attr(Gf.Vec3f(0.29788673, sign * CLIP_Y - 0.00003459, 0.006239608))
        hinge.CreateLowerLimitAttr(-85)
        hinge.CreateUpperLimitAttr(0)
        hinge.CreateCollisionEnabledAttr(False)

        # A small overlap along x lets the catch block vertical movement. Pulling the
        # bottom of the clip toward +x clears the lip before the lid begins moving.
        catch = cube(stage, base, f"{side}_catch", (0.147, sign * CLIP_Y, 0.137), (0.012, 0.024, 0.008))
        hook = cube(stage, clip, "hook", (0.001, 0, -0.045), (0.008, 0.022, 0.008))
        filter_pair(base_mesh, mesh.GetPrim())
        filter_pair(base_mesh, hook)
        filter_pair(catch, mesh.GetPrim())

    # Preserve portable references relative to the output layer, including the original materials.
    layer = stage.GetRootLayer()
    portable_layer = Sdf.Layer.CreateAnonymous()
    portable_layer.TransferContent(layer)
    for path in (SOURCE, instances):
        relative = "./" + os.path.relpath(path, OUTPUT.parent)
        portable_layer.UpdateCompositionAssetDependency(str(path), relative)
    portable_layer.Export(str(OUTPUT))


if __name__ == "__main__":
    main()
