"""python prep_assets.py  ->  assets/prepped/<name>.usda: one rigid body each, joints off, colliders guaranteed."""
import os
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

ASSETS = Path(__file__).resolve().parent / "assets"
OUT = ASSETS / "prepped"
OUT.mkdir(exist_ok=True)

SRC = {
    "lubricant": "consumable_lubricant_grease_white_lithium_a01/simready_usd/sm_consumable_lubricant_grease_white_lithium_a01_01.usd",
    "sealant": "consumable_sealant_window_a01/simready_usd/sm_consumable_sealant_window_a01_01.usd",
    "spray": "consumable_spray_solvent_a01/simready_usd/sm_consumable_spray_solvent_a01_01.usd",
    "glasses": "ppe_eyeProtection_safetyGlasses_a01/simready_usd/sm_ppe_eyeProtection_safetyGlasses_a01_01.usd",
    "tape": "tool_tape_duct_a03/simready_usd/sm_tool_tape_duct_a03_01.usd",
    "pliers": "handTool_pliers_diagonal_cutting_a01/simready_usd/sm_handTool_pliers_diagonal_cutting_a01_01.usd",
    "wrench": "handTool_wrench_adjustable_a01/simready_usd/sm_handTool_wrench_adjustable_a01_01.usd",
}


def prep(name: str, rel: str) -> None:
    out = OUT / f"{name}.usda"
    if out.exists():
        out.unlink()
    stage = Usd.Stage.CreateNew(str(out))
    root = stage.DefinePrim("/RootNode", "Xform")
    root.GetReferences().AddReference(os.path.relpath(ASSETS / rel, OUT))
    stage.SetDefaultPrim(root)

    # 1. un-instance, so every descendant can be edited
    while True:
        inst = [p for p in Usd.PrimRange(root) if p.IsInstanceable()]
        if not inst:
            break
        for p in inst:
            p.SetInstanceable(False)

    # 2. joints off
    joints = [p for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.Joint)]
    for p in joints:
        p.SetActive(False)

    # 3. one rigid body, on the root
    bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI) and p != root]
    for p in bodies:
        p.RemoveAPI(UsdPhysics.RigidBodyAPI)
        p.RemoveAPI(UsdPhysics.MassAPI)
    UsdPhysics.RigidBodyAPI.Apply(root)

    # 4. colliders: keep the asset's own if it has any, else a convex hull per mesh
    n_col = sum(1 for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI))
    added = 0
    if n_col == 0:
        for p in Usd.PrimRange(root):
            if p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(p)
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
                added += 1
    stage.Save()
    print(f"{name:10s} bodies merged={len(bodies)} joints off={len(joints)} colliders found={n_col} added={added}")


for n, r in SRC.items():
    prep(n, r)