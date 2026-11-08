"""Print what a USD asset's physics actually contains: rigid bodies, articulation roots, joints and limits, collision
approximations and size. Runs on your own machine, no GPU:

    pip install usd-core
    python inspect_asset.py path/to/sm_storage_toolbox_standard_hardCase_d01_01.usd
    python inspect_asset.py path/to/sm_storage_toolbox_standard_hardCase_d01_01.usd --set PhysX=Enabled

Open the main .usd (not physics.usda on its own) so the payloads compose, and keep the whole downloaded folder
together. It prints the asset's variant sets first. ``--set NAME=VALUE`` (repeatable) picks a variant on the default
prim before inspecting, so you can compare what the asset contains with, say, the PhysX variant on or off. It also
works on a lone physics.usda: you then see only what that file adds.
"""

from __future__ import annotations

import math
import sys

from pxr import Usd, UsdGeom, UsdPhysics


def _deg(x):
    return "inf" if x is None or math.isinf(x) else f"{x:.1f}"


def _variants(stage) -> None:
    print("\nvariant sets (selected / options):")
    found = 0
    for p in stage.TraverseAll():
        sets = p.GetVariantSets()
        for name in sets.GetNames():
            vs = sets.GetVariantSet(name)
            print(f"  {p.GetPath()}  {name} = {vs.GetVariantSelection()!r}  options: {vs.GetVariantNames()}")
            found += 1
    if not found:
        print("  none")


def main(path: str, chosen: list[str]) -> int:
    stage = Usd.Stage.Open(path)
    print(f"file: {path}")
    print(f"metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)} upAxis={UsdGeom.GetStageUpAxis(stage)} "
          f"defaultPrim={stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None}")
    _variants(stage)
    for item in chosen:
        name, _, value = item.partition("=")
        root = stage.GetDefaultPrim()
        vs = root.GetVariantSets().GetVariantSet(name) if root else None
        if not vs or value not in vs.GetVariantNames():
            print(f"\n--set {item}: no such variant set or option on {root.GetPath() if root else 'the default prim'}")
            return 1
        vs.SetVariantSelection(value)
        print(f"\nselected {name} = {value}")
    prims = list(stage.TraverseAll())

    roots = [p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print("\narticulation roots:", [str(p.GetPath()) for p in roots] or "NONE")

    bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    print(f"\nrigid bodies ({len(bodies)}):")
    for p in bodies:
        mass = UsdPhysics.MassAPI(p).GetMassAttr().Get() if p.HasAPI(UsdPhysics.MassAPI) else None
        print(f"  {p.GetPath()}  mass={mass if mass else 'from density'}")

    print("\ncolliders (path -> approximation):")
    n = 0
    for p in prims:
        if p.HasAPI(UsdPhysics.CollisionAPI):
            approx = UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() if p.IsA(UsdGeom.Mesh) else p.GetTypeName()
            print(f"  {p.GetPath()} -> {approx}")
            n += 1
    if not n:
        print("  none found (collision may only appear in the PhysX payload variant)")

    body_paths = {str(p.GetPath()) for p in bodies}
    joints = [p for p in prims if p.IsA(UsdPhysics.Joint)]
    print(f"\njoints ({len(joints)}):")
    for p in joints:
        j = UsdPhysics.Joint(p)
        b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
        b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
        line = f"  {p.GetName()} [{p.GetTypeName()}]  {b0} -> {b1}"
        if p.IsA(UsdPhysics.RevoluteJoint):
            r = UsdPhysics.RevoluteJoint(p)
            line += f"  axis={r.GetAxisAttr().Get()} limits=({_deg(r.GetLowerLimitAttr().Get())}, {_deg(r.GetUpperLimitAttr().Get())}) deg"
        elif p.IsA(UsdPhysics.PrismaticJoint):
            r = UsdPhysics.PrismaticJoint(p)
            line += f"  axis={r.GetAxisAttr().Get()} limits=({r.GetLowerLimitAttr().Get()}, {r.GetUpperLimitAttr().Get()}) m"
        print(line)
        for t in b0 + b1:
            # a joint target may be a mesh under a rigid body; flag targets with no rigid body at or above them
            probe = t
            while probe and probe not in body_paths:
                probe = probe.rsplit("/", 1)[0] if "/" in probe.strip("/") else ""
            if not probe:
                print(f"    WARNING: {t} is not under any rigid body")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
    if not rng.IsEmpty():
        lo, hi = rng.GetMin(), rng.GetMax()
        size = [(hi[i] - lo[i]) * UsdGeom.GetStageMetersPerUnit(stage) for i in range(3)]
        print(f"\nbounding box (m): min={tuple(round(v, 4) for v in lo)} max={tuple(round(v, 4) for v in hi)}")
        print(f"size x,y,z = {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m")
    else:
        print("\nbounding box: empty (geometry did not load; are the payloads next to this file?)")

    print()
    if not roots:
        print("NOTE: no ArticulationRootAPI. Isaac Lab will not treat this as an articulation as it stands; see README.md,"
              " 'Debugging'. The PhysX payload may add one: open payloads/PhysX/enabled.usda to check.")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-") or any(a != "--set" and not "=" in a for a in args[1:]):
        sys.exit(__doc__)
    picks = [a for a in args[1:] if a != "--set"]
    sys.exit(main(args[0], picks))