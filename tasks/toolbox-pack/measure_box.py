"""python measure_box.py assets/.../D00443.usd"""
import sys

from pxr import Usd, UsdGeom

stage = Usd.Stage.Open(sys.argv[1])
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                   UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide])
for p in stage.Traverse():
    if "/Colliders_00/" in str(p.GetPath()) and p.IsA(UsdGeom.Gprim):
        r = cache.ComputeWorldBound(p).ComputeAlignedRange()
        mn, mx = r.GetMin(), r.GetMax()
        size = [mx[i] - mn[i] for i in range(3)]
        ctr = [(mx[i] + mn[i]) / 2 for i in range(3)]
        flag = "  <-- divider?" if min(size[:2]) < 0.03 and max(size[:2]) > 0.15 else ""
        print(p.GetName(), "centre", [round(c, 3) for c in ctr], "size", [round(s, 3) for s in size], flag)