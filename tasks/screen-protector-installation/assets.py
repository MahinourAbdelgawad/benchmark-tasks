"""Build task-scale compound PhysX bodies, with millimetre collision clearances."""
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from task import PITCH, TAB_Y, patch_y

DIR = Path(__file__).resolve().parent
TEAL = (.03,.34,.36)
BLACK = (.025,.03,.04)
BLUE = (.04,.15,.21)
WHITE = (.7,.78,.75)


def build(name, mass, parts):
    stage = Usd.Stage.CreateNew(str(DIR / (name+'.usda')))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, '/Body').GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
    for label, size, pos, color, friction in parts:
        cube = UsdGeom.Cube.Define(stage, '/Body/'+label)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        prim = cube.GetPrim()
        if label == 'backing':
            cube.MakeInvisible()
        if friction is not None:
            UsdPhysics.CollisionAPI.Apply(prim)
        prim.AddAppliedSchema('PhysxCollisionAPI')
        prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.0003)
        prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
        mat = UsdShade.Material.Define(stage, '/Body/Materials/'+label)
        phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        if friction is not None:
            phys.CreateStaticFrictionAttr(friction)
            phys.CreateDynamicFrictionAttr(friction)
        shader = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild('Surface'))
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(color)
        if label == 'protector':
            shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(.45)
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.14)
        shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
    stage.GetRootLayer().Save()


def main():
    build('phone', .24, [
        ('phone', (.076,.156,.018), (0,0,.009), BLACK,.85),
        ('display',(.071,.147,.00005),(0,0,.01802),(.035,.13,.19),None),
        ('camera_island',(.016,.004,.00006),(0,.062,.01806),BLACK,None),
        ('home_indicator',(.022,.001,.00006),(0,-.065,.01806),WHITE,None)])
    build('glass',.018,[('protector',(.070,.145,.0016),(0,0,0),BLUE,.3)])
    build('carrier',.025,[('vacuum_pad',(.026,.030,.003),(0,0,.0015),BLACK,.6),
                          ('handle',(.026,.012,.034),(0,0,.019),TEAL,1.)])
    for name in ('strip','tab'):
        parts = [('backing',(.070,PITCH-.00015,.0009),(0,0,0),TEAL,.2)]
        if name == 'tab':
            y = TAB_Y-patch_y(0)
            parts.extend([('extension',(.023,.025,.0009),(0,y+.012,0),(.04,.66,.48),.6),
                          ('folded_tab',(.023,.004,.024),(0,y,.0115),(.04,.66,.48),1.),
                          ('printed_band',(.020,.0001,.002),(0,y-.0021,.020),WHITE,None)])
        build(name,.005 if name=='tab' else .002,parts)


if __name__ == '__main__':
    main()
