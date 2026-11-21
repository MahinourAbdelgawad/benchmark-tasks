"""Continuous optical PET surface reconstructed from the native backing links."""
import numpy as np
from pxr import Sdf, UsdGeom, UsdShade, Vt
from scipy.interpolate import CubicHermiteSpline
from task import PATCHES, PITCH


class ReleaseSheet:
    def __init__(self, stage, state):
        self.mesh = UsdGeom.Mesh.Define(stage,'/World/ReleaseSheet')
        self.mesh.CreateDoubleSidedAttr(True)
        self.mesh.CreateSubdivisionSchemeAttr('none')
        self.samples = np.linspace(-.5,PATCHES-.5,4*PATCHES+1)
        n=len(self.samples)-1
        self.mesh.CreateFaceVertexCountsAttr([4]*n)
        self.mesh.CreateFaceVertexIndicesAttr([v for i in range(n) for v in (2*i,2*i+1,2*i+3,2*i+2)])
        mat=UsdShade.Material.Define(stage,'/World/LinerPlastic')
        shader=UsdShade.Shader.Define(stage,'/World/LinerPlastic/PET')
        shader.CreateIdAttr('UsdPreviewSurface')
        for key,value in [('diffuseColor',(.055,.66,.40)),('specularColor',(.65,.85,.82))]:
            shader.CreateInput(key,Sdf.ValueTypeNames.Color3f).Set(value)
        for key,value in [('opacity',.61),('roughness',.105),('ior',1.57),('clearcoat',.9)]:
            shader.CreateInput(key,Sdf.ValueTypeNames.Float).Set(value)
        shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim()).Bind(mat)
        self.update(state)

    def update(self,state):
        links=np.stack([state[f'liner{i}'] for i in range(PATCHES)])
        edges=links[:,:3,3][:,None,:]+np.array([-.035,.035])[None,:,None]*links[:,:3,0][:,None,:]
        tangent=np.repeat((PITCH*links[:,:3,1])[:,None,:],2,axis=1)
        # Extrapolate the two half-patches at their actual body orientations.
        knots=np.r_[-.5,np.arange(PATCHES),PATCHES-.5]
        edges=np.concatenate([edges[:1]-.5*tangent[:1],edges,edges[-1:]+.5*tangent[-1:]])
        tangent=np.concatenate([tangent[:1],tangent,tangent[-1:]])
        points=CubicHermiteSpline(knots,edges,tangent)(self.samples).reshape(-1,3).astype('float32')
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        self.mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack([points.min(0),points.max(0)])))
