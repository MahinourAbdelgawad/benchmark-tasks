"""Author native joints at the physical interfaces of the protector, backing, and lifting tool."""
from task import POSITIONS, PATCHES, PITCH, TAB_Y, patch_y


def spawn_assembly(path, cfg, **kwargs):
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from pxr import UsdPhysics, Gf
    prim = spawn_from_usd(path, cfg, **kwargs)
    parent = prim.GetPath().GetParentPath()
    stage = prim.GetStage()

    def connect(name, kind, a, b, anchor, strength=None):
        joint = kind.Define(stage,parent.AppendChild(name))
        for index, body in enumerate((a,b)):
            getattr(joint,f'CreateBody{index}Rel')().SetTargets([parent.AppendChild(body)])
            offset = Gf.Vec3f(*(x-y for x,y in zip(anchor,POSITIONS[body])))
            getattr(joint,f'CreateLocalPos{index}Attr')(offset)
        joint.CreateExcludeFromArticulationAttr(True)
        if strength:
            joint.CreateBreakForceAttr(strength)
        return joint

    connect('vacuum_seal',UsdPhysics.FixedJoint,'glass','carrier',POSITIONS['carrier'])
    for i in range(PATCHES):
        name=f'liner{i}'
        pos=POSITIONS[name]
        connect(f'liner_bond{i}',UsdPhysics.FixedJoint,'glass',name,pos,100.)
        if i:
            hinge=connect(f'fold{i}',UsdPhysics.RevoluteJoint,f'liner{i-1}',name,
                          (pos[0],pos[1]-PITCH/2,pos[2]))
            hinge.CreateAxisAttr('X')
            hinge.CreateLowerLimitAttr(-135.)
            hinge.CreateUpperLimitAttr(135.)
            # Weak elastic recovery lets the free film uncurl after release.
            drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(),'angular')
            drive.CreateStiffnessAttr(.000012)
            drive.CreateDampingAttr(.000001)
    return prim

class PeelFront:
    """Bulk-supported patches resist lift. Only the loaded edge gets peel-strength bonds."""
    def __init__(self, stage, right):
        from yam_arena.constants import FINGER_JOINTS
        self.joints = [next(p for p in stage.Traverse() if p.GetName()==f'liner_bond{i}')
                       for i in range(PATCHES)]
        self.right = right
        self.fingers = right.find_joints(FINGER_JOINTS,preserve_order=True)[0]
        self.front = -1

    def update(self,state,command):
        import numpy as np
        from yam_arena.runtime import tcp_world
        glass = state['glass']
        tab = state['liner0']
        tcp = tcp_world(self.right)
        grip = tab[:3,3]+tab[:3,:3] @ [0,TAB_Y-patch_y(0),.016]
        self.gap = float(abs(self.right.data.joint_pos.torch[0,self.fingers].cpu().numpy()).sum())
        self.grip_error = float(np.linalg.norm(tcp-grip))
        if self.front < 0 and not (.0003 < self.gap < .032 and self.grip_error < .021):
            return
        if self.front < 0:
            # Subtract the initial difference between handle heights. Common-mode
            # motion lifts the sandwich; a relative downward draw loads its edge.
            if command.left_goal[2]-command.right_goal[2] < .009:
                return
            self.front = 0
            self.joints[0].GetAttribute('physics:breakForce').Set(.23)
        if self.front < PATCHES:
            strip = state[f'liner{self.front}']
            local = glass[:3,:3].T @ (strip[:3,3]-glass[:3,3])
            rest = [0,patch_y(self.front),-.00125]
            separated = np.linalg.norm(local-rest)>.003 or strip[:3,2] @ glass[:3,2]<.91
            if separated:
                self.front += 1
                if self.front < PATCHES:
                    self.joints[self.front].GetAttribute('physics:breakForce').Set(.23)
