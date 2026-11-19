"""Native PhysX deformable balls for the Arena task-file adapter."""
from dataclasses import dataclass
import numpy as np
from yam_arena.task_file import Box, ObjectState


@dataclass(frozen=True)
class TennisBall(Box):
    modulus: float = 1e5

    @property
    def free(self):
        # Arena's rigid-root jitter cannot reposition a nodal asset.
        return False


def ball_asset(spec):
    import isaaclab.sim as sim
    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

    radius = spec.size[0] / 2
    density = 0.057 / (4 * np.pi * radius**3 / 3)
    mesh = sim.MeshSphereCfg(
        radius=radius,
        deformable_props=PhysxDeformableBodyPropertiesCfg(solver_position_iteration_count=32),
        physics_material=PhysxDeformableBodyMaterialCfg(
            youngs_modulus=spec.modulus, poissons_ratio=0.35, density=density),
        visual_material=sim.PreviewSurfaceCfg(diffuse_color=spec.color, roughness=0.95),
    )
    return DeformableObject(name=spec.name, spawner_cfg=mesh,
                            initial_pose=Pose(position_xyz=spec.pos, rotation_xyzw=(0, 0, 0, 1)))


def install():
    from yam_arena import scene
    if getattr(scene, "_tennis_ball_adapter", False):
        return
    spawn, read = scene.task_object, scene.object_states
    names = set()

    def task_object(spec):
        if isinstance(spec, TennisBall):
            names.add(spec.name)
            return ball_asset(spec)
        return spawn(spec)

    def array(tensor):
        return getattr(tensor, "torch", tensor).detach().cpu().numpy()[0]

    def object_states(env, requested):
        base = getattr(env, "unwrapped", env)
        ordinary = [name for name in requested if name not in names]
        states = read(env, ordinary) if ordinary else {}
        for name in names.intersection(requested):
            data = base.scene[name].data
            nodes = array(data.nodal_pos_w) - array(base.scene.env_origins)
            low, high = nodes.min(axis=0), nodes.max(axis=0)
            states[name] = ObjectState(pos=(low + high) / 2, quat_xyzw=np.array([0., 0., 0., 1.]),
                                       low=low, high=high, lin_vel=array(data.nodal_vel_w).mean(axis=0),
                                       ang_vel=np.zeros(3))
        return states

    scene.task_object, scene.object_states = task_object, object_states
    scene._tennis_ball_adapter = True
