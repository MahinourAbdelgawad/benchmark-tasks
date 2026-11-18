"""Task-local adapter for loading an articulated USD through yam_arena."""

from dataclasses import dataclass
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec
from pathlib import Path
import sys
from types import SimpleNamespace

from yam_arena.task_file import Box

USD = Path(__file__).resolve().parent / "assets/toolbox_open.usdc"


@dataclass(frozen=True)
class Toolbox(Box):
    joints: tuple = ()
    rotation: tuple = (0.0, 0.0, 0.0, 1.0)

    def _check(self):
        super()._check()
        if not USD.is_file():
            raise FileNotFoundError(f"{USD} (run prepare_asset.py)")


def arena_object(spec):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab.sim import CollisionPropertiesCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.terms.events import reset_articulation_joints
    from isaaclab_arena.utils.pose import Pose

    class ToolboxObject(Object):
        def _build_reset_event(self):
            return EventTermCfg(func=reset_articulation_joints, mode="reset",
                                params={"asset_cfg": SceneEntityCfg(self.name)})

        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            lid, handle, *clips = spec.joints
            cfg.init_state.joint_pos = dict.fromkeys(spec.joints, 0.0)
            cfg.init_state.joint_vel = dict.fromkeys(spec.joints, 0.0)
            cfg.actuators = {
                "lid": ImplicitActuatorCfg(joint_names_expr=[lid], stiffness=0.0, damping=0.08),
                "clips": ImplicitActuatorCfg(joint_names_expr=clips, stiffness=0.0, damping=0.006,
                                             friction=0.012, dynamic_friction=0.012),
                "handle": ImplicitActuatorCfg(joint_names_expr=[handle], stiffness=0.0, damping=0.03),
            }
            return cfg

    return ToolboxObject(
        name=spec.name, usd_path=str(USD), object_type=ObjectType.ARTICULATION,
        initial_pose=Pose(position_xyz=spec.pos, rotation_xyzw=spec.rotation),
        spawn_cfg_addon={
            "fix_root_link": True,
            "articulation_props": {
                "/Geometry/box_01_obj_00": [PhysxArticulationCfg(
                    enabled_self_collisions=True, solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4)]
            },
            "collision_props": CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        },
    )


def patch_scene(scene):
    task_object, object_states = scene.task_object, scene.object_states
    names = set()

    def load(spec):
        if isinstance(spec, Toolbox):
            names.add(spec.name)
            return arena_object(spec)
        names.discard(spec.name)
        return task_object(spec)

    def states(env, requested):
        result = object_states(env, [name for name in requested if name not in names])
        for name in names.intersection(requested):
            asset = env.unwrapped.scene[name]

            def joint_values(buffer):
                values = getattr(buffer, "torch", buffer)[0].detach().cpu().tolist()
                return dict(zip(asset.joint_names, values, strict=True))

            result[name] = SimpleNamespace(joints=joint_values(asset.data.joint_pos),
                                           joint_velocity=joint_values(asset.data.joint_vel))
        return result

    scene.task_object, scene.object_states = load, states


class SceneLoader(Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        patch_scene(module)


class SceneFinder(MetaPathFinder):
    """Arena imports its simulator-dependent scene module after loading the task."""

    def find_spec(self, fullname, path, target=None):
        if fullname != "yam_arena.scene":
            return None
        sys.meta_path.remove(self)
        spec = find_spec(fullname)
        if spec is not None:
            spec.loader = SceneLoader(spec.loader)
        return spec


if "yam_arena.scene" in sys.modules:
    patch_scene(sys.modules["yam_arena.scene"])
else:
    sys.meta_path.insert(0, SceneFinder())
