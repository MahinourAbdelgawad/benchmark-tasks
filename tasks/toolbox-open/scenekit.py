"""scenekit: USD bodies, articulations and posed fixed shapes for a yam_arena task file.

yam_arena's task files spawn single primitives (yaw only) and library assets. This module adds three object kinds,
for the hinged platform, the lock pin and the mannequin hand:

- :class:`UsdBody`: a body loaded from a USD file next to the task file. With ``articulation=True`` it is an
  articulation (here, the stand and the platform on its hinge, fixed base) whose joints start at ``joint_pos``.
- :class:`Shape`: a fixed box or capsule in any orientation.

They subclass ``yam_arena.task_file.Box`` only so the task-file checks accept them. Importing this module hooks
``yam_arena.scene`` and the embodiment's step and reset, without changing yam_arena:

- ``success()`` and ``score()`` receive a :class:`BodyState` for these objects: the body origin's pose (not a
  bounding-box centre) and, for articulations, the joint positions;
- callbacks registered with :func:`on_step` run after every control step, before the task file is scored, and
  ones registered with :func:`on_reset` run after every reset; both get the Isaac Lab environment.

Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["BodyState", "Shape", "UsdBody", "arm_joints", "env", "on_reset", "on_step", "robot_links"]

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class UsdBody(Box):
    """A body from ``usd`` (relative to this folder). ``pos``/``rot_xyzw`` place the file's origin. ``size`` is only
    nominal. With ``mass=None`` a rigid body is kinematic; an articulation keeps the fixed base its file declares."""

    usd: str = ""
    rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    articulation: bool = False
    joint_pos: tuple[tuple[str, float], ...] = ()
    joint_damping: float = 0.03
    friction: tuple[float, float] = (0.8, 0.7)  # static, dynamic
    friction_min: bool = False  # combine friction with "min", so this surface's friction rules its contacts
    solver_iters: int = 16
    torsional_patch: float | None = None  # m, for bodies held in a pinch (lets the pads resist twisting)

    def _check(self) -> None:
        if not (HERE / self.usd).is_file():
            raise FileNotFoundError(f"{self.name}: {HERE / self.usd} not found")


@dataclass(frozen=True)
class Shape(Box):
    """A fixed box (``size``) or capsule (``radius``, ``height`` of the cylindrical part, along ``axis``), posed
    with ``rot_xyzw``. It collides unless ``collide`` is False."""

    kind: str = "box"
    radius: float = 0.01
    height: float = 0.02
    axis: str = "Z"
    rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    collide: bool = True

    def _check(self) -> None:
        if self.kind not in ("box", "capsule"):
            raise ValueError(f"{self.name}: kind must be 'box' or 'capsule'")
        if self.mass is not None:
            raise ValueError(f"{self.name}: a Shape is fixed (no mass)")


@dataclass(frozen=True)
class BodyState(ObjectState):
    """The body origin's pose, velocities, bounds and joint positions (rad or m)."""

    joints: dict = field(default_factory=dict)

    def rotation(self) -> np.ndarray:
        x, y, z, w = self.quat_xyzw
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# -- Arena objects -------------------------------------------------------------------------------------------------

_SPECS: dict[str, UsdBody | Shape] = {}


def _usd_object(spec: UsdBody):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    material = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0], dynamic_friction=spec.friction[1],
                                              friction_combine_mode="min" if spec.friction_min else "average")
    addon = {"physics_material": material}
    if spec.articulation:
        addon["articulation_props"] = sim_utils.ArticulationRootPropertiesCfg(
            solver_position_iteration_count=spec.solver_iters)
    else:
        addon["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=spec.solver_iters,
                                                                kinematic_enabled=spec.mass is None)
        if spec.torsional_patch:
            addon["collision_props"] = sim_utils.CollisionPropertiesCfg(
                torsional_patch_radius=spec.torsional_patch, min_torsional_patch_radius=0.8 * spec.torsional_patch)
        if spec.mass is not None:
            addon["mass_props"] = sim_utils.MassPropertiesCfg(mass=spec.mass)

    class UsdObject(Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            joints = dict(spec.joint_pos)
            cfg.init_state.joint_pos = joints
            cfg.actuators = {"joints": ImplicitActuatorCfg(joint_names_expr=list(joints) or [".*"], stiffness=0.0,
                                                           damping=spec.joint_damping, friction=0.0)}
            return cfg

    obj = UsdObject(name=spec.name, usd_path=str(HERE / spec.usd),
                    object_type=ObjectType.ARTICULATION if spec.articulation else ObjectType.RIGID,
                    spawn_cfg_addon=addon)
    obj.set_initial_pose(Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=tuple(spec.rot_xyzw)))
    return obj


def _shape_object(spec: Shape):
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from yam_arena.station import linear_rgb

    body = {"visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.6)}
    if spec.collide:
        body["collision_props"] = sim_utils.CollisionPropertiesCfg()
    if spec.kind == "box":
        cfg = sim_utils.CuboidCfg(size=tuple(spec.size), **body)
    else:
        cfg = sim_utils.CapsuleCfg(radius=spec.radius, height=spec.height, axis=spec.axis, **body)
    obj = Object(name=spec.name, object_type=ObjectType.BASE, spawner_cfg=cfg)
    obj.set_initial_pose(Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=tuple(spec.rot_xyzw)))
    return obj


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _state(environment, spec: UsdBody | Shape) -> BodyState:
    if isinstance(spec, Shape):
        half = np.asarray(spec.size) / 2
        pos = np.asarray(spec.pos, float)
        return BodyState(pos=pos, quat_xyzw=np.asarray(spec.rot_xyzw, float), lin_vel=np.zeros(3),
                         ang_vel=np.zeros(3), low=pos - half, high=pos + half)
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0]
    joints = {}
    if spec.articulation:
        q = _np(asset.data.joint_pos)[0]
        joints = {n: float(q[i]) for i, n in enumerate(asset.joint_names)}
    pos = pose[:3] - origin
    return BodyState(pos=pos, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6], low=pos, high=pos,
                     joints=joints)


def robot_links(environment) -> dict[str, dict[str, np.ndarray]]:
    """Each arm's link origins by link name (station frame, env 0), for proximity checks."""
    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    out = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        pos = _np(robot.data.body_pos_w)[0] - origin
        out[side] = {name: pos[i] for i, name in enumerate(robot.body_names)}
    return out


def arm_joints(environment, side: str, names) -> np.ndarray:
    """One arm's measured joint positions, in the order of ``names``."""
    robot = environment.unwrapped.scene[f"{side}_robot"]
    ids = robot.find_joints(list(names), preserve_order=True)[0]
    return _np(robot.data.joint_pos)[0, ids]


# -- hooking yam_arena -----------------------------------------------------------------------------------------------

_STEP: list = []
_RESET: list = []
_ENV: dict = {"env": None}


def on_step(fn):
    """Run ``fn(env)`` after every control step, before the task file is scored."""
    _STEP.append(fn)
    return fn


def on_reset(fn):
    """Run ``fn(env)`` after every reset, once the scene has settled."""
    _RESET.append(fn)
    return fn


def env():
    return _ENV["env"]


def _patch_scene(scene) -> None:
    if getattr(scene, "_scenekit", False):
        return
    task_object, object_states = scene.task_object, scene.object_states

    def task_object_hook(spec):
        if isinstance(spec, UsdBody | Shape):
            _SPECS[spec.name] = spec
            return _usd_object(spec) if isinstance(spec, UsdBody) else _shape_object(spec)
        return task_object(spec)

    def object_states_hook(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = object_states(environment, plain) if plain else {}
        out.update({n: _state(environment, _SPECS[n]) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states = task_object_hook, object_states_hook
    scene._scenekit = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_scenekit", False):
        return
    task_step, reset = cls._task_step, cls.reset

    def task_step_hook(self, *a, **k):
        _ENV["env"] = self._env
        for fn in _STEP:
            fn(self._env)
        return task_step(self, *a, **k)

    def reset_hook(self, *a, **k):
        obs = reset(self, *a, **k)
        _ENV["env"] = self._env
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = task_step_hook, reset_hook
    cls._scenekit = True


class _Finder(importlib.abc.MetaPathFinder):
    """Patches yam_arena's scene and embodiment modules as they are imported (they need Isaac Sim, which starts after
    the task file is loaded)."""

    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader, patch = spec.loader, self.targets[fullname]
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Hook yam_arena now for modules already imported, and on import for the rest. Idempotent."""
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
