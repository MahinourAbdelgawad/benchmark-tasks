"""Scripted bimanual clip release and lid opening through YAM gripper contact."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ


class Arm:
    """One arm commanded in Cartesian space (IK on the URDF), each move starting where the last command left it."""

    def __init__(self, side: str):
        from yam_arena import ik, station
        from yam_arena.constants import READY

        self.side, self.base = side, np.asarray(station.BASES[side], float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.home = (self.pos.copy(), self.rot.copy())
        self.worst_mm = 0.0

    def aim(self, pos, rot) -> None:
        from yam_arena import ik

        q, err, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=self.q, base_pos=self.base, iters=80)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), np.asarray(rot, float)
        self.worst_mm = max(self.worst_mm, err * 1000)

    def move(self, pos, rot=None, seconds: float = 1.0, grip: float | None = None):
        """Generator: a smooth straight-line move to ``pos``/``rot``; ``grip`` 1 open .. 0 closed."""
        from yam_arena import ik

        rot = self.rot if rot is None else rot
        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self.aim(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def set_grip(self, grip: float, seconds: float = 0.8):
        yield from self.move(self.pos, self.rot, seconds, grip)

    def command(self) -> np.ndarray:
        return np.r_[self.q, self.grip]


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def together(*gens):
    """Run generators side by side, one step each per control step."""
    gens = [iter(g) for g in gens]
    while gens:
        for gen in list(gens):
            try:
                next(gen)
            except StopIteration:
                gens.remove(gen)
        if gens:
            yield



def rotation_y(degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Ctx:
    def __init__(self, emb, task):
        self.emb, self.task = emb, task
        self.left, self.right = Arm("left"), Arm("right")
        self.stage = "approach"
        self.grasp_tracking = {}

    def angles(self):
        from yam_arena.scene import object_states

        return self.task.angles(object_states(self.emb._env, ("toolbox",)))

    def record_grasp(self):
        from yam_arena.constants import FINGER_JOINTS, TCP_OFFSET
        from yam_arena.urdf import quat_xyzw_to_matrix

        env = self.emb._env.unwrapped
        origin = env.scene.env_origins[0].cpu().numpy()
        for arm in (self.left, self.right):
            robot = env.scene[f"{arm.side}_robot"]
            body = robot.find_bodies("gripper")[0][0]
            finger_ids = robot.find_joints(FINGER_JOINTS, preserve_order=True)[0]
            pose = robot.data.body_link_pose_w.torch[0, body].cpu().numpy()
            joints = robot.data.joint_pos.torch[0, finger_ids].cpu().numpy()
            actual = pose[:3] - origin + quat_xyzw_to_matrix(pose[3:]) @ TCP_OFFSET
            self.grasp_tracking[arm.side] = {
                "commanded_tcp_m": arm.pos.tolist(),
                "actual_tcp_m": actual.tolist(),
                "tcp_error_mm": ((actual - arm.pos) * 1000).round(2).tolist(),
                "finger_opening_m": float(-joints.sum()),
            }

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.command(), self.right.command()))


CLIP_APPROACH_PITCH = -60.0  # enter on either side of the lower tab, 30 degrees downward
CLIP_STOP = 85.0
CLIP_PAD_CONTACT = 0.009  # leave 4 mm of pad ahead of the side-face contact
CLIP_INSERT_OPENING = 1.0


def clip_rotation(side, pitch):
    from yam_arena.demo import grip_rotation

    return rotation_y(pitch) @ grip_rotation()


def clip_grasp_pose(side, contact, pitch):
    rot = clip_rotation(side, pitch)
    return contact - CLIP_PAD_CONTACT * rot[:, 2], rot


def grasp_clip(arm, g):
    pos, rot = clip_grasp_pose(arm.side, g.clip_tab(arm.side, 0), CLIP_APPROACH_PITCH)
    pregrasp = pos - 0.055 * rot[:, 2]
    yield from arm.move(pregrasp + [0, 0, 0.075], rot, 2.5)
    yield from arm.move(pregrasp, rot, 1.0, grip=CLIP_INSERT_OPENING)
    yield from arm.move(pos, rot, 1.5)
    yield from arm.set_grip(0.68)


def swing_clip(arm, g):
    for angle in np.linspace(0, CLIP_STOP, 100):
        # Rotate with the clip so the jaws stay across its edge during the pull.
        arm.aim(*clip_grasp_pose(arm.side, g.clip_tab(arm.side, angle), CLIP_APPROACH_PITCH + angle))
        yield


def raise_lid(arm, g, clip_angle):
    tab = g.clip_tab(arm.side, clip_angle)
    radius = (np.asarray(g.ORIGIN)[0] - tab[0]) - g.HINGE[0]
    height = tab[2] - g.ORIGIN[2] - g.HINGE[2]
    y = g.ORIGIN[1] - tab[1]
    for angle in np.linspace(0, 32, 80):
        contact = g.place(g.HINGE + g.rotate_open((radius, y, height), angle))
        pitch = CLIP_APPROACH_PITCH + CLIP_STOP - angle
        arm.aim(*clip_grasp_pose(arm.side, contact, pitch))
        yield


def push_lid(arm, holder, g, start):
    from yam_arena.demo import grip_rotation

    # Start in front of the rim, then enter the gap under the raised lid.
    yield from arm.set_grip(1.0, 0.4)
    yield from arm.move(arm.pos + [-0.045, 0, 0], seconds=0.6)
    yield from arm.set_grip(0.0, 0.4)
    radius = 0.247

    def pose(angle, clearance):
        # Horizontal approach stays within the wrist limits along the full arc.
        point = g.lid_point(radius, angle, arm.side, height=-clearance) - [0.013, 0, 0]
        rot = rotation_y(-90) @ grip_rotation()
        return point, rot

    p, r = pose(start, 0.048)
    yield from arm.move(p + [-0.07, 0, 0], r, 1.0)
    yield from arm.move(p, r, 1.0)
    contact, _ = pose(start, 0.003)
    yield from arm.move(contact, r, 0.7)
    yield from holder.set_grip(1.0, 0.4)
    yield from holder.move(holder.pos + [-0.045, 0, 0], seconds=0.7)
    for angle in np.linspace(start, 92, 115):
        arm.aim(*pose(angle, 0.003))
        yield
    yield from arm.move(arm.pos + [-0.06, 0, 0], seconds=1.0)


def script(ctx, g):
    arms = (ctx.left, ctx.right)
    yield from together(*(grasp_clip(arm, g) for arm in arms))
    ctx.record_grasp()
    ctx.stage = "release clips"
    yield from together(*(swing_clip(arm, g) for arm in arms))
    lid, left, right = ctx.angles()
    if min(left, right) < ctx.task.RELEASE_DEG:
        ctx.stage = "clip grasp did not release both clips"
        return
    ctx.stage = "lift from the clips"
    yield from together(*(raise_lid(arm, g, angle) for arm, angle in zip(arms, (left, right))))
    lid, _, _ = ctx.angles()
    if lid < 24:
        ctx.stage = "lid did not lift far enough to push"
        return
    ctx.stage = "push lid"
    yield from push_lid(ctx.right, ctx.left, g, lid)
    ctx.stage = "settle"
    yield from wait(2)


def waypoint_demo():
    """Open-loop version of the verified rollout for Inspect Robots' ``yam_scripted`` policy."""
    import geometry
    import yam_arena.demo as yam_demo
    from yam_arena.constants import FINGER_OPEN

    down = yam_demo.grip_rotation(0.0)

    def pitched_grip_rotation(pitch_degrees=0.0):
        return rotation_y(pitch_degrees) @ down

    # Waypoint only carries one orientation scalar. For this task it represents
    # pitch in degrees instead of the usual downward-grasp yaw in radians.
    yam_demo.grip_rotation = pitched_grip_rotation

    plans = {"left": [], "right": []}
    state = {}

    def add(side, duration, pos, grip, pitch, label):
        plans[side].append(
            yam_demo.Waypoint(duration, tuple(np.asarray(pos, float)), FINGER_OPEN * grip, label, pitch)
        )
        state[side] = (np.asarray(pos, float), grip, pitch)

    def hold(side, duration, label):
        pos, grip, pitch = state[side]
        add(side, duration, pos, grip, pitch, label)

    # Establish the clasp grasps simultaneously.
    for side in plans:
        pos, rot = clip_grasp_pose(side, geometry.clip_tab(side, 0), CLIP_APPROACH_PITCH)
        pregrasp = pos - 0.055 * rot[:, 2]
        add(side, 2.5, pregrasp + [0, 0, 0.075], 1.0, CLIP_APPROACH_PITCH, "above clasp")
        add(side, 1.0, pregrasp, 1.0, CLIP_APPROACH_PITCH, "approach clasp")
        add(side, 1.5, pos, 1.0, CLIP_APPROACH_PITCH, "insert around clasp")
        add(side, 0.8, pos, 0.68, CLIP_APPROACH_PITCH, "pinch clasp")

    for angle in np.linspace(0, CLIP_STOP, 100):
        for side in plans:
            pos, _ = clip_grasp_pose(side, geometry.clip_tab(side, angle), CLIP_APPROACH_PITCH + angle)
            add(side, DT, pos, 0.68, CLIP_APPROACH_PITCH + angle, "release clasp")

    # These are the physical clip angles measured at the end of the successful
    # GPU release. The commanded arc reaches 85 degrees while contact settles
    # near 70 degrees.
    released_angles = {"left": 69.9, "right": 70.9}
    for angle in np.linspace(0, 32, 80):
        for side in plans:
            tab = geometry.clip_tab(side, released_angles[side])
            radius = (np.asarray(geometry.ORIGIN)[0] - tab[0]) - geometry.HINGE[0]
            height = tab[2] - geometry.ORIGIN[2] - geometry.HINGE[2]
            y = geometry.ORIGIN[1] - tab[1]
            contact = geometry.place(geometry.HINGE + geometry.rotate_open((radius, y, height), angle))
            pitch = CLIP_APPROACH_PITCH + CLIP_STOP - angle
            pos, _ = clip_grasp_pose(side, contact, pitch)
            add(side, DT, pos, 0.68, pitch, "lift lid")

    # The right arm enters under the raised lid while the left arm holds it.
    side, holder = "right", "left"
    right_pos, _, right_pitch = state[side]
    add(side, 0.4, right_pos, 1.0, right_pitch, "open right gripper")
    hold(holder, 0.4, "hold lid")
    add(side, 0.6, right_pos + [-0.045, 0, 0], 1.0, right_pitch, "clear clasp")
    hold(holder, 0.6, "hold lid")
    add(side, 0.4, state[side][0], 0.0, right_pitch, "close right gripper")
    hold(holder, 0.4, "hold lid")

    start = 56.0
    radius = 0.247

    def lid_pose(angle, clearance):
        return geometry.lid_point(radius, angle, side, height=-clearance) - [0.013, 0, 0]

    p = lid_pose(start, 0.048)
    add(side, 1.0, p + [-0.07, 0, 0], 0.0, -90.0, "approach lid gap")
    hold(holder, 1.0, "hold lid")
    add(side, 1.0, p, 0.0, -90.0, "enter lid gap")
    hold(holder, 1.0, "hold lid")
    add(side, 0.7, lid_pose(start, 0.003), 0.0, -90.0, "contact lid")
    hold(holder, 0.7, "hold lid")
    hold(side, 0.4, "support lid")
    holder_pos, _, holder_pitch = state[holder]
    add(holder, 0.4, holder_pos, 1.0, holder_pitch, "release left clasp")
    hold(side, 0.7, "support lid")
    add(holder, 0.7, holder_pos + [-0.045, 0, 0], 1.0, holder_pitch, "clear left clasp")

    for angle in np.linspace(start, 92, 115):
        add(side, DT, lid_pose(angle, 0.003), 0.0, -90.0, "push lid open")
        hold(holder, DT, "clear of lid")
    add(side, 1.0, state[side][0] + [-0.06, 0, 0], 0.0, -90.0, "retreat")
    hold(holder, 1.0, "clear of lid")
    return plans


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    task = emb.task.module
    import geometry as g

    obs = emb.reset(Scene(id="demo", instruction=task.INSTRUCTION))
    ctx = Ctx(emb, task)
    gen = script(ctx, g)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 40 == 0:
            lid, left, right = ctx.angles()
            print(f"{steps=} {ctx.stage}: lid {lid:.1f}°, clips {left:.1f}°/{right:.1f}°", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    lid, left, right = ctx.angles()
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "stage": ctx.stage,
              "grasp_tracking": ctx.grasp_tracking,
              "lid_deg": round(lid, 1), "clip_deg": {"left": round(left, 1), "right": round(right, 1)},
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height)
