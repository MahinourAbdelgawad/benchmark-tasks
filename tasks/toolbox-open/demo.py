"""First-pass scripted demo for the toolbox-open task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

UNVERIFIED. Nothing here has run on the real simulator. The clip positions (geometry.CLIP_TAB), the direction that
releases the latch (RELEASE_DIR), the grasp orientation and the lid-edge grasp are all guesses to tune. The control
helpers (Arm, together) are copied from lock-before-loading/demo.py.

1. Both arms take their clip's tab and pull it along RELEASE_DIR until the measured latch angle shows it released.
2. Both arms take the lid's front edge and swing it up along its arc to about 50 degrees.
3. Both arms let go, drop under the lid and push it over its last degrees onto the hinge stop, then go home.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ
DEBUG = bool(os.environ.get("DEMO_DEBUG"))
PAD_BELOW_TCP = 0.013   # yam_arena.constants.PAD_TIP_PAST_TCP: fingertips below the tool point


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



# -- the task-specific part -----------------------------------------------------------------------------------------
RELEASE_DIR = np.array([-0.6, 0.0, 0.8])   # GUESS: up and toward the robot. Flip a component if the latch doesn't turn
RELEASE_MAX_M = 0.06                        # give up pulling after this far
LIFT_TO_DEG, PUSH_TO_DEG = 50.0, 95.0
LIFT_RATE, PUSH_RATE = 12.0, 15.0           # degrees per second


class Ctx:
    def __init__(self, emb, task):
        self.emb, self.task = emb, task
        self.left, self.right = Arm("left"), Arm("right")

    def lid_deg(self) -> float:
        return float(self.task._STATE["lid_deg"])

    def latch_deg(self) -> float:
        return float(self.task._STATE["latch_deg"])

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.command(), self.right.command()))


def release_clip(arm, ctx, g):
    from yam_arena.demo import grip_rotation

    down = grip_rotation(0.0)
    tab = g.clip_tab(arm.side)
    yield from arm.move(tab + [0, 0, 0.08], down, 2.5, grip=1.0)
    yield from arm.move(tab, down, 1.5)
    yield from arm.set_grip(0.0, 0.8)
    d, direction = 0.0, RELEASE_DIR / np.linalg.norm(RELEASE_DIR)
    target = ctx.task.LATCH_RELEASED_DEG + 5.0
    while ctx.latch_deg() < target and d < RELEASE_MAX_M:   # closed loop on the measured latch angle
        d += 0.02 * DT                                    # 2 cm/s
        arm.aim(tab + direction * d, down)
        yield
    yield from wait(0.5)
    yield from arm.set_grip(1.0, 0.6)


def lift_lid(arm, ctx, g):
    from yam_arena.demo import grip_rotation

    pinch = grip_rotation(math.pi / 2)         # GUESS: fingers closing along x, across the lid's front lip
    edge = g.lid_point(g.LID_GRIP_R, 0.0, arm.side)
    yield from arm.move(edge + [0, 0, 0.08], pinch, 2.0, grip=1.0)
    yield from arm.move(edge + [0, 0, 0.004], pinch, 1.5)
    yield from arm.set_grip(0.0, 0.8)
    theta = 0.0
    while theta < LIFT_TO_DEG:                  # open loop along the lid's arc
        theta += LIFT_RATE * DT
        arm.aim(g.lid_point(g.LID_GRIP_R, theta, arm.side) + [0, 0, 0.004], pinch)
        yield
    yield from arm.set_grip(1.0, 0.6)


def push_over(arm, ctx, g):
    from yam_arena.demo import grip_rotation

    down = grip_rotation(0.0)
    theta = LIFT_TO_DEG
    n = g.ROT @ np.array([-math.sin(math.radians(theta)), 0.0, math.cos(math.radians(theta))])
    start = g.lid_point(g.LID_PUSH_R, theta, arm.side) - 0.03 * n     # under the lid's inner face
    yield from arm.move(start + [-0.05, 0.0, 0.03], down, 2.0, grip=0.0)
    yield from arm.move(start, down, 1.0)
    while theta < PUSH_TO_DEG and ctx.lid_deg() < ctx.task.LID_FULL_DEG + 3.0:
        theta += PUSH_RATE * DT
        n = g.ROT @ np.array([-math.sin(math.radians(theta)), 0.0, math.cos(math.radians(theta))])
        arm.aim(g.lid_point(g.LID_PUSH_R, theta, arm.side) - 0.03 * n, down)
        yield


def go_home(arm):
    yield from arm.move(arm.home[0], arm.home[1], 2.5, grip=1.0)


def script(ctx, g):
    yield from together(release_clip(ctx.left, ctx, g), release_clip(ctx.right, ctx, g))
    yield from together(lift_lid(ctx.left, ctx, g), lift_lid(ctx.right, ctx, g))
    yield from together(push_over(ctx.left, ctx, g), push_over(ctx.right, ctx, g))
    yield from wait(1.5)
    yield from together(go_home(ctx.left), go_home(ctx.right))


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
    sys.modules.setdefault("task", task)
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
            S = task._STATE
            print(f"step {steps} score {info.get('score')} lid {S['lid_deg']:.1f} latch {S['latch_deg']:.1f} "
                  f"moved {S['moved']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "lid_deg": round(S["lid_deg"], 1),
              "latch_deg": round(S["latch_deg"], 1), "problems": list(S["why"]),
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
