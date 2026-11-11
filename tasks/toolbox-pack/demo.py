"""Runner for the toolbox scene.

    python demo.py task.py demo_out [--max-steps 300] [--no-record]

Writes demo_out/setup.png (first frame), demo_out/demo_<camera>.mp4 and demo_out/result.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF) from wherever its last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena.constants import FINGER_OPEN, READY

        self.name = name
        self.base = np.asarray(base_pos, dtype=float)
        self.q = np.array(READY, dtype=float)
        self.finger = FINGER_OPEN
        self.cmd = np.r_[self.q, 1.0]
        self._yaw = 0.0

    def tcp(self) -> np.ndarray:
        from yam_arena import ik

        return ik.tcp_pose(self.q, self.base)[0]

    def move(self, pos, *, yaw: float = 0.0, finger: float | None = None, duration: float = 1.0, label: str = ""):
        """Generator: TCP to ``pos`` (world m), straight-down grip turned ``yaw``; ``finger`` in m (None keeps it)."""
        from yam_arena.demo import ArmPlan, Waypoint

        f = self.finger if finger is None else finger
        plan = ArmPlan([Waypoint(duration, tuple(map(float, pos)), f, label, yaw)], self.base, q_start=self.q,
                       finger_start=self.finger)
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self.cmd, _ = plan.action7(k / CONTROL_HZ)
            self.q = plan.q.copy()
            yield
        self.finger = f
        self._yaw = yaw

    def grip(self, finger: float, duration: float = 0.6):
        yield from self.move(self.tcp(), yaw=self._yaw, finger=finger, duration=duration, label="grip")

    def wait(self, duration: float):
        for _ in range(max(1, round(duration * CONTROL_HZ))):
            yield


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.t = 0.0

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def run(task_path: str, out_dir: str, max_steps: int = 300, record: bool = True, cam_width: int = 640,
        cam_height: int = 352, hold_s: float = 2.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    ctx = Ctx(emb)
    script = emb.task.module.demo_script(ctx)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, info = 0, None, {}
    while steps < max_steps:
        try:
            next(script)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        ctx.t += 1.0 / CONTROL_HZ
        info = res.info
        snap(res.observation)
        if steps % 40 == 0:
            print(f"step {steps} score {info.get('score')}", flush=True)
        if res.terminated:
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break

    result = {"score": float(info.get("score", 0.0) or 0.0), "steps": steps}
    if record:
        import imageio

        for cam, fs in frames.items():
            imageio.mimsave(out / f"demo_{cam}.mp4", [np.asarray(f)[..., :3] for f in fs], fps=CONTROL_HZ,
                            codec="libx264", output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", np.asarray(frames[top][0])[..., :3])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height)