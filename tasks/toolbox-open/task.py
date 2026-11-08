"""On the station table, 33 cm out from the arm bases: a closed hard-case toolbox (NVIDIA SimReady asset, CC-BY-4.0)
with its two front clips toward the robot and its hinged lid on the far side. The clips are one rigid latch bar in the
asset: releasing either releases both. The lid opens on a hinge to 90 degrees at most.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out

FIRST-PASS BRING-UP: the joint names, thresholds and clip positions below are read off physics.usda or guessed, and
have not been run in Isaac Lab. See README.md, "Debugging".
"""

from __future__ import annotations

import collections
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Open the toolbox: release both latches and swing the lid all the way open."

# -- the asset -------------------------------------------------------------------------------------------------------
USD = "assets/toolbox/simready_usd/sm_storage_toolbox_standard_hardCase_d01_01.usd"
LID_JOINT = "joint_top_01_joint_00"      # box -> lid hinge, limits -90..0 deg (physics.usda)
LATCH_JOINT = "joint_lock_01_joint_00"   # lid -> latch bar, limits -90..+14.1 deg
HANDLE_JOINT = "joint_hand_01_joint_00"  # lid -> carry handle, limits -90..0 deg

# Initial joint positions are the closed pose. The lid is closed at 0. The latch's closed angle is a guess (0): look
# at the first frames of a run, and set it to where the latch rests in its catch.
TOOLBOX = UsdBody("toolbox", usd=USD, pos=tuple(g.ORIGIN), rot_xyzw=g.yaw_xyzw(), size=g.BOX_SIZE,
                  articulation=True, joint_pos=((LID_JOINT, 0.0), (LATCH_JOINT, 0.0), (HANDLE_JOINT, 0.0)),
                  joint_damping=0.05, friction=(0.6, 0.5))
OBJECTS = [TOOLBOX]

# -- thresholds (degrees unless noted) -------------------------------------------------------------------------------
LATCH_RELEASED_DEG = 10.0   # latch bar turned this far from where it rested at the start. TUNE: depends on the catch
LID_CRACKED_DEG = 10.0      # lid open this far
LID_OPEN_DEG = 45.0
LID_FULL_DEG = 85.0         # the hinge stops at 90, so 85 means "all the way"
STILL_DEG = 1.0             # the lid moved less than this over the last STILL_STEPS steps
STILL_STEPS = 10            # 0.5 s at 20 Hz; success must then hold SUCCESS_HOLD_S more
MOVED_M = 0.03              # the toolbox must not slide more than this

_STATE: dict = {}


def _fresh() -> dict:
    return {"q0": None, "pos0": None, "lid_deg": 0.0, "latch_deg": 0.0, "moved": False,
            "hist": collections.deque(maxlen=STILL_STEPS), "why": []}


_STATE.update(_fresh())


def _joint(tb, name: str) -> float:
    try:
        return float(tb.joints[name])
    except KeyError:
        raise RuntimeError(
            f"toolbox has no joint {name!r}. Joints found: {sorted(tb.joints)}. If that list is empty, the USD was "
            "loaded without an articulation root: see README.md, 'Debugging'."
        ) from None


def capture(tb) -> None:
    """Remember where everything rests at the start (called after every reset, once the scene has settled)."""
    _STATE.clear()
    _STATE.update(_fresh())
    _STATE["q0"] = {n: _joint(tb, n) for n in (LID_JOINT, LATCH_JOINT)}
    _STATE["pos0"] = np.asarray(tb.pos, float).copy()


def update(tb) -> None:
    """Fold one control step's toolbox state into the monitor. Angles are measured from the starting rest angles with
    absolute values, so they don't depend on which way the joints are signed."""
    if _STATE.get("q0") is None:
        capture(tb)
    S = _STATE
    S["lid_deg"] = math.degrees(abs(_joint(tb, LID_JOINT) - S["q0"][LID_JOINT]))
    S["latch_deg"] = math.degrees(abs(_joint(tb, LATCH_JOINT) - S["q0"][LATCH_JOINT]))
    S["hist"].append(S["lid_deg"])
    if not S["moved"] and float(np.linalg.norm(np.asarray(tb.pos, float) - S["pos0"])) > MOVED_M:
        S["moved"] = True
        S["why"].append("the toolbox slid more than 3 cm")


@on_reset
def _reset(environment) -> None:
    from yam_arena.scene import object_states

    capture(object_states(environment, ["toolbox"])["toolbox"])


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    update(object_states(environment, ["toolbox"])["toolbox"])


def _still() -> bool:
    h = _STATE["hist"]
    return len(h) == STILL_STEPS and max(h) - min(h) <= STILL_DEG


def success(objects) -> bool:
    """The lid open at least 85 degrees and still, and the toolbox not moved more than 3 cm at any point; holds 1 s."""
    return bool(not _STATE["moved"] and _STATE["lid_deg"] >= LID_FULL_DEG and _still())


def score(objects) -> float:
    """Partial credit (success scores 1.0): 0.75 lid open at least 45 degrees; 0.5 at least 10 degrees; 0.25 latch
    released (turned at least 10 degrees from its rest angle) with the lid still shut. 0 if the toolbox slid more
    than 3 cm at any point."""
    S = _STATE
    if S["moved"]:
        return 0.0
    if S["lid_deg"] >= LID_OPEN_DEG:
        return 0.75
    if S["lid_deg"] >= LID_CRACKED_DEG:
        return 0.5
    return 0.25 if S["latch_deg"] >= LATCH_RELEASED_DEG else 0.0