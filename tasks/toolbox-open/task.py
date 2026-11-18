"""Open both toolbox clips and push the lid to its hinge stop on the YAM station."""

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry
from scenekit import Toolbox

SCENE = "station"
INSTRUCTION = "Release both yellow toolbox clips, then lift and push the lid all the way open."
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
LID = "joint_top_01_joint_00"
HANDLE = "joint_hand_01_joint_00"
CLIPS = ("left_clip_joint", "right_clip_joint")
RELEASE_DEG = 25.0
FULL_DEG = 85.0

OBJECTS = [Toolbox("toolbox", pos=geometry.ORIGIN, size=geometry.SIZE, joints=(LID, HANDLE, *CLIPS),
                   rotation=geometry.YAW_QUAT)]


def demo():
    """Scripted full-score rollout used by Inspect Robots' ``yam_scripted`` policy."""
    from demo import waypoint_demo

    return waypoint_demo()


def angles(states):
    joints = states["toolbox"].joints
    return tuple(max(0.0, -math.degrees(joints[name])) for name in (LID, *CLIPS))


def success(states):
    lid, left, right = angles(states)
    return (lid >= FULL_DEG and min(left, right) >= RELEASE_DEG
            and abs(states["toolbox"].joint_velocity[LID]) <= math.radians(2))


def score(states):
    lid, left, right = angles(states)
    released = sum(angle >= RELEASE_DEG for angle in (left, right))
    if released < 2:
        return 0.25 * released
    if lid >= 45:
        return 0.75
    return 0.5
