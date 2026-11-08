"""Geometry of the toolbox-open scene, and where the toolbox stands on the Dream station table.

Pure NumPy, so the task file can import it before Isaac Sim starts.

"Asset frame" is the toolbox USD's own frame (x toward the latch side, y along the hinge axis, z up). Values marked
FROM THE USD come straight from payloads/Physics/physics.usda; values marked MEASURE ME are placeholders to replace
after you look at the toolbox in the viewport (or run inspect_asset.py).

The toolbox is placed turned 180 degrees about the vertical, so its latches face the robot and its hinge is on the far
side. The lid then opens away from the robot, front edge up and over.
"""

from __future__ import annotations

import math

import numpy as np

# -- asset frame (metres) --------------------------------------------------------------------------------------------
HINGE = np.array([-0.15408, 0.0, 0.16517])      # FROM THE USD: lid hinge axis, along y, through this point
LID_COM = np.array([0.15938, 0.0, 0.02895])     # FROM THE USD: lid centre of mass, in the lid frame (origin at the hinge)
LID_LEN = 2 * LID_COM[0]                        # estimate: lid depth, from its centre of mass (assumes a plain slab)
LATCH_PIVOT = HINGE + np.array([0.29789, 0.0, 0.00624])  # FROM THE USD: the latch bar's hinge, on the lid's front edge
BOX_MASS, LID_MASS = 3.0, 0.5                   # FROM THE USD (kg)

# MEASURE ME: the two clips are one rigid bar in the USD; these are where a gripper should take hold of each one.
CLIP_Y = 0.10                                   # half the spacing between the two clips, along y
CLIP_TAB = np.array([0.19, CLIP_Y, 0.13])       # the grippable tab of the clip at +y (the -y one mirrors it)
LID_GRIP_R = 0.30                               # where the arms take the lid's front edge: distance from the hinge
LID_PUSH_R = 0.14                               # where they push it over at the end

# -- placement on the station ----------------------------------------------------------------------------------------
STATION_TABLE = 0.7475                          # yam_arena.task_file.TABLE_TOP["station"]
BOX_XY = (0.37, 0.0)                            # MEASURE ME: where the asset's origin goes; keeps the latches ~22 cm out
YAW = math.pi                                   # latches toward the robot
DROP = 0.001                                    # set it down 1 mm above the table

_C, _S = math.cos(YAW), math.sin(YAW)
ROT = np.array([[_C, -_S, 0.0], [_S, _C, 0.0], [0.0, 0.0, 1.0]])
ORIGIN = np.array([BOX_XY[0], BOX_XY[1], STATION_TABLE + DROP])


def place(p) -> np.ndarray:
    """An asset-frame point on the station (world frame)."""
    return ROT @ np.asarray(p, float) + ORIGIN


def yaw_xyzw() -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(YAW / 2), math.cos(YAW / 2))


def clip_tab(side: str) -> np.ndarray:
    """Station position of a clip's tab. ``side`` is the arm that takes it: the left arm is at +y on the station."""
    p = CLIP_TAB.copy()
    # asset +y maps to station -y after the half turn, so the left arm (station +y) takes the asset -y clip
    p[1] = -CLIP_Y if side == "left" else CLIP_Y
    return place(p)


def lid_point(radius: float, open_deg: float, side: str) -> np.ndarray:
    """Station position of the point on the lid's surface ``radius`` from the hinge, with the lid open ``open_deg``,
    in line with the left or right clip."""
    t = math.radians(open_deg)
    y = -CLIP_Y if side == "left" else CLIP_Y
    return place(HINGE + np.array([radius * math.cos(t), y, radius * math.sin(t)]))
