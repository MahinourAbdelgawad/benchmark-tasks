"""Station-space targets derived from the toolbox joint frames (metres)."""

import math

import numpy as np
from yam_arena.task_file import TABLE_TOP

ORIGIN = (0.37, 0.0, TABLE_TOP["station"] + 0.001)
YAW_QUAT = (0.0, 0.0, 1.0, 0.0)
SIZE = (0.314, 0.531, 0.279)
HINGE = np.array([-0.15407753, 0.0, 0.1651678])
CLIP_PIVOT = np.array([0.1438092, 0.0, 0.1714074])
CLIP_Y = 0.155
# Midpoint of the measured clasp faces along the jaw-closing axis at the grasp.
# See measure_grasp.py for the USD section and station-space coordinates.
CLIP_TAB = np.array([0.004191, 0.0, -0.042907])


def rotate_open(point, degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = point
    return np.array([c * x - s * z, y, s * x + c * z])


def place(point):
    return np.asarray(ORIGIN) + np.asarray(point) * [-1, -1, 1]


def clip_tab(side, degrees):
    pivot = CLIP_PIVOT + [0, -CLIP_Y if side == "left" else CLIP_Y, 0]
    return place(pivot + rotate_open(CLIP_TAB, degrees))


def lid_point(radius, degrees, side, height=0.0):
    y = -0.085 if side == "left" else 0.085
    return place(HINGE + rotate_open((radius, y, height), degrees))
