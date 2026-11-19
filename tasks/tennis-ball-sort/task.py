"""Sort visually identical tennis balls by firmness."""
import os
import sys
from pathlib import Path
from random import Random
import numpy as np
from yam_arena.task_file import Box, TABLE_TOP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tennis_physics import TennisBall, install

install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
INSTRUCTION = "Compare the tennis balls by squeezing them. Leave the softer ball in the orange tray and the firmer ball in the blue tray."
OBJECTS = []
CENTERS = {"soft": (.36, -.19), "firm": (.36, .19)}
classes = ["soft", "firm"]
Random(int(os.environ.get("TENNIS_SEED", "0"))).shuffle(classes)
targets = {}
for index, kind in enumerate(classes):
    name = f"sample_{index}"
    targets[name] = CENTERS[kind]
    OBJECTS.append(TennisBall(name, pos=(.36, -.055 + index * .11, TOP + .034),
                              size=(.067,) * 3, color=(.72, .88, .04),
                              modulus={"soft": 65_000., "firm": 180_000.}[kind]))

for kind, (x, y) in CENTERS.items():
    color = (.95, .33, .06) if kind == "soft" else (.08, .34, .88)
    pieces = [("floor", (x, y, TOP + .003), (.14, .14, .006))]
    for sign in (-1, 1):
        pieces.extend([
            (f"x_{sign}", (x + sign * .067, y, TOP + .035), (.006, .14, .058)),
            (f"y_{sign}", (x, y + sign * .067, TOP + .035), (.128, .006, .058)),
        ])
    for label, pos, size in pieces:
        OBJECTS.append(Box(f"{kind}_{label}".replace("-", "m"), pos=pos, size=size, color=color))


def score(objects):
    total = 0.
    for name, (x, y) in targets.items():
        ball = objects[name]
        if (np.all(ball.low[:2] >= np.array([x, y]) - .064)
                and np.all(ball.high[:2] <= np.array([x, y]) + .064)
                and TOP + .003 <= ball.low[2] <= TOP + .009
                and np.linalg.norm(ball.lin_vel) < .02):
            total += .5
    return total


def success(objects):
    return score(objects) == 1.
