"""Hazard-aware kit packing: put supplies and tools into the two compartments of the box, obeying handling rules.

Run:  python demo.py task.py demo_out --max-steps 300
"""

from __future__ import annotations

import math
import pathlib
import sys, os

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import propkit  # noqa: E402,F401  (hooks yam_arena so Prop objects are simulated)
from propkit import prop, rot_matrix  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"


def _find(name: str) -> str:
    hits = sorted((HERE / "assets").rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {HERE / 'assets'} (did you run prep_assets.py?)")
    return str(hits[0])


# -- the box ---------------------------------------------------------------------------------------------------------
BOX_SCALE, BOX_YAW = 0.85, math.radians(20)
BOX = prop("box", _find("D00443.usd"), ((-0.2715, -0.1949, 0.003), (0.2703, 0.1953, 0.2242)), (0.40, 0.22),
           table_top=TOP, yaw=BOX_YAW, scale=BOX_SCALE, fixed=True)
BOX_FLOOR_Z, BOX_RIM_Z = BOX.pos[2] - BOX.size[2] / 2, BOX.pos[2] + BOX.size[2] / 2

HALF = (0.2709, 0.1951)
DIV_AXIS, DIV_AT = 0, 0.0
DIV_WALL, WALL = 0.025, 0.02
SUPPLIES_SIGN = +1          # +1: supplies go on the +axis side of the divider. Flip after looking at setup.png
SUPPLIES_SIDE, TOOLS_SIDE = "left", "right"   # words for the instruction: set them from what setup.png shows

# -- the items: name -> (file, asset bbox, origin xy, yaw deg, scale, mass kg) ---------------------------------------
ITEMS = {
    "lubricant": ("lubricant.usda", ((-0.0413, -0.0413, 0.0004), (0.0413, 0.0413, 0.2466)), (0.420, -0.10), 0, 0.6, 0.30),
    "spray": ("spray.usda", ((-0.031, -0.031, 0.0), (0.031, 0.031, 0.1864)), (0.500, -0.10), 0, 0.8, 0.25),
    "sealant": ("sealant.usda", ((-0.0794, -0.0179, 0.0001), (0.0351, 0.0182, 0.0233)), (0.422, -0.22), 0, 1.0, 0.08),
    "tape": ("tape.usda", ((-0.0681, -0.0681, 0.0002), (0.0697, 0.0681, 0.0459)), (0.540, -0.22), 0, 0.6, 0.04),
    "glasses": ("glasses.usda", ((-0.0825, -0.0845, 0.0001), (0.0845, 0.0845, 0.0596)), (0.460, -0.36), 0, 0.6, 0.03),
    "hammer": ("hammer.usda", ((-0.0651, -0.1651, 0.0), (0.0651, 0.1651, 0.0292)), (0.230, -0.40), 90, 0.6, 0.25),
    "pliers": ("pliers.usda", ((-0.1108, -0.0345, 0.0), (0.0811, 0.0342, 0.0259)), (0.242, -0.29), 0, 0.8, 0.12),
    "wrench": ("wrench.usda", ((-0.0922, -0.0302, 0.0003), (0.1174, 0.0334, 0.0142)), (0.219, -0.19), 0, 0.9, 0.12),
    "screwdriver": ("screwdriver.usda", ((-0.1029, -0.0166, -0.0035), (0.1027, 0.0166, 0.0319)), (0.230, -0.09), 0, 0.8, 0.08),
}
SUPPLIES = ("lubricant", "sealant", "spray", "glasses", "tape")
TOOLS = ("hammer", "pliers", "wrench", "screwdriver")
PACK = SUPPLIES + TOOLS

SPEC = {n: prop(n, _find(f), b, xy, table_top=TOP, yaw=math.radians(yaw), scale=s, mass=m)
        for n, (f, b, xy, yaw, s, m) in ITEMS.items()}
OBJECTS = [BOX, *SPEC.values()]


def _check_layout(margin: float = 0.01) -> None:
    """Fail at import (not inside the simulator) if two items start overlapping."""
    rects = {}
    for n, o in SPEC.items():
        c, s = abs(math.cos(o.yaw)), abs(math.sin(o.yaw))
        hx, hy = (c * o.size[0] + s * o.size[1]) / 2, (s * o.size[0] + c * o.size[1]) / 2
        rects[n] = (o.pos[0] - hx, o.pos[0] + hx, o.pos[1] - hy, o.pos[1] + hy)
    names = list(rects)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = rects[a], rects[b]
            if ra[0] < rb[1] + margin and rb[0] < ra[1] + margin and ra[2] < rb[3] + margin and rb[2] < ra[3] + margin:
                raise ValueError(f"layout: {a} and {b} overlap or are closer than {margin * 100:.0f} cm")


_check_layout()

INSTRUCTION = (
    f"Pack the kit into the box. Put the supplies (grease bottle, window sealant, spray can, safety glasses, duct tape) "
    f"in the {SUPPLIES_SIDE} compartment and the hand tools (hammer, pliers, wrench, screwdriver) in the "
    f"{TOOLS_SIDE} compartment. Keep the grease bottle and the spray can upright, do not drop the safety glasses, "
    f"do not put anything on top of the safety glasses, and do not rest tools on the bottle or the can."
)

# -- rules -------------------------------------------------------------------------------------------------------------
UPRIGHT = {"lubricant": 25.0, "spray": 25.0}   # max tilt (degrees) from vertical
NO_ABOVE = {                                    # lower item -> items that must never rest on it
    "glasses": tuple(n for n in PACK if n != "glasses"),
    "spray": TOOLS,
    "lubricant": TOOLS,
}
DROP_SPEED = 1.0          # m/s: the glasses moving faster than this at any time counts as dropped / thrown
FALL_Z = TOP - 0.05       # an item centre below this left the table
SETTLED_SPEED = 0.03      # m/s: success needs everything this still


def compartment(p) -> str | None:
    """'supplies' / 'tools' if world point ``p`` is inside the box on that side of the divider, else None."""
    d = np.asarray(p[:2], float) - np.asarray(BOX.pos[:2], float)
    c, s = math.cos(-BOX_YAW), math.sin(-BOX_YAW)
    local = ((c * d[0] - s * d[1]) / BOX_SCALE, (s * d[0] + c * d[1]) / BOX_SCALE)
    if abs(local[0]) > HALF[0] - WALL or abs(local[1]) > HALF[1] - WALL:
        return None
    if not (BOX_FLOOR_Z < p[2] < BOX_RIM_Z + 0.08):
        return None
    off = local[DIV_AXIS] - DIV_AT
    if abs(off) < DIV_WALL:
        return None
    return "supplies" if (off > 0) == (SUPPLIES_SIGN > 0) else "tools"


def tilt_deg(st) -> float:
    return math.degrees(math.acos(float(np.clip(rot_matrix(st.quat_xyzw)[2, 2], -1.0, 1.0))))


# -- monitor: latched events that a final-pose check would miss ----------------------------------------------------------
_MON: dict = {"dropped": False, "fell": set(), "err": None}


def _reset_monitor() -> None:
    _MON.update(dropped=False, fell=set())


def _watch(emb) -> None:
    from yam_arena.scene import object_states

    try:
        st = object_states(emb._env, emb.task.names)
    except Exception as e:  # pragma: no cover
        if _MON["err"] is None:
            _MON["err"] = repr(e)
            print(f"[pack] monitor error: {e!r}", flush=True)
        return
    if float(np.linalg.norm(st["glasses"].lin_vel)) > DROP_SPEED and not _MON["dropped"]:
        _MON["dropped"] = True
        print("[pack] safety glasses moved too fast (dropped?)", flush=True)
    for n in PACK:
        if st[n].pos[2] < FALL_Z and n not in _MON["fell"]:
            _MON["fell"].add(n)
            print(f"[pack] {n} fell off the table", flush=True)


def _install_hooks() -> None:
    try:
        from yam_arena import inspect_embodiment as ie
    except Exception:  # pragma: no cover
        return
    cls = ie.YamArenaEmbodiment
    if getattr(cls.step, "_pack_hooked", False):
        return
    base_reset, base_step = cls.reset, cls.step

    def reset(self, *a, **k):
        out = base_reset(self, *a, **k)
        _reset_monitor()
        return out

    def step(self, *a, **k):
        out = base_step(self, *a, **k)
        _watch(self)
        return out

    reset._pack_hooked = step._pack_hooked = True
    cls.reset, cls.step = reset, step


_install_hooks()


# -- scoring -----------------------------------------------------------------------------------------------------------
def violations(objects) -> list[str]:
    v = []
    if _MON["dropped"]:
        v.append("safety glasses dropped / handled too fast")
    v += [f"{n} fell off the table" for n in sorted(_MON["fell"])]
    for lower, uppers in NO_ABOVE.items():
        f = objects[lower]
        for u in uppers:
            o = objects[u]
            if (o.low[:2] < f.high[:2]).all() and (o.high[:2] > f.low[:2]).all() and o.low[2] > f.high[2] - 0.005:
                v.append(f"{u} on top of {lower}")
    return v


def _item_ok(name: str, st) -> bool:
    want = "supplies" if name in SUPPLIES else "tools"
    if compartment(st.pos) != want:
        return False
    return name not in UPRIGHT or tilt_deg(st) <= UPRIGHT[name]


def score(objects) -> float:
    """Fraction of items in the right compartment (and upright where required), minus 0.2 per rule violation."""
    frac = sum(_item_ok(n, objects[n]) for n in PACK) / len(PACK)
    return round(max(0.0, frac - 0.2 * len(violations(objects))), 3)


def success(objects) -> bool:
    still = all(float(np.linalg.norm(objects[n].lin_vel)) < SETTLED_SPEED for n in PACK)
    return bool(all(_item_ok(n, objects[n]) for n in PACK) and not violations(objects) and still)


def distance(objects) -> float:
    """Number of items not yet correctly packed."""
    return float(sum(not _item_ok(n, objects[n]) for n in PACK))


# -- scripted demo -----------------------------------------------------------------------------------------------------
ORDER = ("lubricant", "spray", "sealant", "tape", "glasses", "hammer", "wrench", "pliers", "screwdriver")
# Glasses go in last among the supplies, so nothing is ever carried over them.

# name -> (compartment, across fraction, along fraction, place yaw vs box (deg), grasp yaw vs item (deg))
# "across" runs along the divider axis, "along" the other box axis; fractions are of the compartment's width/length,
# measured from its centre. Place yaw 90 = item's asset-x long axis runs along the compartment. Grasp yaw is the
# direction of the item's long axis, so the fingers close across its short side (hammer's long axis is asset y: +90).
SLOT = {
    "glasses": ("supplies", -0.21, -0.32, 0, 0),
    "sealant": ("supplies", 0.34, -0.29, 90, 0),
    "tape": ("supplies", -0.23, 0.03, 0, 0),
    "lubricant": ("supplies", 0.31, 0.07, 0, 0),
    "spray": ("supplies", 0.00, 0.35, 0, 0),
    "hammer": ("tools", -0.29, -0.17, 0, 90),
    "pliers": ("tools", 0.13, -0.17, 90, 0),
    "screwdriver": ("tools", 0.39, -0.17, 90, 0),
    "wrench": ("tools", 0.00, 0.35, 0, 0),
}

GRIP_OVER = 0.012      # TCP this far above the item's centre when grasping (pad tips just below the centre)
INNER_FLOOR_Z = BOX_FLOOR_Z + 0.015   # GUESS: top of the box's inner floor. Raise/lower after watching a release
DROP_GAP = 0.01        # release the item this far above the floor


def _wrap(a: float) -> float:
    return (a + math.pi / 2) % math.pi - math.pi / 2


def _slot_xy(name: str) -> tuple[float, float]:
    which, fa, fb = SLOT[name][:3]
    side = 1.0 if (which == "supplies") == (SUPPLIES_SIGN > 0) else -1.0
    ax, ot = DIV_AXIS, 1 - DIV_AXIS
    W = (HALF[ax] - WALL - DIV_WALL) * BOX_SCALE
    L = 2 * (HALF[ot] - WALL) * BOX_SCALE
    ca = (DIV_AT + side * (DIV_WALL + HALF[ax] - WALL) / 2) * BOX_SCALE
    pa, pb = ca + fa * W, fb * L
    bx, by = (pa, pb) if DIV_AXIS == 0 else (pb, pa)
    c, s = math.cos(BOX_YAW), math.sin(BOX_YAW)
    return BOX.pos[0] + c * bx - s * by, BOX.pos[1] + s * bx + c * by


def _pick_arm(ctx, pick_xy, place_xy):
    """The arm whose base is closest to the farther of the pick and place points."""
    def worst(arm):
        return max(np.hypot(*(arm.base[:2] - np.array(p))) for p in (pick_xy, place_xy))
    return min((ctx.left, ctx.right), key=worst)
    # return ctx.right


def _pick_place(ctx, name):
    from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN

    st = ctx.states()[name]
    px, py, pz = (float(v) for v in st.pos)
    R = rot_matrix(st.quat_xyzw)
    th = math.atan2(R[1, 0], R[0, 0])
    _, _, _, poff, goff = SLOT[name]
    tx, ty = _slot_xy(name)
    arm = _pick_arm(ctx, (px, py), (tx, ty))
    hz = SPEC[name].size[2] / 2
    z_pick = max(pz + GRIP_OVER, TOP + 0.015)
    dz = z_pick - pz                                   # TCP height above the item's centre while held
    z_place = INNER_FLOOR_Z + DROP_GAP + hz + dz
    hi = max(TOP + 0.12, BOX_RIM_Z + 0.03) + dz + hz   # carry height: the item's underside clears the box rim
    g1 = _wrap(th + math.radians(goff))
    g2 = _wrap(BOX_YAW + math.radians(poff + goff))
    print(f"[demo] {name}: {arm.name} arm, pick=({px:.3f},{py:.3f}) place=({tx:.3f},{ty:.3f}) hi={hi - TOP:.2f}m", flush=True)

    yield from arm.move((px, py, hi), yaw=g1, finger=FINGER_OPEN, duration=2.0)
    yield from arm.move((px, py, z_pick), yaw=g1, duration=1.2)
    yield from arm.grip(FINGER_CLOSED, 0.8)
    yield from arm.move((px, py, hi), yaw=g1, duration=1.2)
    yield from arm.move((tx, ty, hi), yaw=g2, duration=2.0)
    yield from arm.move((tx, ty, z_place), yaw=g2, duration=1.2)
    yield from arm.grip(FINGER_OPEN, 0.6)
    yield from arm.move((tx, ty, hi), yaw=g2, duration=1.0)
    yield from arm.wait(1.0)
    st = ctx.states()
    print(f"[demo] {name} -> {compartment(st[name].pos)}, tilt {tilt_deg(st[name]):.0f} deg, score {score(st):.3f}",
          flush=True)


def demo_script(ctx):
    """Pack every item one at a time. PACK_ONLY=spray,hammer runs just those (for debugging one item)."""
    from yam_arena.constants import FINGER_OPEN

    only = [n for n in os.environ.get("PACK_ONLY", "").split(",") if n]
    print(f"[demo] gripper opens to {FINGER_OPEN} m; box rim {BOX_RIM_Z - TOP:.3f} m above the table", flush=True)
    for n in (n for n in ORDER if not only or n in only):
        yield from _pick_place(ctx, n)
    yield from ctx.right.wait(2.0)
    st = ctx.states()
    print(f"[demo] done: score {score(st):.3f} success {success(st)} violations {violations(st)}", flush=True)