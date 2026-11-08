"""Checks of task.py's scoring and geometry.py's placement, without the simulator:

    python -m pytest -q test_toolbox_open.py
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import geometry as g
import task as T


def tb(lid_deg=0.0, latch_deg=0.0, pos=(0.0, 0.0, 0.0), sign=-1.0):
    """The toolbox's state: joint angles in radians, signed like the USD (the lid opens toward negative)."""
    return SimpleNamespace(
        joints={T.LID_JOINT: sign * math.radians(lid_deg), T.LATCH_JOINT: sign * math.radians(latch_deg),
                T.HANDLE_JOINT: 0.0},
        pos=np.array(pos, float),
    )


def run(*states):
    """Capture the first state as the start, then feed the rest as control steps."""
    T.capture(states[0])
    for s in states[1:]:
        T.update(s)
    return T.success({}), T.score({})


def hold(**kw):
    return [tb(**kw)] * (T.STILL_STEPS + 1)


def test_start_scores_zero():
    assert run(tb(), *hold()) == (False, 0.0)


@pytest.mark.parametrize(("kw", "score"), [
    (dict(latch_deg=9.0), 0.0), (dict(latch_deg=11.0), 0.25),
    (dict(lid_deg=9.0), 0.0), (dict(lid_deg=11.0), 0.5),
    (dict(lid_deg=44.0), 0.5), (dict(lid_deg=46.0), 0.75),
])
def test_tiers(kw, score):
    assert run(tb(), *hold(**kw))[1] == score


def test_full_open_and_still_is_success():
    ok, s = run(tb(), *hold(lid_deg=88.0))
    assert ok and s == 0.75  # 1.0 comes from success() holding for 1 s, not from score()


def test_just_short_of_full_is_not_success():
    assert not run(tb(), *hold(lid_deg=84.0))[0]


def test_a_lid_still_swinging_is_not_success():
    swing = [tb(lid_deg=80.0 + 2.0 * (i % 2) + 5.0) for i in range(T.STILL_STEPS)]
    assert not run(tb(), *swing)[0]


def test_angles_do_not_depend_on_the_joint_sign():
    assert run(tb(sign=-1.0), *hold(lid_deg=88.0, sign=-1.0))[0]
    assert run(tb(sign=+1.0), *hold(lid_deg=88.0, sign=+1.0))[0]


def test_sliding_the_box_scores_zero_even_after_it_opens():
    ok, s = run(tb(), tb(pos=(0.04, 0, 0)), *hold(lid_deg=88.0, pos=(0.0, 0, 0)))
    assert not ok and s == 0.0


def test_a_slide_under_3_cm_is_fine():
    assert run(tb(), *hold(lid_deg=88.0, pos=(0.02, 0, 0)))[0]


def test_a_missing_joint_explains_itself():
    bad = SimpleNamespace(joints={}, pos=np.zeros(3))
    with pytest.raises(RuntimeError, match="articulation root"):
        T.capture(bad)


def test_the_latches_face_the_robot_and_the_lid_opens_away_from_it():
    hinge = g.place(g.HINGE)
    assert g.place(g.LATCH_PIVOT)[0] < hinge[0]          # latches nearer the arms (smaller x) than the hinge
    closed = g.lid_point(g.LID_GRIP_R, 0.0, "left")
    upright = g.lid_point(g.LID_GRIP_R, 90.0, "left")
    assert closed[0] < hinge[0] and upright[0] == pytest.approx(hinge[0])
    assert upright[2] == pytest.approx(hinge[2] + g.LID_GRIP_R)


def test_the_left_arm_gets_the_clip_on_its_own_side():
    assert g.clip_tab("left")[1] > 0 > g.clip_tab("right")[1]
    assert g.lid_point(0.2, 30.0, "left")[1] > 0 > g.lid_point(0.2, 30.0, "right")[1]
