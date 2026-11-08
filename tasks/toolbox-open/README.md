# Unlatch a toolbox and open the lid all the way (bring-up draft)

A closed hard-case toolbox stands on the table with its two front clips toward the robot. The robot releases the
clips and swings the lid open to the hinge stop (90 degrees). Scoring is by the lid's angle.

**Status: nothing here has run in Isaac Lab yet.** The joint names and limits come from the asset's `physics.usda`;
the clip positions, the latch's release direction, the thresholds and the whole demo are guesses to tune.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the toolbox, a monitor, `success()` and `score()` |
| `geometry.py` | Asset dimensions from the USD, placement on the station, clip and lid positions |
| `scenekit.py` | Copied from `lock-before-loading`: loads a USD articulation and reads its joint angles |
| `demo.py` | First-pass scripted bimanual demo |
| `inspect_asset.py` | Local (no GPU) script that prints the asset's bodies, joints, limits, colliders and size |
| `test_toolbox_open.py`, `conftest.py`, `_shim/` | Local scoring tests with a stand-in for `yam_arena`. Delete `_shim/` before submitting |

## Setup

1. Copy the whole downloaded `storage_toolbox_standard_hardCase_d01` folder to `assets/toolbox/`, so that
   `assets/toolbox/simready_usd/sm_storage_toolbox_standard_hardCase_d01_01.usd` exists.
2. Look at what the asset really contains (your own machine, `pip install usd-core`):

       python inspect_asset.py assets/toolbox/simready_usd/sm_storage_toolbox_standard_hardCase_d01_01.usd

3. Run the scoring tests (no GPU): `python -m pytest -q test_toolbox_open.py`.

## Run on the cloud machine

Same pattern as the other tasks (not yet run for this one):

```bash
python demo.py task.py demo_out --max-steps 1200

inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Open the toolbox: release both latches and swing the lid all the way open." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/toolbox-dream
```

## Debugging, in the order I'd do it

1. **Does the toolbox load as an articulation?** The task raises an error listing the joint names it found. If the
   list is empty, the USD has no articulation root. `inspect_asset.py` shows whether one exists once the payloads
   compose (the PhysX payload may add it). If none, make `assets/toolbox/toolbox.usda` that references the asset and
   applies `PhysicsArticulationRootAPI` to `/RootNode`, and point `USD` in `task.py` at it. Don't add a second root
   if one already exists: nested roots fail.
2. **Does the latch actually block the lid?** Before running the demo, push the lid's front edge with the latch
   engaged. If the lid opens anyway, the hooks don't block it and the task has a shortcut: fix the collision shapes or
   add a blocker. Run this check before anything else about scoring.
3. **Where does the latch rest, and which way does it release?** Check the printed `latch` angle in the first
   frames. If it is not near 0, set the initial `joint_pos` in `task.py`. Tune `LATCH_RELEASED_DEG` and the demo's
   `RELEASE_DIR`.
4. **Clip positions and reach.** Replace `CLIP_TAB`, `CLIP_Y` and `BOX_XY` in `geometry.py` after looking at the
   scene. Check `worst_ik_mm` in `result.json`. The lid's free edge is about 0.48 m above the table when fully open,
   which may be out of reach: that is why the demo's last stage pushes the lid over from underneath, at a small radius
   from the hinge. By the USD's numbers the lid's weight should carry it onto the stop once it is past about
   80 degrees. Confirm this in the sim.
5. **Does the box slide?** It is a 3 kg floating body. If it moves more than 3 cm, raise `friction` on the `UsdBody`
   or fix its base in the wrapper USD.
6. **Joint feel.** The lid hinge and handle have no friction in the asset; `joint_damping` in `task.py` is what
   settles them. If the lid bounces off the stop, raise it. `STILL_DEG` and `STILL_STEPS` set how still the lid has to
   be.
7. **Grasp.** The lid-edge grasp in `demo.py` (fingers closing along x across the lip) is a guess.

## Scoring

| Score | Condition |
|---|---|
| 1.0 | lid open at least 85 degrees and still (the hinge stops at 90), toolbox moved less than 3 cm; success holds 1 s |
| 0.75 | lid open at least 45 degrees |
| 0.5 | lid open at least 10 degrees |
| 0.25 | latch bar turned at least 10 degrees from its rest angle, lid still shut |
| 0 | anything else, or the toolbox slid more than 3 cm at any point |

Angles are taken from the starting rest angles with absolute values, so the joints' signs don't matter.

## Credit

The toolbox is NVIDIA's SimReady asset `storage_toolbox_standard_hardCase_d01`, CC-BY-4.0. Keep the asset's
`LICENSE.txt` with it.
