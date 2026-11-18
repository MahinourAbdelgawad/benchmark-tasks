# Release both toolbox clips and open the lid

The two YAM arms face a closed hard-plastic toolbox on the station table. Release the two yellow clips, then lift and push the lid to its 90° stop. The base is anchored at `(0.37, 0, 0.7485)` m, with the hinge away from the arms. The case is 31.4 × 53.1 cm in plan and the clips are 31 cm apart.

## Asset

The original USD loads with its geometry, materials, bodies and PhysX colliders. It uses metres and Z-up. Its two clips are disjoint mesh islands on one rigid link and one hinge, and it has no articulation root.

`assets/toolbox_open.usdc` is the task-ready layer. It references the original asset, splits the clip geometry and UVs, gives each clip an independent passive joint, and adds small hook and catch colliders. It also targets the actual rigid bodies in the lid and handle joints. At spawn, the task adapter anchors the base at its station pose. The render asset and its relative dependencies should stay together under `assets/`.

The converted layer is included. With a Python environment that has NumPy and the OpenUSD `pxr` bindings, inspect or regenerate it from this directory.

```bash
python inspect_asset.py assets/storage_toolbox*/simready_usd/sm*.usd
python prepare_asset.py
python inspect_asset.py assets/toolbox_open.usdc
```

## Run the scripted demonstration

Activate the YAM Isaac Lab-Arena environment on a machine with a compatible NVIDIA GPU. From the repository root, run the submission entry point through Inspect Robots.

```bash
source /data/IsaacLab-Arena/yam.env
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/toolbox-open/task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Release both yellow toolbox clips, then lift and push the lid all the way open." \
  --scorer task_score --max-steps 1200 --store-frames \
  --log-dir ~/yam-eval/toolbox-open-demo
```

The open-loop demonstration approaches 30° below horizontal with the jaws fully open, pinches each lower tab across its width and swings both clips outward. The arms lift the lid from the released clips, then one gripper holds the lid while the other gets underneath and pushes. The fixed layout uses seed 0 by default. A trial takes approximately 27 simulated seconds to reach success and may use up to 1,200 steps including settling time.

## Run Dream-1

Create a Dream Labs API key at `platform.dreamlabs.ai/api-keys`, then save it where the installed Dream policy reads it. Replace the placeholder without committing the key.

```bash
mkdir -p ~/.dreamlabs
printf 'DREAM_API_KEY=%s\n' 'dl_live_REPLACE_WITH_YOUR_KEY' > ~/.dreamlabs/credentials
chmod 600 ~/.dreamlabs/credentials
```

Run the required model evaluation from the repository root.

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/toolbox-open/task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Release both yellow toolbox clips, then lift and push the lid all the way open." \
  --scorer task_score --max-steps 2400 --store-frames \
  --log-dir ~/yam-eval/toolbox-open-dream
```

The model trial allows 2,400 steps at 20 Hz, or 120 simulated seconds. The layout is fixed across seeds. Reset restores the lid, handle, and both clip joints to their closed positions and velocities, and returns both arms to ready. On the validation VM with frame capture enabled, the seed-0 scripted trial reached full score in 540 steps, 27 simulated seconds and approximately 129 seconds wall-clock. The Dream-1 trial reached the 2,400-step limit with score 0.0 in approximately 345 seconds wall-clock.

## Check clasp reach

Run `python measure_grasp.py` with the same OpenUSD Python environment used for asset inspection. It prints the clasp bounds, the two mesh intersections along the jaw-closing axis, the commanded TCP, and the TCP computed from the YAM URDF. Coordinates are in station metres.

At `x = 0.222 m` and `z = 0.877 m`, the lower tab spans 71.92 mm along station Y. The target TCP is `(0.214206, ±0.155000, 0.881500)`. Positive Y is the left arm. The jaws enter at 93.9 mm and command a 63.85 mm gap to apply pressure. The demo logs the actual TCP and opening after closing in `result.json` under `grasp_tracking`. A GPU rollout measured openings of 71.69 mm on the left and 71.63 mm on the right.

## Scoring

| Score | Condition |
|---|---|
| 0 | Neither clip is at least 25° open |
| 0.25 | One clip is at least 25° open |
| 0.5 | Both clips are at least 25° open |
| 0.75 | Both clips are released and the lid is at least 45° open |
| 1.0 | Both clips are released and the lid stays at or above 85° with angular speed at most 2°/s for one second |

Partial credit uses the final state. Full credit uses Arena's one-second success hold.

The fixed-layout GPU demo scored 1.0 after 540 control steps (27 simulated seconds), with the lid at 90.0° and the clips at 82.0° and 68.8°. Peak nominal IK position error later in the motion was approximately 131 mm. The recorded grasp positions were within 0.5 mm of their targets. Separate trials for hook contact with zero or one clip released remain unverified.

Original asset by NVIDIA, `storage_toolbox_standard_hardCase_d01`, CC BY 4.0. The license is included in the original asset's `.metadata` directory.
