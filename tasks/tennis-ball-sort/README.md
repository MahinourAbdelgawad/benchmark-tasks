# Sort tennis balls by firmness

Two unmarked 67 mm yellow-green balls have the same spherical rest shape and nominal 57 g mass. Native PhysX volume deformables provide gripper contact and shape change. Effective Young’s moduli of 65 and 180 kPa are provisional tuning values. The model approximates loss of internal pressure through compliance. It does not simulate gas or claim a pressure in psi.

From the repository root in the GPU Arena environment:

```bash
source /data/IsaacLab-Arena/yam.env
TENNIS_SEED=0 inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/tennis-ball-sort/task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --max-steps 1200 --scorer task_score \
  --instruction "Compare the tennis balls by squeezing them. Leave the softer ball in the orange tray and the firmer ball in the blue tray."
```

Budget is 60 simulated seconds at 20 Hz. Wall time is unmeasured. Repeat with TENNIS_SEED=1 to reverse the two samples. The starting centers and tray locations stay fixed. The scorer awards half credit per correctly contained ball resting at floor height at the end. It evaluates the sorting outcome and cannot verify that a squeeze informed the decision.

Before collecting submission media, verify native scene launch and reset, grasp stability, and camera-visible deformation at matched jaw loading. Measure two real same-model balls under matched loads and fit both simulated compression responses. A full-score demonstration and model scores are pending.
