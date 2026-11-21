# Install a loose phone screen protector

The phone and the backed protector start beside each other on the YAM table. Both arms grasp the assembly on the table, the left on the teal removable carrier and the right on the green folded tab at the short end of the plastic liner. They lift together so the backing stays attached before the peel. After peeling the backing and putting it aside, the left arm carries the bare protector over the phone, touches down one edge, rolls the protector flat, vents the carrier and lifts the tool away.

## Replay

Run from this directory on a GPU host with the installed Isaac Lab-Arena/YAM environment and ffmpeg. The local wrapper selects the workspace's environment. Alternatively use `python -B demo.py --condition clean` in an activated Arena environment.

```bash
./run.sh clean
./run.sh crooked
```

The full-speed demo uses the close camera as the main view. The clean run also exports `peel-slow.mp4`, a one-third-speed close-up cut from the native recording around the measured separation.

The crooked control commands the same pickup, peel and application with a 6 mm placement offset. Each condition directory contains the native station and oblique-camera recording, a scored endpoint, and a trace. The colored grid beside the two views is a simulator diagnostic. Grey is covered by backing, amber is exposed, green is bonded, and violet indicates a sealed pocket. The phase label identifies the current arm command.

## Assets and mechanics

All task-specific assets are generated locally by `assets.py` and included as USDA files. The arms, table and overhead camera come from YAM Arena. The phone measures 76 × 156 × 18 mm and is a free rigid body. The 70 × 145 × 1.6 mm protector carries a pre-fitted vacuum handle. The backing is a single 70 × 145 mm tinted plastic sheet with a folded 23 mm wide grasp tab at the middle of its short end. Eight hinged sections approximate its mechanics. The glossy, translucent surface interpolates their measured positions and tangents. A weak rotational spring at each hinge lets the discarded sheet relax. The tab’s pinch region is thickened to 4 mm for gripper contact. These are purpose-built simulation tools, with no measured retail SKU geometry or peel-force calibration.

The grippers manipulate bodies through PhysX contacts. A fixed joint represents the pre-engaged vacuum seal, and a tool vent releases that joint. The backing sections are linked by transverse hinges. A pinched tab and a relative downward draw load the leading edge. Its break force drops from 100 N to 0.23 N. Measured separation or rotation exposes the next edge. These values are numerical parameters of the segmented model. Covered sites are computed from each section's measured pose. Local screen bonds form within 1.2 mm of the screen, anchor where they form, apply spring forces to both bodies and release after 4 mm displacement. Optical bubble and dust inspection and real hardware validation are outside the evidence provided by this simulation. The initial screen is clean.

The privileged scripted controller reads body poses for grasp approach and visual-servo-style alignment. The final score uses the measured protector and phone geometry. Success requires all 288 sites bonded, each within 1.5 mm of the screen, zero remaining liner coverage, and less than 2 mm maximum in-plane displacement. These conditions must hold for the final second with the carrier clear of the glass and the liner set aside. The scorer also requires the protector to start beside the phone, lift with its backing, pass through partial coverage, and become fully exposed before transfer. Replay the independent checks with `python -B score.py clean/trace.json`. Each trial runs for 602 actions, or 30.1 simulated seconds at 20 Hz.
