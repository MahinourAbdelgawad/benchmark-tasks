"""Native Isaac Lab-Arena scripted run. All poses used by the evaluator are measured."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np

from task import POSITIONS, PATCHES
from yam_arena import station
from script import Controller, STAGES
from yam_arena.analysis import pose7_to_matrix

HERE = Path(__file__).resolve().parent


def setup():
    from assembly import spawn_assembly
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, enable_cameras=True)
    from isaaclab.sim import UsdFileCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from yam_arena import scene
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from yam_arena.embodiment import BimanualYamEmbodiment

    rig = station.rig('demo')
    robot = BimanualYamEmbodiment(enable_cameras=True, wrist_cam_mount='link', overview_cam=True, contact_sensors=False, rig=rig)
    from scipy.spatial.transform import Rotation
    eye = np.array([.56,-.27,1.03])
    back = eye - [.23,-.005,.802]
    back /= np.linalg.norm(back)
    across = np.cross([0,0,1],back)
    across /= np.linalg.norm(across)
    camera = robot.camera_config.overview_cam
    camera.width, camera.height = 960, 540
    camera.offset.pos = tuple(eye)
    camera.offset.rot = tuple(Rotation.from_matrix(np.column_stack([across,np.cross(back,across),back])).as_quat())
    camera.offset.convention = 'opengl'
    camera.spawn.focal_length = 15.
    assets = scene.task_room('station', rig.top_cam)
    for name, position in POSITIONS.items():
        asset = ('tab' if name=='liner0' else 'strip') if name.startswith('liner') else name
        opts = {'func':spawn_assembly} if name==f'liner{PATCHES-1}' else {}
        assets.append(Object(name=name, object_type=ObjectType.RIGID,
                             spawner_cfg=UsdFileCfg(usd_path=str(HERE / (asset+'.usda')), **opts),
                             initial_pose=Pose(position_xyz=position)))
    world = scene.IsaacLabArenaEnvironment(name='screen_protector_installation', scene=scene.Scene(assets),
                                           embodiment=robot, env_cfg_callback=scene.env_cfg_callback(20))
    env = ArenaEnvBuilder(world, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device='cuda:0')).make_registered()
    return launcher, env


def poses(base):
    return {name:pose7_to_matrix(base.scene[name].data.root_link_pose_w.torch[0].cpu().numpy())
            for name in POSITIONS}


def main():
    import torch
    import cv2
    from adhesion import Film
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=['clean','crooked'], default='clean')
    args = parser.parse_args()
    launcher, env = setup()
    base = env.unwrapped
    parts = {name:base.scene[name] for name in ('glass','phone')}
    env.reset()
    controller = Controller(base,args.condition=='crooked')
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    seal = next(p for p in stage.Traverse() if p.GetName()=='vacuum_seal')
    from skin import ReleaseSheet
    skin = ReleaseSheet(stage,poses(base))
    film = Film()
    from assembly import PeelFront
    peel = PeelFront(stage,base.scene['right_robot'])
    video = None
    snapshots = {}
    history = []
    import tempfile
    import shutil
    destination = HERE / args.condition
    out = Path(tempfile.mkdtemp(prefix=f'.{args.condition}-', dir=HERE))
    try:
        schedule = [(phase,(i+1)/count) for phase,count in STAGES for i in range(count)]
        for step,(phase,fraction) in enumerate(schedule):
            state = poses(base)
            skin.update(state)
            joint = controller.command(phase,fraction,state)
            peel.update(state,controller)
            if phase=='Vent carrier':
                seal.GetAttribute('physics:jointEnabled').Set(False)
            env.step(torch.tensor([joint], device=base.device,dtype=torch.float32))
            state = poses(base)
            result, forces = film.step(state)
            for name,(force,torque) in forces.items():
                def tensor(v):
                    return torch.as_tensor(v, device=base.device,dtype=torch.float32).reshape(1,1,3)
                parts[name].permanent_wrench_composer.set_forces_and_torques_index(forces=tensor(force),torques=tensor(torque),is_global=True)
            result.update(step=step, phase=phase, peel_front=peel.front, clip_grip_error=peel.grip_error, jaw_gap=peel.gap, centres={k:v[:3,3].tolist() for k,v in state.items()})
            history.append(result)
            rgb = base.scene['top_cam'].data.output['rgb'].torch[0,:,:,:3].cpu().numpy().copy()
            rgb = cv2.resize(rgb,(640,360))
            panel = film.panel(result)
            detail = base.scene['overview_cam'].data.output['rgb'].torch[0,:,:,:3].cpu().numpy()
            # Select a frame near half coverage during actual peeling, and
            # label the physical views separately from the diagnostic map.
            if phase in ('Peel plastic liner','Finish peel and set aside'):
                key, cost = 'Pulling the PET film', abs(result['covered']-.5)
            else:
                key = {'Lift backed glass':'Lift with backing', 'Clear workspace':'Backing discarded',
                       'Fine alignment':'Align over phone','Lower first edge':'First edge down',
                       'Inspect':'Carrier removed'}.get(phase)
                cost = 1-fraction
            if key and (key not in snapshots or cost<snapshots[key][0]):
                snapshots[key] = (cost,cv2.resize(detail,(480,270)))
            overview = cv2.resize(rgb,(320,180))
            diagnostics = cv2.copyMakeBorder(panel,0,0,40,40,cv2.BORDER_CONSTANT,value=(26,26,26))
            sidebar = np.vstack([overview,diagnostics])
            frame = np.hstack([cv2.resize(detail,(960,540)), sidebar])
            cv2.rectangle(frame,(0,0),(320,29),(22,28,32),-1)
            cv2.putText(frame,phase,(10,21),cv2.FONT_HERSHEY_SIMPLEX,.56,(245,245,245),1,cv2.LINE_AA)
            if not video:
                video = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x540','-r','20','-i','-','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'demo.mp4')],stdin=subprocess.PIPE)
            video.stdin.write(frame.tobytes())
            if step%20==0:
                print(step,result,flush=True)
        video.stdin.close()
        if video.wait():
            raise RuntimeError('video encoding failed')
        video = None
        if args.condition=='clean':
            tiles=[]
            for name,(_,view) in snapshots.items():
                tile=cv2.copyMakeBorder(view,28,0,0,0,cv2.BORDER_CONSTANT,value=(22,28,32))
                cv2.putText(tile,name,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.53,(245,245,245),1,cv2.LINE_AA)
                tiles.append(tile)
            sheet=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])])
            cv2.imwrite(str(out/'sequence.jpg'),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
        (out/'trace.json').write_text(json.dumps(history))
        cv2.imwrite(str(out/'endpoint.png'), cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        # Require a stable endpoint for the final second, after both hands withdraw.
        from score import evaluate
        checks = evaluate(history)
        passed = all(checks.values())
        summary = dict(condition=args.condition, success=passed, checks=checks, steps=len(history), endpoint=result)
        (out/'result.json').write_text(json.dumps(summary, indent=2)+'\n')
        if args.condition=='clean' and passed:
            from replay_peel import export
            export(out, history)
        destination.mkdir(exist_ok=True)
        for artifact in out.iterdir():
            artifact.replace(destination/artifact.name)
        print('ENDPOINT', summary,flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if video:
            video.stdin.close()
            if video.wait(): raise RuntimeError('video encoding failed')
        shutil.rmtree(out, ignore_errors=True)
        env.close()
    launcher.app.close()


if __name__=='__main__':
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    main()
