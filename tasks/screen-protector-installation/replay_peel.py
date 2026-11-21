"""Export the main camera during the measured peel, at one third speed."""
import json
from pathlib import Path
import subprocess


def export(directory, rows):
    start = next(r['step'] for r in rows if r['peel_front'] >= 0)
    stop = next(r['step'] for r in rows if r['step'] >= start and r['covered'] == 0)
    subprocess.run(['ffmpeg','-y','-v','error','-ss',str(max(0,start/20-.8)),
                    '-i',str(directory/'demo.mp4'),'-vf',
                    f'trim=duration={(stop-start)/20+1.6},crop=960:540:0:0,setpts=3*(PTS-STARTPTS)',
                    '-r','20','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',
                    str(directory/'peel-slow.mp4')],check=True)


if __name__ == '__main__':
    folder = Path(__file__).resolve().parent/'clean'
    export(folder, json.loads((folder/'trace.json').read_text()))
