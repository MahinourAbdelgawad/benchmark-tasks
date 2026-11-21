"""A loose backed protector beside a stationary phone. World dimensions in metres."""
from yam_arena.task_file import TABLE_TOP

TOP = TABLE_TOP['station']
START = (.20,.078)
PHONE = (.333,.042)
PATCHES = 8
PITCH = .145 / PATCHES
TAB_Y = -.097
def patch_y(i):
    return (i-(PATCHES-1)/2)*PITCH

POSITIONS = {'phone': (*PHONE,TOP), 'glass': (*START,TOP+.0017),
             'carrier': (START[0],START[1]+.028,TOP+.0025)}
for i in range(PATCHES):
    POSITIONS[f'liner{i}'] = (START[0],START[1]+patch_y(i),TOP+.00045)
