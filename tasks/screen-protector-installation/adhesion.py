"""Reduced-order adhesive patches on a rigid protector. Units are metres and newtons."""
from task import PATCHES, PITCH
import numpy as np
from scipy.ndimage import binary_fill_holes


class Film:
    def __init__(self):
        self.shape = (24,12)
        yy,xx = np.meshgrid(np.linspace(-.071,.071,24),np.linspace(-.034,.034,12),indexing='ij')
        self.sites = np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,-.0008)])
        self.bond = np.zeros(xx.size,bool)
        self.anchors = np.zeros_like(self.sites)
        self.air = np.zeros(xx.size, bool)

    def step(self, state):
        glass, tray = state['glass'], state['phone']
        world = self.sites @ glass[:3,:3].T + glass[:3,3]
        local = (world-tray[:3,3]) @ tray[:3,:3]
        covered = np.zeros(len(world),bool)
        for i in range(PATCHES):
            sheet = state[f'liner{i}']
            on_sheet = (world-sheet[:3,3]) @ sheet[:3,:3]
            covered |= (abs(on_sheet[:,0]) <= .0352) & (abs(on_sheet[:,1]) <= PITCH/2+.0002) & (abs(on_sheet[:,2]) < .002)
        gap = local[:,2]-.018
        inside = (abs(local[:,0])<=.037) & (abs(local[:,1])<=.077)
        # A pocket bounded on four sides by bonded sites loses its escape route.
        sealed = binary_fill_holes(self.bond.reshape(self.shape)).ravel() & ~self.bond
        self.air = sealed & ~covered
        new = ~self.air & ~covered & inside & (abs(gap)<.0012) & ~self.bond
        new &= glass[:3,2] @ tray[:3,2] > .90
        self.anchors[new] = local[new]
        self.anchors[new,2] = .018
        self.bond |= new
        delta = self.anchors-local
        self.bond &= np.linalg.norm(delta,axis=1)<.004
        active = self.bond & (abs(gap)<.0015)
        patch_forces = np.where(self.bond[:,None], delta @ tray[:3,:3].T, 0.) * .5
        force = patch_forces.sum(axis=0)
        torque = np.cross(world-glass[:3,3],patch_forces).sum(axis=0)
        tray_torque = np.cross(world-tray[:3,3],-patch_forces).sum(axis=0)
        alignment = np.max(np.linalg.norm(local[:,:2]-self.sites[:,:2],axis=1))
        result = dict(bonded=float(active.mean()),covered=float(covered.mean()),
                      corner_error_mm=float(alignment*1000), air_sites=int(self.air.sum()),
                      gap_mm=float(abs(gap).max()*1000))
        result['success'] = bool(active.all() and not covered.any() and alignment<.002)
        self.covered = covered
        return result, {'glass':(force,torque),'phone':(-force,tray_torque)}

    def panel(self, result):
        import cv2
        panel = np.full((360,240,3),26,np.uint8)
        grid = np.full((len(self.bond),3),[220,160,75],np.uint8)
        grid[self.covered] = [170,178,187]
        grid[self.bond] = [39,166,147]
        grid[self.air] = [126,115,216]
        panel[45:237,76:172] = cv2.resize(grid.reshape(*self.shape,3)[::-1],(96,192),interpolation=cv2.INTER_NEAREST)
        for text,y in [('Adhesion model',28),('Bonded   %.0f%%'%(100*result['bonded']),265),('Covered  %.0f%%'%(100*result['covered']),289),('Offset   %.2f mm'%result['corner_error_mm'],313),('Max gap  %.2f mm'%result['gap_mm'],337)]:
            cv2.putText(panel,text,(18,y),cv2.FONT_HERSHEY_SIMPLEX,.56,(236,238,240),1,cv2.LINE_AA)
        return panel
