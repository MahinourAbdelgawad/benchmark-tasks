"""Cartesian controller. Every arm motion is issued as joint and jaw targets."""
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena import ik, station
from yam_arena.demo import grip_rotation
from yam_arena.runtime import tcp_world, arm_q
from task import START, TOP, TAB_Y

STAGES = [('Approach tools',35),('Lower to tools',30),('Pinch end tab and carrier',25),('Lift backed glass',55),('Peel plastic liner',95),
          ('Finish peel and set aside',40),('Release backing',15),('Clear workspace',25),
          ('Align above phone',65),('Fine alignment',32),('Lower first edge',55),('Roll glass flat',50),
          ('Vent carrier',15),('Remove carrier',40),('Inspect',25)]


class Controller:
    def __init__(self, base, crooked):
        self.robots = [base.scene[k+'_robot'] for k in ('left','right')]
        self.crooked = crooked
        self.last = None

    def command(self, stage, fraction, state):
        hands = [tcp_world(r) for r in self.robots]
        joints = [arm_q(r) for r in self.robots]
        rotations = [ik.tcp_pose(q,station.BASES[s])[1] for s,q in zip(('left','right'),joints)]
        glass = state['glass']
        centre = glass[:3,3]
        if stage != self.last:
            self.start = np.array(hands)
            self.glass_start = glass.copy()
            self.rot = rotations
            self.last = stage
            if stage == 'Peel plastic liner':
                self.tab_start = hands[1].copy()
            if stage == 'Align above phone':
                self.bias = np.zeros(3)
                self.rot_bias = np.zeros(3)
        goal = self.start.copy()
        orient = list(self.rot)
        left = stage not in ('Approach tools','Lower to tools')
        right = stage in ('Pinch end tab and carrier','Lift backed glass','Peel plastic liner','Finish peel and set aside')
        if stage in ('Approach tools','Lower to tools','Pinch end tab and carrier','Lift backed glass'):
            lift = {'Approach tools':.082,'Lift backed glass':.094}.get(stage,0.)
            goal[0] = [START[0],START[1]+.028,TOP+.026+lift]
            goal[1] = [START[0],START[1]+TAB_Y,TOP+.018+lift]
            orient = [grip_rotation(),grip_rotation()]
        if stage == 'Peel plastic liner':
            # Draw down around the starting edge and then back underneath the glass.
            u = ik.smoothstep(fraction)
            goal[1] = self.tab_start + [.044*u, -.090*u, -.065*np.sin(u*np.pi/2)]
        if stage == 'Finish peel and set aside':
            goal[1] = [.17,-.20,TOP+.041]
        if stage == 'Clear workspace':
            goal[1] = [.14,-.24,TOP+.110]
        placement = stage in ('Align above phone','Fine alignment','Lower first edge','Roll glass flat','Vent carrier')
        if placement:
            phone = state['phone']
            u = ik.smoothstep(min(1.,fraction*1.2))
            angle = 0.09 if stage in ('Align above phone','Fine alignment','Lower first edge') else 0.
            target_rotation = phone[:3,:3] @ Rotation.from_rotvec([angle,0.,0.]).as_matrix()
            height = {'Align above phone':.064,'Fine alignment':.040}.get(stage,.0188+.0725*np.sin(angle))
            final = phone[:3,3]+phone[:3,:3] @ [(.006 if self.crooked else 0),0,height]
            target = self.glass_start[:3,3]*(1-u)+u*final
            rotation = ik.slerp_matrix(self.glass_start[:3,:3],target_rotation,u)
            error = target-centre
            angular = Rotation.from_matrix(rotation @ glass[:3,:3].T).as_rotvec()
            # Compensate persistent grip deflection once the interpolation has
            # stopped. Integrating transport lag biases the first touchdown.
            if u >= .999:
                self.bias = np.clip(self.bias+.07*error,-.006,.006)
                self.rot_bias = np.clip(self.rot_bias+.05*angular,-.06,.06)
            correction = Rotation.from_rotvec(angular+self.rot_bias).as_matrix()
            goal[0] = centre+error+self.bias+correction @ (hands[0]-centre)
            orient[0] = correction @ rotations[0]
        if stage == 'Remove carrier':
            goal[0] += [0,.015,.072]
        self.right_goal = goal[1].copy()
        self.left_goal = goal[0].copy()
        result=[]
        for i,side in enumerate(('left','right')):
            blend = ik.smoothstep(fraction)
            if (stage=='Peel plastic liner' and i==1) or (placement and i==0):
                blend=1.
            pos = self.start[i]+blend*(goal[i]-self.start[i])
            rot = ik.slerp_matrix(self.rot[i],orient[i],blend)
            q,*_ = ik.solve(pos,ik.matrix_to_quat_xyzw(rot),joints[i],station.BASES[side],iters=70)
            result.extend([*q,0. if (left,right)[i] else 1.])
        return result
