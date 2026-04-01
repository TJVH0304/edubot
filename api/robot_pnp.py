from robot import Robot, JOINT_LIMITS_SIMULATION
from path import PickAndPlacePath
from transform import Transformation

import numpy as np


rb = Robot()
rb.set_lim(JOINT_LIMITS_SIMULATION)

rb.gripper_close = 0.12
rb.gripper_open = 0.2

z_height = 0.055


load = np.array(
    [-0.09, 0.11, z_height]
)

place = np.array(
    [0.1, 0.13, z_height]
)

home = np.array([
    0.0, 0.15, z_height
]) 

pp = PickAndPlacePath(load, place, home, h_blocks=0.02)

print(pp.pickup_points)
print(pp.place_points)

q_arr = []
for i, p in enumerate(pp.points):
    target = Transformation.fromEuler(roll=np.pi, T=p)
    q, success, _ = rb.IK(target, orientation_mode='axis', weights=(1, 1e-3))

    if i in pp.pick_idx:
        print('pick', i)
        q[-1] = rb.gripper_close
    elif i in pp.place_idx:
        print('place', i)
        q[-1] = rb.gripper_open
    else:
        if i == 0:
            q[-1] = rb.gripper_open
        else:
            q[-1] = q_arr[-1][-1]
    
    q[-2] = 0

    if success:
        q_arr.append(q)
    else:
        print(f'point {p} failed')

# np.savetxt('ros_ws/pnp_path.csv', q_arr, delimiter=', ')
np.savetxt('pnp_path.csv', q_arr, delimiter=', ')
