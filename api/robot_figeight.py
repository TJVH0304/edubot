from robot import Robot, JOINT_LIMITS_SIMULATION
from path import FigureEightPath
from transform import Transformation

import numpy as np


rb = Robot()
rb.set_lim(JOINT_LIMITS_SIMULATION)

fp = FigureEightPath(center=(0, 0.3, 0.2), points=40, scale=0.1)


q_arr = []
for i, p in enumerate(fp.points):
    target = Transformation.fromEuler(yaw=np.pi, T=p)
    q, success, _ = rb.IK(target, orientation_mode='full', weights=(1, 0))

    if success:
        print(i, 'ok')
        q_arr.append(q)
    else:
        print(f'point {p} failed')


np.savetxt('figeight_path.csv', q_arr, delimiter=', ')
