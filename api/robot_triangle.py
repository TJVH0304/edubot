from robot import Robot, JOINT_LIMITS_SIMULATION
from path import TrianglePath
from transform import Transformation

import numpy as np


rb = Robot()
rb.set_lim(JOINT_LIMITS_SIMULATION)

tp = TrianglePath([
    [-0.075, 0.2, 0.05],
    [0, 0.2, 0.2],
    [0.075, 0.2, 0.05]
], points_per_edge=3)


q_arr = []
for p in tp.points:
    target = Transformation.fromEuler(yaw=np.pi, T=p)
    q, success, _ = rb.IK(target, orientation_mode='axis', weights=(1, 0))

    if success:
        q_arr.append(q)
    else:
        print(f'point {p} failed')

np.savetxt('ros_ws/triangle_path.csv', q_arr, delimiter=', ')

