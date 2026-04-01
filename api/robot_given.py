from robot import Robot, JOINT_LIMITS_SIMULATION
from path import RobotPath
from transform import Transformation

import numpy as np


rb = Robot()
rb.set_lim(JOINT_LIMITS_SIMULATION)

targets = [
        Transformation.fromEuler(roll=0, pitch=1.750, yaw=0.650, T=np.array([0.2, 0.2, 0.2])),
        Transformation.fromEuler(roll=0, pitch=0, yaw=-0.785, T=np.array([0.2, 0.1, 0.4])),
        Transformation.fromEuler(roll=0, pitch=-0.785, yaw=1.570, T=np.array([0, 0, 0.4])),
        Transformation.fromEuler(roll=3.141, pitch=0, yaw=0, T=np.array([0, 0, 0.7])),
        Transformation.fromEuler(roll=-0.785, pitch=0, yaw=3.141, T=np.array([0, 0.0452, 0.45]))
    ]


q_arr = []
for t in targets:
    print(t.t())
    q, success, _ = rb.IK(t, orientation_mode='full', weights=(1, 1e-3))

    if success:
        q_arr.append(q)
    else:
        print(f'point {t.t()} failed')


np.savetxt('given_path.csv', q_arr, delimiter=', ')

