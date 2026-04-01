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
# q_out = []
# steps_between = 20  # increase for smoother motion

# for i in range(len(q_arr) - 1):
#     q_start = q_arr[i]
#     q_end = q_arr[i + 1]

#     for t in np.linspace(0, 1, steps_between, endpoint=False):
#         q_interp = (1 - t) * q_start + t * q_end
#         q_out.append(q_interp)

# # append the final point
# q_out.append(q_arr[-1])

# q_out = np.array(q_out)


np.savetxt('ros_ws/triangle_path.csv', q_arr, delimiter=', ')

