# custom imports
from robot import Robot, JOINT_LIMITS_SIMULATION
from path import TrianglePath
from transform import Transformation

# imports
import numpy as np

# initialize the robot and set the joint limits
rb = Robot()
rb.set_lim(JOINT_LIMITS_SIMULATION)

# create the triangle trajectory
tp = TrianglePath([
    [-0.075, 0.2, 0.05],
    [0, 0.2, 0.2],
    [0.075, 0.2, 0.05]
], points_per_edge=3)

# create an array with the joint angles to send to the controller
q_arr = []

# iterate over all the points in the trajectory
for p in tp.points:
    # convert the points from the trajectory into a transformation matrix
    target = Transformation.fromEuler(yaw=np.pi, T=p)
    # compute the IK for all points
    q, success, _ = rb.IK(target, orientation_mode='axis', weights=(1, 0))

    # only add the joint angles to the output if the IK was a success
    if success:
        q_arr.append(q)
    else:
        print(f'point {p} failed')

# save the joint angles to a csv file to run the controller
np.savetxt('triangle_path.csv', q_arr, delimiter=', ')

