# custom imports
from robot import Robot, JOINT_LIMITS_SIMULATION
from path import FigureEightPath
from transform import Transformation

# imports
import numpy as np

# initialize the robot and set the joint limits
rb = Robot()
rb.set_lim(JOINT_LIMITS_SIMULATION)

# create the figure eight trajectory
fp = FigureEightPath(center=(0, 0.3, 0.2), points=40, scale=0.1)

# create an array with the joint angles to send to the controller
q_arr = []

# iterate over all the points in the trajectory
for i, p in enumerate(fp.points):
    # convert the points from the trajectory into a transformation matrix
    target = Transformation.fromEuler(yaw=np.pi, T=p)
    # compute the IK for all points
    q, success, _ = rb.IK(target, orientation_mode='full', weights=(1, 0))

    # only add the joint angles to the output if the IK was a success
    if success:
        q_arr.append(q)
    else:
        print(f'point {p} failed')

# save the joint angles to a csv file to run the controller
np.savetxt('figeight_path.csv', q_arr, delimiter=', ')
