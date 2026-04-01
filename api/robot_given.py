# custom imports
from robot import Robot, JOINT_LIMITS_SIMULATION
from path import RobotPath
from transform import Transformation

# imports
import numpy as np

# initialize the robot and set the joint limits
rb = Robot()
rb.set_lim(JOINT_LIMITS_SIMULATION)

# create the targets
targets = [
        Transformation.fromEuler(roll=0, pitch=1.750, yaw=0.650, T=np.array([0.2, 0.2, 0.2])),
        Transformation.fromEuler(roll=0, pitch=0, yaw=-0.785, T=np.array([0.2, 0.1, 0.4])),
        Transformation.fromEuler(roll=0, pitch=-0.785, yaw=1.570, T=np.array([0, 0, 0.4])),
        Transformation.fromEuler(roll=3.141, pitch=0, yaw=0, T=np.array([0, 0, 0.7])),
        Transformation.fromEuler(roll=-0.785, pitch=0, yaw=3.141, T=np.array([0, 0.0452, 0.45]))
    ]

# create an array with the joint angles to send to the controller
q_arr = []

# iterate over all the points in the trajectory
for t in targets:
    # compute the IK for all points
    q, success, _ = rb.IK(t, orientation_mode='full', weights=(1, 1e-3))

    # only add the joint angles to the output if the IK was a success
    if success:
        q_arr.append(q)
    else:
        print(f'point {t.t()} failed')

# save the joint angles to a csv file to run the controller
np.savetxt('given_path.csv', q_arr, delimiter=', ')

