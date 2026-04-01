# custom imports
from robot import Robot, JOINT_LIMITS_SIMULATION
from path import StackingPath
from transform import Transformation

# imports
import numpy as np

# initialize the robot and set the joint limits
rb = Robot()
rb.set_lim(JOINT_LIMITS_SIMULATION)

# set the loading and placing points
z_height = 0.03 # gripper center -> gripper end distance

load = np.array(
    [0.0, 0.15, z_height]
)

place = np.array(
    [0.09, 0.15, z_height]
)

# create the stacking trajectory
sp = StackingPath(load, place, n_blocks=5, h_blocks=0.02)

# create an array with the joint angles to send to the controller
q_arr = []

# iterate over all the points in the trajectory
for i, p in enumerate(sp.points):
    # convert the points from the trajectory into a transformation matrix
    target = Transformation.fromEuler(roll=np.pi, T=p) # roll fixed to pi to have the gripper facing down
    # compute the IK for all points
    q, success, _ = rb.IK(target, orientation_mode='full', weights=(1, 1e-3))

    # open the gripper if placing, close if picking up, and use the last known state elsewhere
    if i in sp.pick_idx:
        print('pick', i)
        q[-1] = rb.gripper_close
    elif i in sp.place_idx:
        print('place', i)
        q[-1] = rb.gripper_open
    else:
        if i == 0:
            # start with gripper open
            q[-1] = rb.gripper_open
        else:
            q[-1] = q_arr[-1][-1]

    q[-2] = np.pi/2 # set the wrist angle to not have the jaw interfere with the blocks

    # only add the joint angles to the output if the IK was a success
    if success:
        q_arr.append(q)
    else:
        print(f'point {p} failed')

# save the joint angles to a csv file to run the controller
np.savetxt('stack_path.csv', q_arr, delimiter=', ')
