# custom imports
from robot import Robot, JOINT_LIMITS_SIMULATION
from path import PickAndPlacePath
from transform import Transformation

# imports
import numpy as np

# initialize the robot and set the joint limits
rb = Robot()
rb.set_lim(JOINT_LIMITS_SIMULATION)

# redefine the opening and closing angles for the jaw
rb.gripper_close = 0.12
rb.gripper_open = 0.2

# set the loading, placing, and home points
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

# create the PnP trajectory
pp = PickAndPlacePath(load, place, home, h_blocks=0.02)

# create an array with the joint angles to send to the controller
q_arr = []

# iterate over all the points in the trajectory
for i, p in enumerate(pp.points):
    # convert the points from the trajectory into a transformation matrix
    target = Transformation.fromEuler(roll=np.pi, T=p)
    # compute the IK for all points
    q, success, _ = rb.IK(target, orientation_mode='axis', weights=(1, 1e-3))

    # open the gripper if placing, close if picking up, and use the last known state elsewhere
    if i in pp.pick_idx:
        print('pick', i)
        q[-1] = rb.gripper_close
    elif i in pp.place_idx:
        print('place', i)
        q[-1] = rb.gripper_open
    else:
        if i == 0:
            # start with gripper open
            q[-1] = rb.gripper_open
        else:
            q[-1] = q_arr[-1][-1]
    
    q[-2] = 0 # set the wrist angle to not have the jaw interfere with the grape

    # only add the joint angles to the output if the IK was a success
    if success:
        q_arr.append(q)
    else:
        print(f'point {p} failed')

# save the joint angles to a csv file to run the controller
np.savetxt('pnp_path.csv', q_arr, delimiter=', ')
