# Physical Interaction for Aerial and Space Robots – Group 6 Code

Please refer to the [edubot repository](https://github.com/BioMorphic-Intelligence-Lab/edubot) for installation instructions and the provided launch scripts and controllers.

## Repository Layout

Added w.r.t. the edubot repository are the following:

- _api/_, where the following python files are present:
  - _robot_, defining the custom _Robot_ class which houses the forward and inverse kinematics
  - _transform_, which features the custom api for transformation matrices using the _Transformation_ class
  - _path_, where custom _Path_ objects are defined to generate the trajectories to feed into the IK
  -  _robot\_*_, where the used paths are defined and fed into the robot to find the joint angles
-  _tasks/_, where extra code to complete some of the tasks can be found
-  _data/_, where the csv files used in the videos can be found

Additionally, the following controller nodes are added in _ros\_ws/src/python\_controllers/python\_controllers/_:

Python file | Command | File read | Use
-|-|-|-
_traj_ | traj | _path.csv_ | Move along an arbitrary path
_given\_traj_ | given | _given\_path.csv_ | Move to the 5 given points in task 2.1.2 if they are feasible
_triangle\_traj_ | triangle | _triangle\_path.csv_ | Move along the triangle
_figeight\_traj_ | figeight | _figeight\_path.csv_ | Move along the figure eight
_pnp\_traj_ | pnp | _pnp\_path.csv_ | Perform the pick and place trajectory
_stack\_traj_ | stack | _stack\_path.csv_ | Perform the stacking trajectory
_vel\_traj.py_ | vel | _angular\_velocity\_joints.csv_ | Perform the velocity control path

Note that the full command is:

ros2 run python\_controllers *,

and the file name is read from the place that command is started from. 

All but the last work using the position launch scripts, while the last requires the velocity launch script.

Note that _ros\_ws/src/python\_controllers/setup.py_ has been modified to add these as well.

## Generating results

To obtain the trajectory csv files for the task 2.1.2 points, triangle, figure eight, pick and place, and stacking, one can simply run the _robot\_*.py_ files in _api/_. This then saves the csv file which is read by the controller nodes.

To obtain the csv file for the angular velocities (task 3.3), one can run ***.

To obtain the workspace plot, one can run ***.

The Sympy notebook for the symbolic derivations can be found in _tasks/symbolics.ipynb_.