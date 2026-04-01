import rclpy
import numpy as np
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import csv
import os


class ExampleTraj(Node):

    def __init__(self):
        super().__init__('example_trajectory')

        self._HOME = [np.deg2rad(0), np.deg2rad(70),
                      np.deg2rad(-40), np.deg2rad(-60),
                      np.deg2rad(0)]

        self._START = [0.66532326, -0.22961752, -0.22210515, -0.07535317, 0.2]

        # Load CSV
        csv_path = os.path.expanduser('angular_velocity_joints.csv')
        self._velocities = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self._velocities.append([float(v) for v in row])
        
        self._dt = 0.04
        self._home_speed = 0.3  # rad/s, cap for homing moves

        # Start estimated position at home
        self._current_pos = list(self._HOME)

        self._index = 0
        # States: 'to_start' -> 'trajectory' -> 'to_home' -> 'done'
        self._state = 'to_start'
        self._brake_ticks = 0
        self._BRAKE_DURATION = 2.0  # 10 ticks * 0.04s = 0.4s of zero velocity


        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        self._timer = self.create_timer(self._dt, self.timer_callback)
        self.get_logger().info('Phase 1: Moving to start position...')

    def _proportional_move(self, target):
        """Returns (velocities, at_target). Proportional controller towards target."""
        errors = [target[i] - self._current_pos[i] for i in range(5)]
        max_err = max(abs(e) for e in errors)
        #self.get_logger().info(f"Max error: {max_err}")
        

        if max_err < 0.001:
            return [0.0] * 5, True

        vels = []
        for i in range(5):
            v = np.clip(errors[i] * 2.0, -self._home_speed, self._home_speed)
            vels.append(v)
        return vels, False

    def _update_pos(self, vels):
        for i in range(5):
            self._current_pos[i] += vels[i] * self._dt

    def _publish(self, vels):
        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()
        point = JointTrajectoryPoint()
        point.velocities = vels
        msg.points = [point]
        self._publisher.publish(msg)

    def timer_callback(self):
            if self._state == 'to_start':
                vels, arrived = self._proportional_move(self._START)
                self._publish(vels)
                self._update_pos(vels)
                if arrived:
                    self.get_logger().info('Start position reached. Phase 2: Executing trajectory...')
                    self._state = 'braking1'
                    self._brake_ticks = 0 # Reset brake ticks before starting trajectory
            
            elif self._state == 'braking1':
                self._publish([0.0] * 5) # Send zero velocity
                self._brake_ticks += 1
                
                # Calculate how many ticks are needed (e.g., 0.4s / 0.04s = 10 ticks)
                max_ticks = int(self._BRAKE_DURATION / self._dt)
                
                if self._brake_ticks >= max_ticks:
                    self._state =  'trajectory'

            elif self._state == 'trajectory':
                # 1. Check if we've reached the end of the list
                if self._index >= len(self._velocities):
                    self.get_logger().info('CSV trajectory complete. Phase 3: Returning home...')
                    self._publish([0.0] * 5)
                    self._state = 'braking2' # Fixed: added underscore to match elif
                    self._brake_ticks = 0 # Reset brake ticks before homing
                    return # CRITICAL: Stop execution of this callback here

                # 2. Get the current row
                row = self._velocities[self._index]
                self._publish(row)

                # 3. Update internal position estimate
                for i in range(5):
                    self._current_pos[i] += row[i] * self._dt

                self._index += 1

            elif self._state == 'braking2':
                self._publish([0.0] * 5) # Send zero velocity
                self._brake_ticks += 1
                
                # Calculate how many ticks are needed (e.g., 0.4s / 0.04s = 10 ticks)
                max_ticks = int(self._BRAKE_DURATION / self._dt)
                
                if self._brake_ticks >= max_ticks:
                    self._state =  'to_home'


            elif self._state == 'to_home':
                vels, arrived = self._proportional_move(self._HOME)
                self._publish(vels)
                self._update_pos(vels)
                if arrived:
                    self.get_logger().info('Home reached. Done.')
                    self._publish([0.0] * 5)
                    self._state = 'done'
                    self._timer.cancel() # Now it is safe to stop the timer

def main(args=None):
    rclpy.init(args=args)
    example_traj = ExampleTraj()
    rclpy.spin(example_traj)
    example_traj.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()