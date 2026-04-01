import rclpy
from rclpy.node import Node

import numpy as np
from numpy.typing import NDArray

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class RobotMover(Node):
    """
    ROS2 node that moves a robot arm through a sequence of joint-space waypoints.

    Workflow:
      1. Pre-solve IK for all Cartesian waypoints.
      2. Move the robot to the first waypoint and wait for user confirmation.
      3. Execute the full trajectory using a fixed-rate timer.
    """

    def __init__(self, duration: float = 1.0) -> None:
        super().__init__('robot_mover')

        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)

        self._joint_trajectory = np.genfromtxt('path.csv', delimiter=',')

        # if not self._joint_trajectory:
        #     self.get_logger().error("No valid waypoints found!")
        #     return

        self._total_duration: float = duration
        self._num_segments: int = len(self._joint_trajectory) - 1
        self._time_per_segment: float = self._total_duration / self._num_segments

        self._run_with_confirmation()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _run_with_confirmation(self) -> None:
        """
        Move the robot to the start position, then wait for the user to confirm
        before executing the full trajectory.
        """
        self.get_logger().info("Moving to START position...")
        self._send_single_point(self._joint_trajectory[0])

        print("\n" + "=" * 40)
        user_input = input(" Robot at start. Press ENTER to begin trajectory: ")
        print("=" * 40 + "\n")

        if user_input.lower() in ['']:
            self._begin_trajectory()
        else:
            self.get_logger().info("Trajectory cancelled by user.")
            self.destroy_node()
            rclpy.shutdown()

    # ------------------------------------------------------------------
    # Trajectory execution
    # ------------------------------------------------------------------

    def _begin_trajectory(self) -> None:
        """Start the timer that drives the trajectory execution."""
        self._start_time = self.get_clock().now()
        self._timer = self.create_timer(0.04, self._timer_callback)
        self.get_logger().info("Executing trajectory...")

    def _timer_callback(self) -> None:
        """
        Interpolate between waypoints based on elapsed time and publish the
        current joint configuration at 25 Hz.

        Each segment uses a 95 % move / 5 % dwell profile:
          - First 95 % of segment time: linearly interpolate from start to end.
          - Last  5 % of segment time:  hold the end configuration.
        """
        now = self.get_clock().now()
        elapsed_seconds: float = (now - self._start_time).nanoseconds / 1e9
        segment_idx: int = int(elapsed_seconds // self._time_per_segment)

        if segment_idx >= self._num_segments:
            self._send_single_point(self._joint_trajectory[-1])
            self.get_logger().info("Trajectory complete.")
            self._timer.cancel()
            return

        q_start = self._joint_trajectory[segment_idx]
        q_end   = self._joint_trajectory[segment_idx + 1]

        # Normalised time within current segment [0, 1)
        t_normalised: float = (elapsed_seconds % self._time_per_segment) / self._time_per_segment

        move_fraction = 0.95
        if t_normalised < move_fraction:
            t_interp = t_normalised / move_fraction
            current_q = q_start + (q_end - q_start) * t_interp
        else:
            current_q = q_end

        self._send_single_point(current_q)

    def _send_single_point(self, joint_values: NDArray[np.float64]) -> None:
        """
        Publish a single joint configuration to the trajectory topic.
        """
        msg = JointTrajectory()
        point = JointTrajectoryPoint()
        point.positions = joint_values.tolist()
        msg.points = [point]
        self._publisher.publish(msg)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)

    # # --- Constants -----------------------------------------------------------
    # down_direction: list[int] = [0, 0, -1]

    # gripper_open:      float = 0.4
    # gripper_closed:    float = 0.2
    # gripper_more_open: float = 0.55

    # lift_height:  float = 0.05   # metres above the stack before descending
    # feed_x:       float = 0.0
    # feed_y:       float = 0.16
    # stack_y:      float = 0.14

    # total_blocks: int   = 9
    # z_limit:      float = 0.035  # current top of the stack (grows each iteration)

    # # --- Path construction ---------------------------------------------------
    # path: list[Waypoint] = []
    # block_index: int = 0

    # while block_index < total_blocks:

    #     # ---- Pick from feed -------------------------------------------------
    #     if block_index < 2:
    #         pick_gripper   = gripper_open
    #         place_x        = 0.09
    #         place_z_factor = 1.0        # exact z_limit
    #         place_gripper  = gripper_open

    #     elif block_index < 4:
    #         pick_gripper   = gripper_more_open
    #         place_x        = 0.09
    #         place_z_factor = 1.1
    #         place_gripper  = gripper_more_open

    #     elif block_index < 8:
    #         pick_gripper   = gripper_more_open
    #         place_x        = 0.085
    #         place_z_factor = 1.1
    #         place_gripper  = gripper_more_open

    #     else:
    #         pick_gripper   = gripper_more_open
    #         place_x        = 0.085
    #         place_z_factor = 0.99
    #         place_gripper  = gripper_more_open

    #     # Approach feed location
    #     path += build_approach_path(
    #         [feed_x, feed_y, lift_height + z_limit, down_direction, pick_gripper],
    #         z_limit=0.03,
    #         num_points=2,
    #     )

    #     # Grasp at feed
    #     path += [[feed_x, feed_y, 0.03, down_direction, gripper_closed]]

    #     # Retreat from feed (only needed when the stack is already tall)
    #     if z_limit > 0.15:
    #         path += build_approach_path(
    #             [feed_x, feed_y, lift_height + z_limit, down_direction, gripper_closed],
    #             z_limit=0.03,
    #             num_points=2,
    #         )[::-1]

    #     # Approach stack location
    #     place_z = z_limit * place_z_factor
    #     path += build_approach_path(
    #         [place_x, stack_y, lift_height + z_limit, down_direction, gripper_closed],
    #         z_limit=z_limit,
    #         num_points=2,
    #     )

    #     # Place on stack
    #     path += [[place_x, stack_y, place_z, down_direction, place_gripper]]

    #     # Retreat from stack (skip on the very last block)
    #     if block_index != total_blocks - 1:
    #         path += build_approach_path(
    #             [place_x, stack_y, lift_height + place_z, down_direction, place_gripper],
    #             z_limit=z_limit,
    #         )[::-1]

    #     z_limit     += 0.02
    #     block_index += 1

    # --- Execute -------------------------------------------------------------
    mover = RobotMover(duration=5.0)

    if rclpy.ok():
        rclpy.spin(mover)

    mover.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()