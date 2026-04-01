import rclpy
from rclpy.node import Node

import numpy as np
from numpy.typing import NDArray

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class RobotTraj(Node):
    def __init__(self, duration: float = 1.0) -> None:
        super().__init__('robot_mover')

        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)

        self._joint_trajectory = np.genfromtxt('stack_path.csv', delimiter=',')

        self._total_duration: float = duration
        self._num_segments: int = len(self._joint_trajectory) - 1
        self._time_per_segment: float = self._total_duration / self._num_segments

        self._run_with_confirmation()

    def _run_with_confirmation(self) -> None:
        self.get_logger().info("Moving to start")
        self._send_single_point(self._joint_trajectory[0])

        user_input = input("Press enter to start trajectory...")

        if user_input.lower() in ['']:
            self._begin_trajectory()
        else:
            self.destroy_node()
            rclpy.shutdown()

    def _begin_trajectory(self) -> None:
        self._start_time = self.get_clock().now()
        self._timer = self.create_timer(0.04, self._timer_callback)
        self.get_logger().info("Executing trajectory...")

    def _timer_callback(self) -> None:
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

        t_normalized: float = (elapsed_seconds % self._time_per_segment) / self._time_per_segment

        move_fraction = 0.5
        if t_normalized < move_fraction:
            t_interp = t_normalized / move_fraction
            current_q = q_start + (q_end - q_start) * t_interp
        else:
            current_q = q_end

        self._send_single_point(current_q)

    def _send_single_point(self, joint_values: NDArray[np.float64]) -> None:
        msg = JointTrajectory()
        point = JointTrajectoryPoint()
        point.positions = joint_values.tolist()
        msg.points = [point]
        self._publisher.publish(msg)

def main(args=None) -> None:
    rclpy.init(args=args)

    mover = RobotTraj(duration=45.0)

    if rclpy.ok():
        rclpy.spin(mover)

    mover.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()