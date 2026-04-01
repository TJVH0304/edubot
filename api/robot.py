'''
Custom Robot class to hold forward kinematics, inverse kinematics, and helpers like the numerical jacobian in one place.

Initialize using robot = Robot()
Set joint limits using robot.set_lim(LIMITS)

Default joint limits (unconstrained, simulation constrained, hardware constrained) are also defined
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.optimize import minimize

import time

from transform import Transformation


class Robot:
    def __init__(self):
        self.world_to_base = Transformation.fromEuler(roll=0, pitch=0, yaw=3.14159, T=[0, 0, 0])
        self.base_to_shoulder = Transformation.fromEuler(roll=0, pitch=0, yaw=0, T=[0, -0.0452, 0.0165])
        self.shoulder_rotation = Transformation.identity()  # empty as no q initialized
        self.shoulder_to_upper_arm = Transformation.fromEuler(roll=0, pitch=-1.57079, yaw=0, T=[0, -0.0306, 0.1025])
        self.upper_arm_rotation = Transformation.identity()  # empty as no q initialized
        self.upper_arm_to_lower_arm = Transformation.fromEuler(roll=0, pitch=0, yaw=0, T=[0.11257, -0.028, 0])
        self.lower_arm_rotation = Transformation.identity()  # empty as no q initialized
        self.lower_arm_to_wrist = Transformation.fromEuler(roll=0, pitch=0, yaw=1.57079, T=[0.0052, -0.1349, 0])
        self.wrist_rotation = Transformation.identity()  # empty as no q initialized
        self.wrist_to_gripper = Transformation.fromEuler(roll=0, pitch=-1.57079, yaw=0, T=[-0.0601, 0, 0])
        self.gripper_rotation = Transformation.identity()  # empty as no q initialized
        self.gripper_to_gripper_center = Transformation.fromEuler(roll=0, pitch=0, yaw=0, T=[0, 0, 0.075])
        self.gripper_to_jaw = Transformation.fromEuler(roll=1.57079, pitch=3.14159, yaw=0, T=[-0.0202, 0, 0.0244])
        self.jaw_rotation = Transformation.identity()  # empty as no q initialized

        self.base_transforms = (
            self.world_to_base,
            self.base_to_shoulder,
            self.shoulder_to_upper_arm,
            self.upper_arm_to_lower_arm,
            self.lower_arm_to_wrist,
            self.wrist_to_gripper,
            self.gripper_to_gripper_center,
            self.gripper_to_jaw
        )

        self.joint_limits = np.array([])

        self.gripper_close = 0.1
        self.gripper_open = 0.4

    def set_lim(self, lim):
        self.joint_limits = lim

    def _apply_q(self, q):
        # all joints rotate about their local z axis, thus the joint input is a yaw angle
        self.shoulder_rotation = Transformation.fromEuler(yaw=q[0])
        self.upper_arm_rotation = Transformation.fromEuler(yaw=q[1])
        self.lower_arm_rotation = Transformation.fromEuler(yaw=q[2])
        self.wrist_rotation = Transformation.fromEuler(yaw=q[3])
        self.gripper_rotation = Transformation.fromEuler(yaw=q[4])
        self.jaw_rotation = Transformation.fromEuler(yaw=q[5])

    def FK(self, q):
        """
        The forward kinematics of the robot
        :param q: (6,) numpy array of the joint angles [rad]
        :return: (3,) numpy array of the end effector position
        """
        self._apply_q(q)

        T = (
                self.world_to_base
                @ self.base_to_shoulder
                @ self.shoulder_rotation
                @ self.shoulder_to_upper_arm
                @ self.upper_arm_rotation
                @ self.upper_arm_to_lower_arm
                @ self.lower_arm_rotation
                @ self.lower_arm_to_wrist
                @ self.wrist_rotation
                @ self.wrist_to_gripper
                @ self.gripper_rotation
                @ self.gripper_to_gripper_center
        )

        return T

    def _clamp_joints(self, q):
        if self.joint_limits.shape != (6,2):
            return q

        q_clamped = np.array(q).copy()
        for i, (qmin, qmax) in enumerate(self.joint_limits):
            q_clamped[i] = np.clip(q_clamped[i], qmin, qmax)

        return q_clamped

    @staticmethod
    def _rotation_matrix_to_axis_angle_error(R_current, R_target):
        R_err = R_target @ R_current.T
        error = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])
        return error

    @staticmethod
    def _orientation_error_tool_axis_only(R_current, R_target, axis='z'):
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        idx = axis_map[axis]

        a_current = R_current[:, idx]
        a_target = R_target[:, idx]

        return np.cross(a_current, a_target)

    def _numerical_jacobian(self, q, eps=1e-6, orientation_mode='full', axis='z'):
        q = np.array(q, dtype=np.float64)

        n = len(q)

        T0 = self.FK(q)
        p0 = T0.t()
        R0 = T0.R()

        J = np.zeros((6, n), dtype=np.float64)

        for i in range(n):
            q_perturbed = q.copy()
            q_perturbed[i] += eps

            T1 = self.FK(q_perturbed)
            p1 = T1.t()
            R1 = T1.R()

            dp = (p1 - p0) / eps

            if orientation_mode == 'full':
                domega = self._rotation_matrix_to_axis_angle_error(R0, R1) / eps
            elif orientation_mode == 'axis':
                e0 = self._orientation_error_tool_axis_only(R0, R0, axis=axis)
                e1 = self._orientation_error_tool_axis_only(R0, R1, axis=axis)

                domega = (e1 - e0) / eps

            else:
                raise ValueError('invalid orientation mode')

            J[:3, i] = dp
            J[3:, i] = domega

        return J

    def IK(self, target, q0=None, max_iter=200, weights=(1, 0.1), damping=1e-3, step=0.5, tol=(1e-3, 1e-2),
           orientation_mode='axis', tool_axis='z', joint_center_weight=0):
        target_pos = target.t()
        target_rot = target.R()

        if q0 is None:
            q0 = np.zeros(6)
            q = np.zeros(6)
        else:
            q = np.zeros(6)

        q = self._clamp_joints(q)
        n = len(q)

        q_center = np.zeros(6)

        last_pos_error_norm = None
        last_rot_error_norm = None

        for it in range(max_iter):
            T = self.FK(q)
            p = T.t()
            R = T.R()

            pos_err = target_pos - p
            if orientation_mode == 'full':
                rot_err = self._rotation_matrix_to_axis_angle_error(R, target_rot)
            elif orientation_mode == 'axis':
                rot_err = self._orientation_error_tool_axis_only(R, target_rot, axis=tool_axis)
            else:
                raise ValueError('invalid orientation mode')

            last_pos_error_norm = np.linalg.norm(pos_err)
            last_rot_error_norm = np.linalg.norm(rot_err)

            pos_ok = last_pos_error_norm < tol[0]
            rot_ok = last_rot_error_norm < tol[1]

            if pos_ok and rot_ok:
                return q, True, {
                    'iterations': it,
                    'pos_err_norm': last_pos_error_norm,
                    'rot_err_norm': last_rot_error_norm
                }

            task_err = np.hstack((pos_err, rot_err))

            W = np.diag([
                weights[0], weights[0], weights[0],
                weights[1], weights[1], weights[1]
            ])

            J = self._numerical_jacobian(q, orientation_mode=orientation_mode, axis=tool_axis)

            A = J.T @ W @ J + (damping ** 2) * np.eye(n)
            b = J.T @ W @ task_err

            if joint_center_weight > 0:
                b += joint_center_weight * (q_center - q)

            try:
                dq = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(A) @ b

            q = q + step * dq
            q = self._clamp_joints(q)


        # if still failed, run this
        best_res = None
        best_cost = np.inf

        for i in range(max_iter):
            x0 = q0 if i == 0 else np.array([np.random.uniform(low, high) for low, high in self.joint_limits])

            res = minimize(
                self.__cost_fn, x0=x0,
                args=(target, weights, orientation_mode, tool_axis),
                bounds=self.joint_limits,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12}
            )

            if res.fun < best_cost:
                best_cost = res.fun
                best_res = res

            if best_cost < tol[0]:
                return best_res.x, True, {
                    'iterations': i,
                    'pos_err_norm': 'calc',
                    'rot_err_norm': 'calc',
                }

        return q, False, {
            'iterations': max_iter,
            'pos_err_norm': last_pos_error_norm,
            'rot_err_norm': last_rot_error_norm
        }

    def __cost_fn(self, q, target, weights, orientation_mode, tool_axis):
        T = self.FK(q)

        p = T.t()
        R = T.R()

        error_pos = np.linalg.norm(p - target.t())
        if orientation_mode == 'full':
            error_rot = self._rotation_matrix_to_axis_angle_error(R, target.R())
        elif orientation_mode == 'axis':
            error_rot = self._orientation_error_tool_axis_only(R, target.R(), axis=tool_axis)
        else:
            raise ValueError('invalid orientation mode')

        error_rot = np.linalg.norm(error_rot)

        return weights[0] * error_pos + weights[1] * error_rot

JOINT_LIMITS_UNCONSTRAINED = np.array([
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [-np.pi, np.pi]
])

JOINT_LIMITS_SIMULATION = np.array([
    [-2, 2],
    [-np.pi / 2, np.pi / 2],
    [-np.pi / 2, np.pi / 2],
    [-np.pi / 2, np.pi / 2],
    [-np.pi, np.pi],
    [-np.pi, np.pi]
])

JOINT_LIMITS_HARDWARE = np.array([
    [-1.998, 2.140],
    [-2.002, 1.879],
    [-1.642, 1.695],
    [-1.77328, 1.81009],
    [-2.9206, 2.9529],
    [0, 2.201]
])
