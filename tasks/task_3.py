import numpy as np
from itertools import product
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

#np.set_printoptions(precision=6, suppress=True)

def rotation_joint_matrix(angle):
    R_z = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                    [np.sin(angle), np.cos(angle), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    return R_z

def rotation_matrix(angle_vector):
    angle_x = angle_vector[0]
    angle_y = angle_vector[1]
    angle_z = angle_vector[2]
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)]])
    R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    R_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x

def homogenous_transformation_matrix(rotation_vector, translation_vector):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix(rotation_vector) 
    T[:3, 3] = translation_vector
    return T

def precompute_transformations(data):
    T_world_to_base = homogenous_transformation_matrix(data['rotation_world_to_base'], data['translation_world_to_base'])
    T_base_to_shoulder = homogenous_transformation_matrix(data['rotation_base_to_shoulder'], data['translation_base_to_shoulder'])
    T_shoulder_to_upper_arm = homogenous_transformation_matrix(data['rotation_shoulder_to_upper_arm'], data['translation_shoulder_to_upper_arm'])
    T_upper_arm_to_lower_arm = homogenous_transformation_matrix(data['rotation_upper_arm_to_lower_arm'], data['translation_upper_arm_to_lower_arm'])
    T_lower_arm_to_wrist = homogenous_transformation_matrix(data['rotation_lower_arm_to_wrist'], data['translation_lower_arm_to_wrist'])
    T_wrist_to_gripper = homogenous_transformation_matrix(data['rotation_wrist_to_gripper'], data['translation_wrist_to_gripper'])
    T_gripper_to_gripper_center = homogenous_transformation_matrix(data['rotation_gripper_to_gripper_center'], data['translation_gripper_to_gripper_center'])
    T_gripper_center_to_jaw = homogenous_transformation_matrix(data['rotation_gripper_center_to_jaw'], data['translation_gripper_center_to_jaw'])
    return T_world_to_base, T_base_to_shoulder, T_shoulder_to_upper_arm, T_upper_arm_to_lower_arm, T_lower_arm_to_wrist, T_wrist_to_gripper, T_gripper_to_gripper_center, T_gripper_center_to_jaw   

def compute_fast_forward_kinematics(T_world_to_base, T_base_to_shoulder, T_shoulder_to_upper_arm, T_upper_arm_to_lower_arm, T_lower_arm_to_wrist, T_wrist_to_gripper, T_gripper_to_gripper_center, T_gripper_center_to_jaw, joint_angles):
    joint_transformation_shoulder = rotation_joint_matrix(joint_angles[0])
    joint_transformation_upper_arm = rotation_joint_matrix(joint_angles[1])
    joint_transformation_lower_arm = rotation_joint_matrix(joint_angles[2])
    joint_transformation_wrist = rotation_joint_matrix(joint_angles[3])
    joint_transformation_gripper = rotation_joint_matrix(joint_angles[4])

    T_world_to_jaw = T_world_to_base @ T_base_to_shoulder @ joint_transformation_shoulder @ T_shoulder_to_upper_arm @ joint_transformation_upper_arm @ T_upper_arm_to_lower_arm @ joint_transformation_lower_arm @ T_lower_arm_to_wrist @ joint_transformation_wrist @ T_wrist_to_gripper @ joint_transformation_gripper @ T_gripper_to_gripper_center #  @ T_gripper_center_to_jaw
    return T_world_to_jaw

def rotation_matrix_to_euler_angles(R,target_rotation, z = 0):
    y1 = np.arcsin(-R[2, 0])
    y2 = np.pi - y1
    if y1 != np.pi/2 and y1 != -np.pi/2:
        x1 = np.arctan2(R[2, 1] / np.cos(y1), R[2, 2]/np.cos(y1))
        z1 = np.arctan2(R[1, 0] / np.cos(y1), R[0, 0]/np.cos(y1))
    elif y1 == np.pi/2:
        z1 = z
        x1 = z1 + np.atan2(R[0,1],R[0,2])
    else:
        z1 = z
        x1 = -z1 + np.atan2(-R[0,1],-R[0,2])
    if y2 != np.pi/2 and y2 != -np.pi/2:
        x2 = np.arctan2(R[2, 1] / np.cos(y2), R[2, 2]/np.cos(y2))
        z2 = np.arctan2(R[1, 0] / np.cos(y2), R[0, 0]/np.cos(y2))
    elif y2 == np.pi/2:
        z2 = z
        x2 = z2 + np.atan2(R[0,1],R[0,2])
    else:
        z2 = z
        x2 = -z2 + np.atan2(-R[0,1],-R[0,2])
    euler1 = np.array([x1, y1, z1])
    euler2 = np.array([x2, y2, z2])
    if np.linalg.norm(euler1 - target_rotation) <= np.linalg.norm(euler2 - target_rotation):
        return euler1
    else:
        return euler2

def jacobian_position_error(T_world_to_jaw, precomputed,joint_angles,delta):
    position = T_world_to_jaw[:3, 3]
    jacobian = np.zeros((3,5))

    for i in range(len(joint_angles)):
        perturbed_angles = np.copy(joint_angles)   
        perturbed_angles[i] += delta
        T_world_to_jaw_perturbed = compute_fast_forward_kinematics(*precomputed, perturbed_angles)
        position_perturbed = T_world_to_jaw_perturbed[:3, 3]
        jacobian[:, i] = (position_perturbed - position) / delta
    return jacobian

def wrap_angle(angle):
    # Wraps an angle to the range [-pi, pi]
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_symbolic_jacobian(joint_angles):
    q1, q2, q3, q4, q5 = joint_angles
    
    J = np.array([[(0.11257*np.sin(q2) + 0.0052*np.sin(q2 + q3) - 0.028*np.cos(q2) - 0.1349*np.cos(q2 + q3) - 0.1351*np.cos(q2 + q3 + q4) - 0.0306)*np.cos(q1), (0.028*np.sin(q2) + 0.1349*np.sin(q2 + q3) + 0.1351*np.sin(q2 + q3 + q4) + 0.11257*np.cos(q2) + 0.0052*np.cos(q2 + q3))*np.sin(q1), (0.1349*np.sin(q2 + q3) + 0.1351*np.sin(q2 + q3 + q4) + 0.0052*np.cos(q2 + q3))*np.sin(q1), 0.1351*np.sin(q1)*np.sin(q2 + q3 + q4), 0], [(0.11257*np.sin(q2) + 0.0052*np.sin(q2 + q3) - 0.028*np.cos(q2) - 0.1349*np.cos(q2 + q3) - 0.1351*np.cos(q2 + q3 + q4) - 0.0306)*np.sin(q1), -(0.028*np.sin(q2) + 0.1349*np.sin(q2 + q3) + 0.1351*np.sin(q2 + q3 + q4) + 0.11257*np.cos(q2) + 0.0052*np.cos(q2 + q3))*np.cos(q1), -(0.1349*np.sin(q2 + q3) + 0.1351*np.sin(q2 + q3 + q4) + 0.0052*np.cos(q2 + q3))*np.cos(q1), -0.1351*np.sin(q2 + q3 + q4)*np.cos(q1), 0], [0, -0.11257*np.sin(q2) - 0.0052*np.sin(q2 + q3) + 0.028*np.cos(q2) + 0.1349*np.cos(q2 + q3) + 0.1351*np.cos(q2 + q3 + q4), -0.0052*np.sin(q2 + q3) + 0.1349*np.cos(q2 + q3) + 0.1351*np.cos(q2 + q3 + q4), 0.1351*np.cos(q2 + q3 + q4), 0], [0, np.cos(q1), np.cos(q1), np.cos(q1), -np.sin(q1)*np.cos(q2 + q3 + q4)], [0, np.sin(q1), np.sin(q1), np.sin(q1), np.cos(q1)*np.cos(q2 + q3 + q4)], [1, 0, 0, 0, np.sin(q2 + q3 + q4)]])
    return J

def compute_inverse_kinematics_position(i,final_joint_angles_position, initial_joint_angles, target_position, precomputed, learning_rate, max_iterations, delta, condition):
    joint_angles = np.copy(initial_joint_angles)
    joint_angles = np.array([0.2,0.2,0,0,0.2])
    for iteration in range(max_iterations):
        T_world_to_jaw = compute_fast_forward_kinematics(*precomputed, joint_angles)
        current_position = T_world_to_jaw[:3, 3]
        error = target_position - current_position

        if iteration % 1000 == 0:
            #print(f"Iteration: {iteration}, Error: {error}, Joint Angles: {joint_angles}")
            pass
        if np.linalg.norm(error) < 1e-4:
            #print(f"Converged in {iteration} iterations.")
            #print(f"Final joint angles (radians): {joint_angles} for target position: {target_position}, Error: {error}")

            final_joint_angles_position[i] = joint_angles
            condition.append('converged')
            return joint_angles, final_joint_angles_position, condition
        
        jacobian = get_symbolic_jacobian(joint_angles=joint_angles)[:3, :]
        jacobian_inv = np.linalg.pinv(jacobian)

        joint_angles += learning_rate * (jacobian_inv @ error)
    
    print("Did not converge.")

    final_joint_angles_position[i] = joint_angles
    condition.append('not_converged')
    return joint_angles, final_joint_angles_position, condition

def compute_end_effector_state(T_world_to_base, T_base_to_shoulder, T_shoulder_to_upper_arm, T_upper_arm_to_lower_arm, T_lower_arm_to_wrist, T_wrist_to_gripper, T_gripper_to_gripper_center, T_gripper_center_to_jaw, joint_angles,target_rotation):
    T_world_to_jaw = compute_fast_forward_kinematics(T_world_to_base, T_base_to_shoulder, T_shoulder_to_upper_arm, T_upper_arm_to_lower_arm, T_lower_arm_to_wrist, T_wrist_to_gripper, T_gripper_to_gripper_center, T_gripper_center_to_jaw, joint_angles)
    position = T_world_to_jaw[:3, 3]
    rotation = T_world_to_jaw[:3, :3]
    euler1= rotation_matrix_to_euler_angles(rotation, target_rotation)
    return position, euler1

def triangle_path(points_to_make, y_plane):
    x_points_path1 = np.linspace(-0.2,0, points_to_make)
    z_points_path1 = x_points_path1+0.3
    x_points_path2 = np.linspace(0,0.2, points_to_make)
    z_points_path2 = -x_points_path2+0.3
    x_points_path3 = np.linspace(0.2,-0.2, points_to_make)
    z_points_path3 = np.zeros(x_points_path3.shape[0]) + 0.1

    y_points_path1 = y_plane * np.ones(x_points_path1.shape[0])
    y_points_path2 = y_plane * np.ones(x_points_path2.shape[0])
    y_points_path3 = y_plane * np.ones(x_points_path3.shape[0])

    path1 = np.column_stack((x_points_path1, y_points_path1, z_points_path1))  # (10, 3)
    path2 = np.column_stack((x_points_path2, y_points_path2, z_points_path2))  # (10, 3)
    path3 = np.column_stack((x_points_path3, y_points_path3, z_points_path3))  # (10, 3)

    return np.vstack((path1, path2, path3))  # (30, 3)

def triangle_ik_joints(target_triangle_state, precomputed, learning_rate, max_iterations, delta):
    final_joint_angles_position = np.zeros((target_triangle_state.shape[0],5),dtype=float)
    initial_joint_angles = np.zeros(5,dtype=float)
    condition = []
    for i in range(target_triangle_state.shape[0]):
        target_position = target_triangle_state[i, :]
        joint_angle,final_joint_angles_position, condition =  compute_inverse_kinematics_position(i,final_joint_angles_position, initial_joint_angles, target_position, precomputed, learning_rate, max_iterations, delta, condition)
        initial_joint_angles = joint_angle.copy()

    np.savetxt('final_joint_angles_position.csv', final_joint_angles_position, delimiter=',')
    return final_joint_angles_position, condition

def propagating_velocity(linear_velocity_profile, precomputed, start_position_cartesian, propagation_time, learning_rate, max_iterations, delta):
    initial_joint_angles =np.array([0.2,0.2,0,0,0.2])
    condition = []
    final_joint_angles_position = np.zeros((1,5),dtype=float)

    current_position = start_position_cartesian


    position_list = np.zeros((max_iterations, 3))

    joint_angles = initial_joint_angles
    angular_velocity_joints = np.zeros((max_iterations, 5))
    final_joint_angles =  np.zeros((max_iterations, 5))
    distance_list = np.zeros(max_iterations)


    joint_angles, _ , _ =  compute_inverse_kinematics_position(0 , final_joint_angles_position , joint_angles , current_position, precomputed, learning_rate, max_iterations, delta, condition)

    for iteration in range(max_iterations):
        # Forward kinematics with current joint angles
        T_world_to_jaw = compute_fast_forward_kinematics(*precomputed, joint_angles)
        current_position = T_world_to_jaw[:3, 3]  # update position first

 

        distance = np.linalg.norm(current_position - start_position_cartesian)
        position_list[iteration] = current_position
        distance_list[iteration] = distance
        if distance > 0.1:
            print(f"Converged in {iteration} iterations.")
            position_list = position_list[:iteration]
            distance_list = distance_list[:iteration]
            angular_velocity_joints = angular_velocity_joints[:iteration]
            final_joint_angles = final_joint_angles[:iteration]
            return angular_velocity_joints, distance_list, position_list, final_joint_angles

        jacobian = get_symbolic_jacobian(joint_angles=joint_angles)[:3, :]
        jacobian_inv = np.linalg.pinv(jacobian)
        current_angular_velocity = jacobian_inv @ linear_velocity_profile

        angular_velocity_joints[iteration] = current_angular_velocity

        final_joint_angles[iteration] = joint_angles
        joint_angles = joint_angles + current_angular_velocity * propagation_time
        
    return angular_velocity_joints, distance_list, position_list, final_joint_angles


if __name__ == "__main__":

    data = {
        'translation_world_to_base': np.array([0, 0, 0]),
        'rotation_world_to_base': np.array([0,0,3.14159]),
        'translation_base_to_shoulder': np.array([0, -0.0452, 0.0165]),
        'rotation_base_to_shoulder': np.array([0, 0, 0]),
        'translation_shoulder_to_upper_arm': np.array([0, -0.0306, 0.1025]),
        'rotation_shoulder_to_upper_arm': np.array([0, -1.57079, 0]),
        'translation_upper_arm_to_lower_arm': np.array([0.11257, -0.028, 0]),
        'rotation_upper_arm_to_lower_arm': np.array([0, 0, 0]),
        'translation_lower_arm_to_wrist': np.array([0.0052, -0.1349, 0]),
        'rotation_lower_arm_to_wrist': np.array([0, 0, 1.57079]),
        'translation_wrist_to_gripper': np.array([-0.0601, 0, 0]),
        'rotation_wrist_to_gripper': np.array([0, -1.57079, 0]),
        'translation_gripper_to_gripper_center': np.array([0, 0, 0.075]),
        'rotation_gripper_to_gripper_center': np.array([0, 0, 0]),
        'translation_gripper_center_to_jaw': np.array([-0.0202, 0, 0.0244]),
        'rotation_gripper_center_to_jaw': np.array([1.57079, 0, 0]),
    }

    plot= False #SET TRUE IF 3d PLOT NEEDED TO VISUALIZE END EFFECTOR PATH

    precomputed = precompute_transformations(data)

    target_triangle_state = triangle_path(10, 0.3)
    learning_rate = 1e-1
    max_iterations = 1000000
    delta = 1e-4


    propagation_time = 0.04

    linear_velocity_profile = np.array([0.02, 0, 0])

    start_position_cartesian = target_triangle_state[0]

    angular_velocity_joints, distance_list, position_list, final_joint_angles = propagating_velocity(linear_velocity_profile, precomputed, start_position_cartesian, propagation_time, learning_rate, max_iterations, delta)
    np.savetxt('angular_velocity_joints.csv', angular_velocity_joints, delimiter=',')
    position = np.zeros((final_joint_angles.shape[0], 3))

    for i in range(final_joint_angles.shape[0]):
        position[i] = compute_end_effector_state(*precomputed, final_joint_angles[i], target_rotation=np.array([0,0,0]))[0]


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(position[:, 0], position[:, 1], position[:, 2], 'b-', linewidth=2)
    ax.scatter(position[0, 0], position[0, 1], position[0, 2], color='g', s=100, label='Start')
    ax.scatter(position[-1, 0], position[-1, 1], position[-1, 2], color='r', s=100, label='End Cartesian Target')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('End Effector Path check')
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-0.25, 0])
    ax.set_ylim([0.2, 0.45]) # Center around your Y=0.3 position
    ax.set_zlim([0, 0.25])
    ax.legend()
    plt.savefig('end_effector_path.png')
    if plot == True:
        plt.show()
    

