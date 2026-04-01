import numpy as np
from itertools import product
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


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

def wrap_angle(angle):
    # Wraps an angle to the range [-pi, pi]
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_ik_jacobian(data, target_position, target_rotation, initial_joint_angles, learning_rate, max_iterations, delta,weight_position, weight_rotation,joint_limits_array, threshold):
    joint_angles = np.copy(initial_joint_angles)
    
    precomputed = precompute_transformations(data)
    
    for iteration in range(max_iterations):
        # 1. Forward Kinematics
        T_world_to_jaw = compute_fast_forward_kinematics(*precomputed, joint_angles)
        current_position = T_world_to_jaw[:3, 3]
        current_rotation = T_world_to_jaw[:3, :3]
        current_euler = rotation_matrix_to_euler_angles(current_rotation, target_rotation)

        error_pos = target_position - current_position
        error_rot = wrap_angle(target_rotation - current_euler)


        error = np.hstack((error_pos * weight_position, error_rot * weight_rotation)) 
        error_norm = np.linalg.norm(error)
        if error_norm < threshold:
            if not all(lo <= angle <= hi for angle, (lo, hi) in zip(joint_angles, joint_limits_array)):
                print(f"Joint angles out of limits, stopping iteration.")
                return joint_angles, False
            return joint_angles, True
        if iteration % 100000 == 0 and iteration >1:
            print(error_pos,error_rot)

        jacobian = np.zeros((6, 5))
        for i in range(5):
            perturbed_angles = np.copy(joint_angles)
            perturbed_angles[i] += delta
            
            T_perturbed = compute_fast_forward_kinematics(*precomputed, perturbed_angles)
            pos_perturbed = T_perturbed[:3, 3]
            euler_perturbed = rotation_matrix_to_euler_angles(T_perturbed[:3, :3], target_rotation)
            
            j_pos = (pos_perturbed - current_position) / delta
            j_rot = wrap_angle(euler_perturbed - current_euler) / delta

            jacobian[:3, i] = j_pos * weight_position
            jacobian[3:, i] = j_rot * weight_rotation

        jacobian_inv = np.linalg.pinv(jacobian, rcond=1e-3) 
        
        step = learning_rate * (jacobian_inv @ error)
        joint_angles = np.clip(joint_angles + step, joint_limits_array[:, 0], joint_limits_array[:, 1])
 

    return joint_angles, False

def compute_end_effector_state(T_world_to_base, T_base_to_shoulder, T_shoulder_to_upper_arm, T_upper_arm_to_lower_arm, T_lower_arm_to_wrist, T_wrist_to_gripper, T_gripper_to_gripper_center, T_gripper_center_to_jaw, joint_angles,target_rotation):
    T_world_to_jaw = compute_fast_forward_kinematics(T_world_to_base, T_base_to_shoulder, T_shoulder_to_upper_arm, T_upper_arm_to_lower_arm, T_lower_arm_to_wrist, T_wrist_to_gripper, T_gripper_to_gripper_center, T_gripper_center_to_jaw, joint_angles)
    position = T_world_to_jaw[:3, 3]
    rotation = T_world_to_jaw[:3, :3]
    euler1= rotation_matrix_to_euler_angles(rotation, target_rotation)
    return position, euler1



def compute_ik_scipy(target_position_single, target_rotation_single, precomputed, joint_limits_array, initial_joint_angle, weight_position, weight_rotation, threshold):
    
    bounds = [(-1.998, 2.14), (-2.002, 1.879), (-1.642, 1.695), (-1.77328, 1.81), (-2.9206, 2.9529)]
    
    best_result = None
    best_cost = np.inf
    
    for i in range(50):
        x0 = initial_joint_angle if i == 0 else np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
        result = minimize(
            compute_cost_5_constraints, x0=x0,
            args=(target_position_single, target_rotation_single, weight_position, weight_rotation, precomputed),
            bounds=bounds, method='L-BFGS-B',
            options={'maxiter': 50000, 'ftol': 1e-16, 'gtol': 1e-12}
        )
        
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result

        if best_cost < threshold:
            condition = True

            break
        if not all(lo <= angle <= hi for angle, (lo, hi) in zip(best_result.x, joint_limits_array)):
                #print(f"Joint angles out of limits, stopping iteration for {i}th target state. Joint angles: {best_result.x}")
                return best_result.x, False
        else:
            condition = False

    return best_result.x, condition


def multiple_convergence_try(target_position_single, target_rotation_single, data, delta, initial_joint_angles, weight_position, weight_rotation, joint_limits_array, learning_rate, max_iterations, threshold):
    final_joint_angle = np.zeros((5,5),dtype=float) 
    condition = np.zeros(5,dtype=float)


    final_joint_angle,condition = compute_ik_jacobian(data, target_position_single, target_rotation_single, initial_joint_angles, learning_rate, max_iterations, delta, weight_position, weight_rotation, joint_limits_array, threshold)

    if condition == False:

        initial_joint_angle = initial_joint_angles
        final_joint_angle, condition = compute_ik_scipy(target_position_single, target_rotation_single, precomputed, joint_limits_array, initial_joint_angle, weight_position, weight_rotation, threshold)

    return final_joint_angle, condition

def compute_cost_5_constraints(joint_angles , target_position , target_rotation, weight_position, weight_rotation, precomputed):
    #target_rotation = [0,target_rotation[0],target_rotation[1]]  # Only consider x and y rotation for cost
    position, euler_angles1 = compute_end_effector_state(*precomputed, joint_angles, target_rotation)
    
    error_position = np.linalg.norm(position - target_position)
    error_rotation_1 = np.linalg.norm(euler_angles1[1:3] - target_rotation[1:3])
    cost_1 = weight_position * error_position + weight_rotation * error_rotation_1
    return cost_1


def multiple_scipy_try(target_pos, target_rot, joint_limits, precomputed, n_guesses):
    # 1. Define a set of diverse starting points
    # Mix of specific poses and random samples
    seeds = [
        np.array([0, 0, 0, 0, 0]),                    # Neutral
        np.array([0.0, 0.8, 0.8, 0, 0]),              # Reach out (Elbow Down)
        np.array([0.0, -0.8, -0.8, 0, 0]),            # Reach out (Elbow Up)
        np.array([1.5, 0.5, 0.5, 0, 0]),              # Side reach Right
        np.array([-1.5, 0.5, 0.5, 0, 0]),             # Side reach Left
    ]
    for _ in range(n_guesses - len(seeds)):
        seeds.append(np.array([np.random.uniform(l, h) for l, h in joint_limits]))
    
    #target_rot = target_rot[1:]  # Ensure it's just the rotation part
    valid_solutions = []

    for x0 in seeds:
        # Run your existing minimize or Jacobian function
        #print(f"Attempt with seed:", x0)
        result = minimize(
            compute_cost_5_constraints, 
            x0=x0,
            args=(target_pos, target_rot, 10, 1, precomputed),
            bounds=joint_limits,
            method='L-BFGS-B'
        )

        if result.success and result.fun < 1e-1:
            # Check if this solution is unique (not just a tiny variation of a previous one)
            is_new = True
            #print(f"Found solution with cost {result.fun:.6f}: {result.x}")
            for sol in valid_solutions:
                #print(valid_solutions)
                if np.allclose(result.x, sol, atol=0.1): # 0.1 rad tolerance
                    is_new = False
                    break
            
            if is_new:
                valid_solutions.append(result.x)

    return valid_solutions

if __name__ == "__main__":
    # Define joint angles for the robot (in radians). convention is for z axis. 
    joints = ['shoulder', 'upper_arm', 'lower_arm', 'wrist','gripper']
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

    joint_limits = {
        'shoulder': (-1.998, 2.14),
        'upper_arm': (-2.002, 1.879),
        'lower_arm': (-1.642, 1.695),
        'wrist': (-1.77328, 1.81),
        'gripper': (-2.9206, 2.9529)
    }

    joint_limits_array = np.array([joint_limits[joint] for joint in joints])
    
    target_state = np.array([[0.2, 0.2, 0.2, 0.0, 1.57, 0.65],
                               [0.2, 0.1, 0.4, 0.0, 0.0, 1.57],
                               [0.0 , 0.0 , 0.4 , 0.0, -0.785, 1.57],
                               #[0.0 , 0.0,0.07, 3.141, 0.0, 0.0],
                               [0.0, 0.0452, 0.45, -0.785, 0.0, 3.141]])


    precomputed = precompute_transformations(data)

    pose_numerical = ['I', 'II', 'III', 'IV-b']
    valid_poses = ['I', 'II', 'III', 'IV-b']
    target_position = target_state[:,:3]
    target_rotation = target_state[:,3:]

    weight_position = 10 # Increased to prioritize position accuracy more strongly. 
    weight_rotation = 1 # Kept the same to maintain some emphasis on orientation, but position is now the dominant factor in the cost function.
    learning_rate = 1e-1 
    max_iterations = 40000 # Increased to 15000 for better convergence
    threshold = 1e-2 # Relaxed threshold to allow for more solutions.
    delta = 1e-5 # Reduced delta for more accurate Jacobian estimation, but be cautious of numerical stability.

    initial_joint_angles = np.array([
        [0, 0, 0, 0, 0],
        [0.0, 0.5,0.5, 0, 0],
        [0.0,-1.2,1.2,0.2,0.2],
        [1.5, 0.2, 0.2, 0, 0],
        [-1.5, 0.2, -0.2, 0, 0],
        [0, 1.5, -1.5, 0, 0]      
                                    ])

    final_joint_angle_all = np.zeros(( target_state.shape[0],initial_joint_angles.shape[0], 5), dtype=float)
    condition_all = np.zeros(( target_state.shape[0],initial_joint_angles.shape[0], 5), dtype=bool)
    all_solutions = []


    open('joint_angles_multiple.csv', 'w').close()
    
    print("Now trying with Jacobian + Scipy for the same target states to find more solutions if possible.")

    for j in range(target_state.shape[0]):
        print(f"Try using Jacobian + Scipy for target pose {pose_numerical[j]}.")
        for i in range(initial_joint_angles.shape[0]):
            final_joint_angle, condition = multiple_convergence_try(target_position[j], target_rotation[j], data, delta, initial_joint_angles[i], weight_position, weight_rotation, joint_limits_array, learning_rate, max_iterations, threshold)
            final_joint_angle_all[j][i] = final_joint_angle
            condition_all[j][i] = condition
        

        for i in range(initial_joint_angles.shape[0]):
            row_to_save = np.array(final_joint_angle_all[j][i]).reshape(1, -1)
            prev_row = np.array(final_joint_angle_all[j][i-1]).reshape(1, -1) if i > 0 else None
            change = np.linalg.norm(row_to_save - prev_row) if prev_row is not None else 1
            all_solutions.append(final_joint_angle_all[j][i])
            if change > 1e-2:
                #print(f"Valid solution for initial joint angles {initial_joint_angles[i]}: {final_joint_angle_all[j][i]}")
                with open('joint_angles_multiple.csv', 'a') as f:
                    np.savetxt(f, row_to_save, delimiter=',')
        print(len(all_solutions[j]), "valid solutions found for target state", valid_poses[j])
        print("--------------------------------------------------")
        print('Saved to joint_angles_multiple.csv')

