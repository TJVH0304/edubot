import numpy as np
from itertools import product
import pyvista as pv
from scipy.spatial import ConvexHull
pv.OFF_SCREEN = True  # Globally force off-screen
pv.set_plot_theme("document") # Optional: makes backgrounds white for reports

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

def find_workspace(data, joint_limits, max_iterations,condition):
    ranges = [
        np.linspace(joint_limits['shoulder'][0], joint_limits['shoulder'][1], max_iterations),
        np.linspace(joint_limits['upper_arm'][0], joint_limits['upper_arm'][1], max_iterations),
        np.linspace(joint_limits['lower_arm'][0], joint_limits['lower_arm'][1], max_iterations),
        np.linspace(joint_limits['wrist'][0], joint_limits['wrist'][1], max_iterations),
        np.linspace(joint_limits['gripper'][0], joint_limits['gripper'][1], 1)
    ]


    all_possible_configurations = np.array(list(product(*ranges)))

    distance = np.zeros((all_possible_configurations.shape[0],3))


    T_world_to_base, T_base_to_shoulder, T_shoulder_to_upper_arm, T_upper_arm_to_lower_arm, T_lower_arm_to_wrist, T_wrist_to_gripper, T_gripper_to_gripper_center, T_gripper_center_to_jaw = precompute_transformations(data)

    for i in range(all_possible_configurations.shape[0]):
        if i % 20000 == 0:
            print('Current iteration:', i)
        joint_angles = all_possible_configurations[i]
        T_world_to_jaw = compute_fast_forward_kinematics(T_world_to_base, T_base_to_shoulder, T_shoulder_to_upper_arm, T_upper_arm_to_lower_arm, T_lower_arm_to_wrist, T_wrist_to_gripper, T_gripper_to_gripper_center, T_gripper_center_to_jaw,joint_angles)
        distance[i] = T_world_to_jaw[:3,3]
    if condition == True:
        distance = distance[distance[:,0]>=0] # Only consider points with x >= 0 for the workspace.
    return distance

def plot_workspace(distance, plot, target_state, target_name, save):


    save_path_1 = 'workspace_point_cloud.png'
    save_path_2 = 'workspace_convex_hull.png'
    # --- PHASE 1: POINT CLOUD ---
    # Use off_screen=True if we are ONLY saving, False if we want to see it
    plotter1 = pv.Plotter(off_screen=(not plot)) 
    point_cloud = pv.PolyData(distance)

    plotter1.add_mesh(
        point_cloud,
        scalars=distance[:, 2], # Color by height for better visuals
        cmap='viridis',
        point_size=5,
        render_points_as_spheres=True
    )
    plotter1.add_axes()
    plotter1.show_grid()

    # CRITICAL CHANGE: Pass the screenshot path INTO the show() command
    if plot:
        plotter1.show(screenshot=save_path_1 if save else None)
    elif save:
        plotter1.show(screenshot=save_path_1, auto_close=True)
    else:
        plotter1.close()

    # --- PHASE 2: CONVEX HULL MESH ---
    hull = ConvexHull(distance)
    faces = np.column_stack((np.full(len(hull.simplices), 3), hull.simplices)).flatten()
    mesh = pv.PolyData(distance, faces)

    plotter2 = pv.Plotter(off_screen=(not plot))
    plotter2.add_mesh(mesh, color='lightblue', opacity=0.5, show_edges=True)

    if target_state is not None and len(target_state) > 0:
        target_state_np = np.array(target_state)
        target_position = target_state_np[:, :3]
        target_cloud = pv.PolyData(target_position)
        plotter2.add_mesh(target_cloud, color='red', point_size=15, render_points_as_spheres=True)
        
        if target_name and len(target_name) > 0:
            labels = [f'{target_name[i]}' for i in range(min(len(target_position), len(target_name)))]
            plotter2.add_point_labels(target_position[:len(labels)], labels, font_size=12, text_color='red', always_visible=True)
            
    plotter2.add_axes()
    plotter2.show_grid()

    # CRITICAL CHANGE: Pass the screenshot path INTO the show() command
    if plot:
        plotter2.show(screenshot=save_path_2 if save else None)
    elif save:
        plotter2.show(screenshot=save_path_2, auto_close=True)
    else:
        plotter2.close()
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

    joint_limits = {
        'shoulder': (-1.998, 2.140),
        'upper_arm': (-2.002, 1.879),
        'lower_arm': (-1.642, 1.695),
        'wrist': (-1.77328, 1.81009),
        'gripper': (-2.9206, 2.9529)
    }
    
    joint_limits_unconstrained = {
        'shoulder': (-np.pi, np.pi),
        'upper_arm': (-np.pi, np.pi),
        'lower_arm': (-np.pi, np.pi),
        'wrist': (-np.pi, np.pi),
        'gripper': (-np.pi, np.pi)
    }

    target_state = np.array([[0.2, 0.2, 0.2, 0.0, 1.57, 0.65],
                               [0.2, 0.1, 0.4, 0.0, 0.0, 1.57],
                               [0.0 , 0.0 , 0.4 , 0.0, -0.785, 1.57],
                               [0.0 , 0.0,0.07, 3.141, 0.0, 0.0],
                               [0.0, 0.0452, 0.45, -0.785, 0.0, 3.141]])

    splits = 20 # Number of splits for each joint angle range (total configurations = splits^5)
    condition = True # Set to True to only consider points with x >= 0, False to consider all points
    plot = True # Set to True to visualize the workspace and target positions, False to skip visualization
    save_csv = False # Set to True to save the workspace points to a CSV file, False to skip saving
    
    
    distance = find_workspace(data, joint_limits, splits, condition)
    distance_unconstrained = find_workspace(data, joint_limits_unconstrained, splits, condition)

    if save_csv == True:
        np.savetxt('workspace_points.csv', distance, delimiter=',')

    plot_workspace(distance, plot, target_state,target_name=['I','II','III','IV-a','IV-b'], save=False)
    plot_workspace(distance_unconstrained, plot, None,target_name=None, save=False)
