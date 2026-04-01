import numpy as np
from itertools import product
import pyvista as pv
from scipy.spatial import ConvexHull


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

def plot_workspace(distance, plot, target_state,target_name):
    plotter = pv.Plotter()
    target_position = target_state[:,:3]
    # Create a point cloud
    point_cloud = pv.PolyData(distance)

    # Add the point cloud with a color map
    plotter.add_mesh(
        point_cloud,
        scalars=np.arange(len(distance)),
        cmap='viridis',
        point_size=5,
        render_points_as_spheres=True
    )

    # Add axes
    plotter.add_axes()
    plotter.show_grid()
    if plot:
        plotter.show()

    hull = ConvexHull(distance)

    faces = []
    for simplex in hull.simplices:
        faces.append([3, simplex[0], simplex[1], simplex[2]])

    faces = np.hstack(faces)
    mesh = pv.PolyData(distance, faces)

    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', opacity=0.5, show_edges=True)

    # Add target positions as red spheres with labels
    target_cloud = pv.PolyData(target_position)
    plotter.add_mesh(
        target_cloud,
        color='red',
        point_size=15,
        render_points_as_spheres=True
    )
    # Add labels for each target
    labels = [f'{target_name[i]}' for i in range(len(target_position))]
    plotter.add_point_labels(target_position, labels, font_size=12, text_color='red', always_visible=True)

    plotter.add_axes()
    plotter.show_grid()
    if plot:
        plotter.show()


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
        'shoulder': (-2, 2),
        'upper_arm': (-np.pi/2, np.pi/2),
        'lower_arm': (-np.pi/2, np.pi/2),
        'wrist': (-np.pi/2, np.pi/2),
        'gripper': (-np.pi, np.pi)
    }

    target_state = np.array([[0.2, 0.2, 0.2, 0.0, 1.57, 0.65],
                               [0.2, 0.1, 0.4, 0.0, 0.0, 1.57],
                               [0.0 , 0.0 , 0.4 , 0.0, -0.785, 1.57],
                               [0.0 , 0.0,0.07, 3.141, 0.0, 0.0],
                               [0.0, 0.0452, 0.45, -0.785, 0.0, 3.141]])

    splits = 10 # Number of splits for each joint angle range (total configurations = splits^5)
    condition = False # Set to True to only consider points with x >= 0, False to consider all points
    plot = True # Set to True to visualize the workspace and target positions, False to skip visualization
    save = False # Set to True to save the workspace points to a CSV file, False to skip saving
    
    
    distance = find_workspace(data, joint_limits, splits, condition)

    if save == True:
        np.savetxt('workspace_points.csv', distance, delimiter=',')

    plot_workspace(distance, plot, target_state,target_name=['I','II','III','IV-a','IV-b']) 
