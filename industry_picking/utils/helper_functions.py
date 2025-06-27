import numpy as np
import open3d as o3d
import cv2 # For loading images (pip install opencv-python)
import os # For os.path.exists if you use it, though not in current snippet
from scipy.spatial.transform import Rotation # For RPY conversion (pip install scipy)
from transforms3d.euler import euler2mat 
import rerun as rr
from PIL import Image


def calibrateHandEye(target_poses,robot_poses):
        target_rvecs, target_tvecs = [], []
        robot_rvecs, robot_tvecs = [], []
        for pose in robot_poses:
            T_inv =  np.linalg.inv(pose)
            R, _ = cv2.Rodrigues(T_inv[:3, :3])
            tvec = T_inv[:3, 3]
            robot_rvecs.append(R)
            robot_tvecs.append(tvec)

        for pose in target_poses:
            R, _ = cv2.Rodrigues(pose[:3, :3])
            tvec = pose[:3, 3]
            target_rvecs.append(R)
            target_tvecs.append(tvec)
                       
            #pose_np = np.array(pose).reshape(4, 4)
            #R, _ = cv2.Rodrigues(pose_np[:3, :3])   
            #tvec = pose_np[:3, 3]
            #target_rvecs.append(R)
            #target_tvecs.append(tvec)

        rvec, tvec = cv2.calibrateHandEye(
            robot_rvecs,
            robot_tvecs,
            target_rvecs,
            target_tvecs,
            method=cv2.CALIB_HAND_EYE_PARK,
        )

        calibration = np.vstack((np.hstack((rvec, tvec)), [0, 0, 0, 1]))
        print("Calibration matrix", calibration)
def create_pose_matrix(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg):
        # Convert the Euler angles from degrees to radians
        roll_rad = np.deg2rad(roll_deg)
        pitch_rad = np.deg2rad(pitch_deg)
        yaw_rad = np.deg2rad(yaw_deg)

        # Create the 3x3 rotation matrix from the Euler angles
        # 'sxyz' is a common convention for roll, pitch, yaw
        rotation_matrix = euler2mat(roll_rad, pitch_rad, yaw_rad, axes='sxyz')

        # ---- 2. Handle the Position ----
        # Create the position vector and convert it from millimeters to meters
        position_vector_meters = np.array([x_mm, y_mm, z_mm]) / 1000.0

        # ---- 3. Assemble the final 4x4 Pose Matrix ----
        # Start with a 4x4 identity matrix (a clean slate)
        pose_matrix = np.identity(4)

        # Place the 3x3 rotation matrix in the top-left corner
        pose_matrix[:3, :3] = rotation_matrix

        # Place the converted position vector in the last column
        pose_matrix[:3, 3] = position_vector_meters

        return pose_matrix 
def generate_hand_eye_test_data():
    """
    Generates a synthetic dataset for testing hand-eye calibration.
    """
    print("--- Generating Synthetic Test Data for Hand-Eye Calibration ---")
    
    # --- Step 1: Define the "Golden" Ground Truth Transformation ---
    # This is the matrix we want our calibration to find.
    # Let's assume the camera is 0.5m above the robot base, rotated 180 deg around Z.
    T_base_to_camera_GOLDEN = create_pose_matrix(x_mm=0.5, y_mm=0.1, z_mm=0.15, roll_deg=0, pitch_deg=0, yaw_deg=180)
    print("\n[Ground Truth] Golden Hand-Eye Matrix:\n", np.round(T_base_to_camera_GOLDEN, 2))

    # --- Step 2: Define the Fixed Offset from Gripper (TCP) to Target ---
    # This represents how the calibration board is attached to the robot's gripper.
    # Let's say the board is 10cm in front of the gripper.
    T_tcp_to_target_FIXED = create_pose_matrix(x_mm=0.1, y_mm=0, z_mm=0, roll_deg=0, pitch_deg=0, yaw_deg=0)
    print("\n[Fixed Offset] TCP to Target Matrix:\n", np.round(T_tcp_to_target_FIXED, 2))

    # --- Step 3: Generate a Set of Simulated Robot Poses ---
    # These are the poses of the robot's TCP relative to its base.
    # In a real test, you would have more complex and varied poses.
    robot_poses = [
        create_pose_matrix(x_mm=0.3, y_mm=0.1, z_mm=0.2, roll_deg=10, pitch_deg=0, yaw_deg=20),
        create_pose_matrix(x_mm=0.4, y_mm=-0.1, z_mm=0.3, roll_deg=-10, pitch_deg=15, yaw_deg=0),
        create_pose_matrix(x_mm=0.5, y_mm=0.0, z_mm=0.4, roll_deg=20, pitch_deg=-15, yaw_deg=-20),
        create_pose_matrix(x_mm=0.3, y_mm=0.2, z_mm=0.2, roll_deg=0, pitch_deg=25, yaw_deg=30)
    ]
    print(f"\nGenerated {len(robot_poses)} simulated robot poses.")

    # --- Step 4: Calculate the Corresponding "Perfect" Target Poses ---
    # For each robot pose, calculate what the camera should see.
    # The formula is: T_cam_to_target = inv(T_base_to_cam) * T_base_to_tcp * T_tcp_to_target
    target_poses = []
    T_camera_to_base_GOLDEN = np.linalg.inv(T_base_to_camera_GOLDEN) # We need the inverse for the calculation
    
    for robot_pose in robot_poses:
        # This calculates the pose of the target relative to the camera
        target_pose = T_camera_to_base_GOLDEN @ robot_pose @ T_tcp_to_target_FIXED
        target_poses.append(target_pose)
    print(f"Calculated {len(target_poses)} corresponding target poses for the camera to see.")

    return robot_poses, target_poses, T_base_to_camera_GOLDEN
def savePosesFile(poses: list, filename: str):
        try:
            with open(filename, 'w') as f:
                for i, pose_matrix in enumerate(poses):
                    # Save each matrix to the file
                    np.savetxt(f, pose_matrix, fmt='%.8f')
                    # Add a separator line between matrices (unless it's the last one)
                    if i < len(poses) - 1:
                        f.write('\n')
            print(f"\nSuccessfully saved {len(poses)} poses to '{filename}'")
        except Exception as e:
            print(f"\nAn error occurred while saving poses to file: {e}")
def loadPosesFile(filename: str) -> list:
        if not os.path.exists(filename):
            print(f"Error: Pose file not found at '{filename}'")
            return []
            
        try:
            # Load the entire file content as a single array
            # The blank lines will be treated as delimiters by fromstring
            with open(filename, 'r') as f:
                content = f.read()
            # Create an array from the string content, then reshape
            # Each matrix is 4x4 = 16 numbers.
            poses_flat = np.fromstring(content, sep=' ')
            num_matrices = len(poses_flat) // 16
            if len(poses_flat) % 16 != 0:
                print("Warning: File content is not a multiple of 16. The file might be corrupted.")
            
            # Reshape the flat array into a list of 4x4 matrices
            poses = poses_flat.reshape(num_matrices, 4, 4)
            print(f"\nSuccessfully loaded {len(poses)} poses from '{filename}'")
            return list(poses) # Convert the top-level array to a list
            
        except Exception as e:
            print(f"\nAn error occurred while loading poses from file: {e}")
            return []
def generate_charuco(ARUCO_DICT= cv2.aruco.DICT_6X6_250,
                     SQUARES_VERTICALLY= 6,SQUARES_HORIZONTALLY= 7,
                     SQUARE_LENGTH= 0.03,MARKER_LENGTH = 0.02,
                     LENGTH_PX= 0.02,MARGIN_PX= 20):
    SAVE_NAME = 'ChArUco_Marker.png'
    # ------------------------------

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
    cv2.imshow("img", img)
    cv2.waitKey(2000)
    cv2.imwrite(SAVE_NAME, img)
def rotation_matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix to roll, pitch, yaw (intrinsic Tait-Bryan xyz) in radians.
    Output order: [roll, pitch, yaw]
    """
    try:
        r = Rotation.from_matrix(matrix)
        # Using 'xyz' intrinsic: roll (alpha), pitch (beta), yaw (gamma)
        # R = Rx(alpha) * Ry(beta) * Rz(gamma)
        rpy = r.as_euler('xyz', degrees=False)
        return rpy
    except Exception as e:
        print(f"Error converting rotation matrix to RPY: {e}")
        print(f"Problematic matrix:\n{matrix}")
        return np.array([0.0, 0.0, 0.0]) # Default on error
def save_mask(mask: np.ndarray, output_path: str):
    """
    Saves a boolean numpy mask as a black and white PNG image.

    Args:
        mask (np.ndarray): The boolean mask (True/False values).
        output_path (str): The full path, including filename, where the mask will be saved.
    """
    try:
        # Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Convert the boolean mask to a black and white image (0 for False, 255 for True)
        mask_image = (mask.astype(np.uint8) * 255)
        
        # Save the image
        cv2.imwrite(output_path, mask_image)
        print(f"Successfully saved mask to: {output_path}")

    except Exception as e:
        print(f"Error saving mask to {output_path}: {e}")
def convert_rgbd_to_pointcloud(rgb, depth, intrinsic_matrix, extrinsic):
#provjera da li je uopste loadovao sliku, da li slika postoji
    if rgb is None or depth is None:
        print("RGB or depth image is None")
        return None
#provjera da li su slike iste rezolucije, bitno
    if rgb.shape[:2] != depth.shape[:2]:
        print(f"RGB shape {rgb.shape[:2]} and depth shape {depth.shape[:2]} have different sizes")
        return None
#konvertuje bgr u rgb
    rgb_converted_to_rgb_format = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    o3d_color_image = o3d.geometry.Image(rgb_converted_to_rgb_format)
    o3d_depth_image = o3d.geometry.Image(depth) 
#generise rgbd sliku, od koje dobijamo point cloud
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color_image,
        o3d_depth_image,
        convert_rgb_to_intensity=False, 
        depth_scale=1.0, 
        depth_trunc=3.0, 
    )
#provjerava da li su interni parametri tj. objekat intrinsic_matrix instanca klase o3d.camera.PinholeCameraIntrinsic
    if not isinstance(intrinsic_matrix, o3d.camera.PinholeCameraIntrinsic):
        print("Error: intrinsic_matrix must be an o3d.camera.PinholeCameraIntrinsic object.")
        return None
#generise point cloud od rgbd slike
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic_matrix, extrinsic 
    )
    return point_cloud
def save_image_to_file(image_data, file_path):
    """
    Saves an image represented as a NumPy array to a file.

    Args:
        image_data (np.ndarray): The image data. Expected to be a NumPy array.
                                 - For RGB: shape (height, width, 3), dtype=np.uint8
                                 - For RGBA: shape (height, width, 4), dtype=np.uint8
                                 - For Grayscale: shape (height, width), dtype=np.uint8
        file_path (str): The full path where the image will be saved,
                         including the desired extension (e.g., 'output/my_image.png').

    Returns:
        bool: True if the image was saved successfully, False otherwise.
    """
    try:
        # Create the directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f":: Created directory: {directory}")

        # Convert the NumPy array to a Pillow Image object
        image = Image.fromarray(image_data)
        
        # Save the image
        image.save(file_path)
        
        print(f":: Successfully saved image to {file_path}")
        return True
        
    except ValueError as e:
        print(f"Error: Invalid data for image conversion. The NumPy array might have an incorrect shape or data type. Details: {e}")
        return False
    except IOError as e:
        print(f"Error: Could not save the file to {file_path}. Check permissions or file path. Details: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def log_o3d_point_cloud_to_rerun(
    pcd: o3d.geometry.PointCloud,
    entity_path: str,
    point_radius: float = 0.002
):
    """
    Logs an Open3D point cloud to a Rerun session.

    Args:
        pcd (o3d.geometry.PointCloud): The Open3D point cloud object to log.
        entity_path (str): The path to log the point cloud to in Rerun
                           (e.g., "world/combined_point_cloud").
        point_radius (float): The radius to use for visualizing the points.
    """
    if not isinstance(pcd, o3d.geometry.PointCloud):
        print("Error: Input is not a valid Open3D PointCloud object.")
        return

    if not pcd.has_points():
        print(f"Warning: Point cloud for entity '{entity_path}' has no points. Nothing to log.")
        return

    # Extract points and convert to NumPy array
    points = np.asarray(pcd.points)

    # Extract colors if they exist, otherwise Rerun will use a default
    colors = None
    if pcd.has_colors():
        # Convert colors from [0, 1] float to [0, 255] uint8
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    # Log to Rerun using rr.Points3D
    rr.log(
        entity_path,
        rr.Points3D(
            positions=points,
            colors=colors,
            radii=point_radius
        )
    )
    print(f"Logged point cloud with {len(points)} points to Rerun entity '{entity_path}'.")
    
def cad_to_pointcloud_and_visualize(
        cad_file_path: str,
        entity_path: str = "world/cad_point_cloud",
        point_radius: float = 0.002,
        sample_points: int = 20000
    ):
        """
        Loads a CAD model (STL/OBJ/PLY), samples points from its surface, and visualizes it in Rerun.

        Args:
            cad_file_path (str): Path to the CAD file (STL, OBJ, PLY, etc.).
            entity_path (str): Rerun entity path for visualization.
            point_radius (float): Radius for visualized points.
            sample_points (int): Number of points to sample from the mesh surface.
        """
        if not os.path.exists(cad_file_path):
            print(f"Error: CAD file '{cad_file_path}' does not exist.")
            return

        mesh = o3d.io.read_triangle_mesh(cad_file_path)
        if mesh.is_empty():
            print(f"Error: Failed to load mesh from '{cad_file_path}'.")
            return
        # Optionally, you can use Poisson disk sampling for more uniform surface points:
        # pcd = mesh.sample_points_poisson_disk(number_of_points=sample_points, init_factor=5)
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
        o3d.visualization.draw_geometries([pcd], window_name="CAD Point Cloud Visualization")
        # Save the sampled point cloud to the specified file path
        pcd_save_path = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\data\stator.pcd"
        try:
            o3d.io.write_point_cloud(pcd_save_path, pcd)
            print(f"Point cloud saved to '{pcd_save_path}'.")
        except Exception as e:
            print(f"Failed to save point cloud: {e}")
        print(f"Visualized CAD model '{cad_file_path}' as point cloud in Rerun.")
        
def load_and_scale_cad_mesh(cad_file_path, sample_points=20000, scale_to_meters=True):
    """
    Loads a CAD mesh, checks its scale, and converts to meters if needed.
    Returns a uniformly sampled point cloud in meters.
    """
    if not os.path.exists(cad_file_path):
        print(f"Error: CAD file '{cad_file_path}' does not exist.")
        return None

    mesh = o3d.io.read_triangle_mesh(cad_file_path)
    if mesh.is_empty():
        print(f"Error: Failed to load mesh from '{cad_file_path}'.")
        return None

    vertices = np.asarray(mesh.vertices)
    max_dim = np.abs(vertices).max()
    print(f"[DEBUG] CAD mesh max dimension: {max_dim:.3f}")

    # Heuristic: if the largest dimension is >10, assume mm and scale to meters
    if scale_to_meters and max_dim > 10:
        print("[INFO] Scaling CAD mesh from millimeters to meters.")
        mesh.vertices = o3d.utility.Vector3dVector(vertices * 0.001)
    else:
        print("[INFO] CAD mesh assumed to be in meters.")

    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
    o3d.io.write_point_cloud(r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\data\stator.pcd", pcd)
    return pcd

