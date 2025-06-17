import numpy as np
import open3d as o3d
import cv2 # For loading images (pip install opencv-python)
import os # For os.path.exists if you use it, though not in current snippet
from scipy.spatial.transform import Rotation # For RPY conversion (pip install scipy)

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

        rvec, tvec = cv2.calibrateHandEye(
            robot_rvecs,
            robot_tvecs,
            target_rvecs,
            target_tvecs,
            method=cv2.CALIB_HAND_EYE_PARK,
        )

        calibration = np.vstack((np.hstack((rvec, tvec)), [0, 0, 0, 1]))
        print("Calibration matrix", calibration)

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
def convert_rgbd_to_pointcloud(rgb, depth, intrinsic_matrix, extrinsic=None):
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
        depth_trunc=1000.0, 
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