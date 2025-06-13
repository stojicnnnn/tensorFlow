import rerun as rr
import numpy as np
import time
import open3d as o3d
import cv2 # For loading images (pip install opencv-python)
import os # For os.path.exists if you use it, though not in current snippet
import copy 
import requests
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation # For RPY conversion (pip install scipy)
from typing import List, Tuple, Optional # For type hinting
from dataclasses import dataclass
import glob
from xarm.wrapper import XArmAPI
import pyrealsense2 as rs
import math 
from transforms3d import euler
import glob # We'll use this for finding mask files
from transforms3d import affines 
from transforms3d.euler import euler2mat 
import cv2.aruco as aruco
# polinomalni koeficijenti look it up!

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
def display_point_clouds(source, target, transformation=None, window_name="Point Cloud Alignment"):
    """Helper function to visualize alignment of two point clouds."""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    source_temp.paint_uniform_color([1, 0.706, 0])  # Source is orange
    target_temp.paint_uniform_color([0, 0.651, 0.929]) # Target is blue
    
    if transformation is not None:
        source_temp.transform(transformation)
        
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)
def preprocess_point_cloud(pcd, voxel_size):
    """Downsample, estimate normals, and compute FPFH features."""
    print(f"Processing cloud with {len(pcd.points)} points.")
    print(f":: Downsampling with a voxel size {voxel_size:.3f}.")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"   Downsampled to {len(pcd_down.points)} u .")

    radius_normal = voxel_size * 2
    print(f":: Estimating normals with search radius {radius_normal:.3f}.")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(f":: Computing FPFH feature with search radius {radius_feature:.3f}.")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh
def execute_global_registration(source_down, target_down, source_fpfh,
                                
                               target_fpfh, voxel_size):
    """Global registration using RANSAC on FPFH features."""
    distance_threshold = voxel_size * 1.5 
    print(":: RANSAC registration on FPFH features with distance threshold %.3f" % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 
        3, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result
def refine_registration_icp(source, target, initial_transformation, voxel_size, use_point_to_plane=True):
    """Refine registration using ICP."""
    distance_threshold_icp = voxel_size * 0.4
    print(":: ICP registration with distance threshold %.3f." % distance_threshold_icp)

    if use_point_to_plane:
        if not source.has_normals(): # Source also needs normals for some ICP criteria or reciprocal checks
            print("   Estimating normals for source cloud (for point-to-plane ICP)...")
            source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        if not target.has_normals():
            print("   Estimating normals for target cloud (for point-to-plane ICP)...")
            target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold_icp, initial_transformation,
        estimation_method,
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=200) # Stricter criteria
    )
    return result
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
def filter_duplicate_waypoints(waypoints: List[np.ndarray], min_distance: float) -> List[np.ndarray]:
    """
    Filters a list of waypoints to remove duplicates based on proximity.

    Args:
        waypoints: A list of generated waypoints, where each waypoint is a NumPy array [x,y,z,r,p,y].
        min_distance: The minimum distance (in meters) between two waypoints to be considered unique.

    Returns:
        A new list of waypoints with duplicates removed.
    """
    if not waypoints:
        return []

    filtered_waypoints = []
    
    for waypoint in waypoints:
        is_duplicate = False
        # Check against the waypoints we've already approved
        for filtered_wp in filtered_waypoints:
            # Calculate the Euclidean distance between the XYZ coordinates
            distance = np.linalg.norm(waypoint[:3] - filtered_wp[:3])
            
            if distance < min_distance:
                is_duplicate = True
                print(f"Found duplicate waypoint. Distance: {distance:.4f}m < threshold: {min_distance:.4f}m. Discarding.")
                break # It's a duplicate of this one, no need to check further
        
        if not is_duplicate:
            filtered_waypoints.append(waypoint)
            
    return filtered_waypoints
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
def get_segmentation_masks(
    rgb_image_path: str,
    sam_server_url: Optional[str],
    sam_query: str,
    masks_input_dir: Optional[str],
    input_rgb_shape: Tuple[int, int]
) -> List[np.ndarray]:
    """
    Gets segmentation masks either from an online SAM server or a local directory.

    Tries the SAM server first. If it fails (due to network error, server down, etc.)
    or if no URL is provided, it falls back to loading masks from the local directory.

    Args:
        rgb_image_path: Path to the original RGB image (for the online server).
        sam_server_url: URL of the SAM server. Can be None to force local mode.
        sam_query: The text prompt for SAM.
        masks_input_dir: Path to the directory containing pre-saved mask images.
        input_rgb_shape: The (height, width) of the original RGB image.

    Returns:
        A list of boolean numpy arrays, where each array is a segmentation mask.
        Returns an empty list if both methods fail.
    """
    masks = []
    H_orig, W_orig = input_rgb_shape

    # --- Mode 1: Try Online SAM Server ---
    if sam_server_url:
        print(f"Attempting to get masks from online SAM server: {sam_server_url}")
        try:
            input_rgb_for_sam = cv2.imread(rgb_image_path)
            if input_rgb_for_sam is None:
                raise ValueError(f"Could not read RGB image at {rgb_image_path}")

            response = requests.post(
                sam_server_url,
                files={"image": cv2.imencode(".jpg", input_rgb_for_sam)[1].tobytes()},
                data={"query": sam_query},
                timeout=15 # A shorter timeout to fail faster if the server is offline
            )
            response.raise_for_status() # Check for HTTP errors like 404 or 500

            decoded_stack = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
            if decoded_stack is None:
                raise ValueError("Could not decode image received from SAM server.")

            # Process the stacked image returned by the server
            num_segments = decoded_stack.shape[0] // H_orig
            for i in range(1, num_segments):
                start_row = i * H_orig
                end_row = (i + 1) * H_orig
                instance_img = decoded_stack[start_row:end_row, :, :]
                # Convert to boolean mask
                binary_mask = np.any(instance_img > 10, axis=2) if len(instance_img.shape) == 3 else instance_img > 10
                masks.append(binary_mask)

            print(f"Successfully retrieved {len(masks)} masks from the online server.")
            return masks

        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"\nOnline SAM server failed: {e}")
            print("--- Falling back to local mask directory. ---\n")
            # If it fails, the function will proceed to the offline mode below

    # --- Mode 2: Fallback to Local Directory ---
    print(f"Loading masks from local directory: {masks_input_dir}")
    if not masks_input_dir or not os.path.isdir(masks_input_dir):
        print(f"Error: Local mask directory not found or not specified: {masks_input_dir}")
        return []

    # Find all png/jpg/jpeg files, sort them to ensure consistent order
    mask_files = sorted(glob.glob(os.path.join(masks_input_dir, "*.png")))
    mask_files.extend(sorted(glob.glob(os.path.join(masks_input_dir, "*.jpg"))))
    mask_files.extend(sorted(glob.glob(os.path.join(masks_input_dir, "*.jpeg"))))

    if not mask_files:
        print(f"Warning: No mask files found in {masks_input_dir}")
        return []

    for mask_file in mask_files:
        mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask_image is not None:
            # Convert to boolean mask (True for any pixel value > 10)
            binary_mask = mask_image > 10
            masks.append(binary_mask)

    print(f"Successfully loaded {len(masks)} masks from the local directory.")
    return masks
def rtvec_to_matrix(rvec, tvec):
    """Converts a rotation vector and a translation vector to a 4x4 transformation matrix."""
    rotation_matrix, _ = cv2.Rodrigues(rvec) # _ to ignore the Jakobian that cv2.Rodrigues returnes
    transformation_matrix = np.eye(4) # Square, neutral matrix, no rotation and no translatio
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = tvec.flatten() #flatten makes sure that we get a simple 2d array 1x3 and not 3x1
    return transformation_matrix
        #[[r11, r12, r13, tx],
        #[r21, r22, r23, ty],
        #[r31, r32, r33, tz],
        #[0.,  0.,  0.,  1.]]
def move(self, pose: np.ndarray):
        x = pose[0, 3] * 1000
        y = pose[1, 3] * 1000
        z = pose[2, 3] * 1000
        roll, pitch, yaw = euler.mat2euler(pose[:3, :3])
        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        error = self._arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=60,
            wait=True,
        )
def get_tcp_pose(self) -> np.ndarray:
        ok, pose = self._arm.get_position()
        if ok != 0:
            return None

        translation = np.array(pose[:3]) / 1000
        eulers = np.array(pose[3:]) * math.pi / 180
        rotation = euler.euler2mat(
            eulers[0], eulers[1], eulers[2], 'sxyz')
        pose = affines.compose(translation, rotation, np.ones(3))
        return pose
def generate_waypoints(
    rgb_image_sam_path: str,
    depth_image_path: str,
    camera_intrinsics_k: np.ndarray,
    camera_extrinsics: np.ndarray,
    reference_object_path: str,
    sam_server_url: Optional[str] = None,
    sam_query: str = "Segment object,1 instance at a time, in order",
    voxel_size_registration: float = 0.001,
    depth_scale_to_meters: float = 1000.0,
    rerun_visualization: bool = False,
    masks_output_dir: Optional[str] = None,
    masks_input_dir: Optional[str] = None
) -> List[np.ndarray]:
    """
    Generates waypoints by segmenting objects, creating point clouds, and registering them.
    Uses a helper function to get masks from an online server or a local directory.
    """
    waypoints = []
    if rerun_visualization:
        try:
            rr.init("generate_waypoints_demo", spawn=True)
        except Exception as e:
            print(f"Rerun initialization failed: {e}. Continuing without Rerun visualization.")
            rerun_visualization = False

    # 1. Load the main RGB image
    input_rgb_for_sam = cv2.imread(rgb_image_sam_path)
    if input_rgb_for_sam is None:
        print(f"Error: Could not read input RGB image: {rgb_image_sam_path}")
        return waypoints
    if rerun_visualization:
        rr.log("world/input_images/rgb_for_sam", rr.Image(cv2.cvtColor(input_rgb_for_sam, cv2.COLOR_BGR2RGB)))

    # 2. Get segmentation masks using the robust helper function
    all_masks = get_segmentation_masks(
        rgb_image_path=rgb_image_sam_path,
        sam_server_url=sam_server_url,
        sam_query=sam_query,
        masks_input_dir=masks_input_dir,
        input_rgb_shape=input_rgb_for_sam.shape[:2]
    )

    if not all_masks:
        print("Could not obtain any segmentation masks. Exiting.")
        return waypoints

    # 3. Load depth image
    depth_image_raw = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if depth_image_raw is None:
        print(f"Failed to load depth image from: {depth_image_path}.")
        return waypoints
    if len(depth_image_raw.shape) == 3 and depth_image_raw.shape[2] >= 1:
        processed_depth_image_original_resolution = depth_image_raw[:, :, 0]
    elif len(depth_image_raw.shape) == 2:
        processed_depth_image_original_resolution = depth_image_raw
    else:
        print(f"Error: Depth image has an unsupported shape: {depth_image_raw.shape}.")
        return waypoints

    # 4. Load and preprocess the reference model
    if not os.path.exists(reference_object_path):
        print(f"Error: Reference object file not found at '{reference_object_path}'.")
        return waypoints
    target_pcd_reference = o3d.io.read_point_cloud(reference_object_path)
    if not target_pcd_reference.has_points():
        print("Error: Target reference point cloud is empty or failed to load.")
        return waypoints
        
    target_down, target_fpfh = preprocess_point_cloud(target_pcd_reference, voxel_size_registration)
    if not target_down.has_points():
        print("Error: Downsampling target reference cloud resulted in an empty cloud. Adjust voxel_size.")
        return waypoints
    if rerun_visualization:
        rr.log("world/reference_model/target_downsampled", rr.Points3D(positions=np.asarray(target_down.points), colors=[0, 0, 255] if not target_down.has_colors() else None))

    # 5. Process each mask to generate a waypoint
    for instance_index, binary_mask_from_sam in enumerate(all_masks):
        print(f"\n--- Processing Mask Instance {instance_index} ---")

        # Resize mask if its dimensions don't match the depth image
        if binary_mask_from_sam.shape != processed_depth_image_original_resolution.shape[:2]:
            print(f"Resizing SAM binary mask from {binary_mask_from_sam.shape} to depth image shape {processed_depth_image_original_resolution.shape[:2]}")
            binary_mask_from_sam_resized = cv2.resize(
                binary_mask_from_sam.astype(np.uint8) * 255,
                (processed_depth_image_original_resolution.shape[1], processed_depth_image_original_resolution.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            binary_mask_from_sam_resized = binary_mask_from_sam

        # Save the mask for future offline use if an output directory is specified
        if masks_output_dir:
            mask_filename = f"mask_instance_{instance_index}.png"
            full_mask_path = os.path.join(masks_output_dir, mask_filename)
            save_mask(binary_mask_from_sam_resized, full_mask_path)
        
        if rerun_visualization:
            rr.log(f"world/instance_{instance_index}/sam_binary_mask", rr.SegmentationImage(binary_mask_from_sam_resized.astype(np.uint8)))

        # Apply mask to RGB and Depth images
        rgb_for_pcd_coloring = input_rgb_for_sam.copy()
        rgb_for_pcd_coloring[~binary_mask_from_sam_resized] = [0, 0, 0]
        
        depth_image_masked_instance = processed_depth_image_original_resolution.copy()
        depth_image_masked_instance[~binary_mask_from_sam_resized] = 0
        
        if np.count_nonzero(depth_image_masked_instance) == 0:
            print(f"Warning: Mask instance {instance_index} resulted in an empty depth map after masking. Skipping.")
            continue
            
        if rerun_visualization:
            rr.log(f"world/instance_{instance_index}/rgb_masked_for_pcd", rr.Image(cv2.cvtColor(rgb_for_pcd_coloring, cv2.COLOR_BGR2RGB)))
            rr.log(f"world/instance_{instance_index}/depth_masked", rr.DepthImage(depth_image_masked_instance.astype(np.float32), meter=depth_scale_to_meters))

        # Prepare for Open3D
        depth_for_o3d_instance = depth_image_masked_instance.astype(np.float32)
        depth_image_scaled_to_meters_instance = depth_for_o3d_instance / depth_scale_to_meters
        
        image_height, image_width = depth_image_scaled_to_meters_instance.shape[:2]
        o3d_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            image_width, image_height,
            camera_intrinsics_k[0, 0], camera_intrinsics_k[1, 1],
            camera_intrinsics_k[0, 2], camera_intrinsics_k[1, 2]
        )
        
        # Generate Point Cloud for this instance
        pcd_instance_camera_frame = convert_rgbd_to_pointcloud(
            rgb_for_pcd_coloring,
            depth_image_scaled_to_meters_instance,
            o3d_camera_intrinsics,
            np.identity(4)
        )

        if pcd_instance_camera_frame is None or not pcd_instance_camera_frame.has_points():
            print(f"Failed to generate point cloud for instance {instance_index}. Skipping.")
            continue
        if rerun_visualization:
            colors_for_rr = (np.asarray(pcd_instance_camera_frame.colors) * 255).astype(np.uint8) if pcd_instance_camera_frame.has_colors() else None
            rr.log(f"world/instance_{instance_index}/segmented_pcd_camera_frame", rr.Points3D(positions=np.asarray(pcd_instance_camera_frame.points), colors=colors_for_rr))

        # Perform Registration against the reference model
        source_down, source_fpfh = preprocess_point_cloud(pcd_instance_camera_frame, voxel_size_registration)
        if not source_down.has_points():
            print(f"Downsampling segmented cloud for instance {instance_index} resulted in empty cloud. Skipping.")
            continue

        coarse_reg_result = execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size_registration
        )
        
        icp_reg_result = refine_registration_icp( # Using your ICP function
            source_down, target_down, coarse_reg_result.transformation, voxel_size_registration, use_point_to_plane=True
        )
        
        if icp_reg_result.fitness < 0.3:
            print(f"Warning: Low fitness ({icp_reg_result.fitness:.4f}) after ICP for instance {instance_index}. Waypoint may be inaccurate.")
            # We can choose to continue or skip bad registrations
            # continue 

        T_target_source = icp_reg_result.transformation

        try:
            T_camera_object = np.linalg.inv(T_target_source)
        except np.linalg.LinAlgError:
            print(f"Error: Could not compute inverse of registration matrix for instance {instance_index}. Skipping.")
            continue
            
        # Transform object pose from Camera Frame to World Frame
        T_world_object = camera_extrinsics @ T_camera_object

        xyz_world = T_world_object[0:3, 3]
        rotation_matrix_world = T_world_object[0:3, 0:3]
        rpy_world = rotation_matrix_to_rpy(rotation_matrix_world)

        waypoint = np.concatenate((xyz_world, rpy_world))
        waypoints.append(waypoint)

        print(f"Instance {instance_index}: Waypoint in World Frame: XYZ=[{xyz_world[0]:.3f}, {xyz_world[1]:.3f}, {xyz_world[2]:.3f}], RPY(xyz)=[{rpy_world[0]:.3f}, {rpy_world[1]:.3f}, {rpy_world[2]:.3f}] rad")
        
        if rerun_visualization:
            rr.log(f"world/instance_{instance_index}/waypoint_pose_world", rr.Transform3D(translation=xyz_world, mat3x3=rotation_matrix_world, axis_length=0.03))
            
            source_down_aligned_to_target = copy.deepcopy(source_down)
            source_down_aligned_to_target.transform(T_target_source)
            rr.log(f"world/instance_{instance_index}/registration_visualization/source_aligned_to_target", 
                   rr.Points3D(positions=np.asarray(source_down_aligned_to_target.points), 
                               colors=[255, 0, 0],
                               radii=0.002))

    if not waypoints:
        print("\nNo waypoints were generated.")
    else:
        print(f"\nSuccessfully generated {len(waypoints)} waypoint(s).")
        
    return waypoints
def get_cam_intrinsics():
   # 1. Create INSTANCES of the pipeline and config objects
    pipeline = rs.pipeline()
    config = rs.config()

    # 2. Call enable_stream() on the config INSTANCE
    # It's good practice to enable specific streams with a known resolution
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 3. Call start() on the pipeline INSTANCE, passing the config INSTANCE
    profile = pipeline.start(config)

    try:
        # Get the active stream profiles from the started profile object
        depth_profile = profile.get_stream(rs.stream.depth)
        color_profile = profile.get_stream(rs.stream.color)

        # Downcast the profile to a video_stream_profile and get intrinsics
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        # --- Print Depth Stream Intrinsics ---
        print("--- Depth Stream Intrinsics ---")
        print(f"  Width: {depth_intrinsics.width}")
        print(f"  Height: {depth_intrinsics.height}")
        print(f"  Principal Point (ppx, ppy): ({depth_intrinsics.ppx:.3f}, {depth_intrinsics.ppy:.3f})")
        print(f"  Focal Length (fx, fy): ({depth_intrinsics.fx:.3f}, {depth_intrinsics.fy:.3f})")
        print(f"  Distortion Model: {depth_intrinsics.model}")
        print(f"  Distortion Coefficients: {depth_intrinsics.coeffs}")
        print("-" * 33)

        # --- Print Color Stream Intrinsics ---
        print("\n--- Color Stream Intrinsics ---")
        print(f"  Width: {color_intrinsics.width}")
        print(f"  Height: {color_intrinsics.height}")
        print(f"  Principal Point (ppx, ppy): ({color_intrinsics.ppx:.3f}, {color_intrinsics.ppy:.3f})")
        print(f"  Focal Length (fx, fy): ({color_intrinsics.fx:.3f}, {color_intrinsics.fy:.3f})")
        print(f"  Distortion Model: {color_intrinsics.model}")
        print(f"  Distortion Coefficients: {color_intrinsics.coeffs}")
        print("-" * 33)

        # You should also return the values if you want to use them elsewhere
        return color_intrinsics, depth_intrinsics

    finally:
        # 4. Call stop() on the pipeline INSTANCE
        #pipeline.stop()
        print("\nGot camera intrinsics.")
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
def test_camera():
    WIDTH = 1280
    HEIGHT = 720
    FPS = 30
    # -----------------------------

    pipeline = None
    try:
        print("Attempting to initialize RealSense camera...")
        pipeline = rs.pipeline()
        config = rs.config()

        print(f"Requesting Depth Stream: {WIDTH}x{HEIGHT} @ {FPS} FPS")
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        
        print(f"Requesting Color Stream: {WIDTH}x{HEIGHT} @ {FPS} FPS")
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

        print("\nStarting pipeline...")
        profile = pipeline.start(config)
        
        print("\nSUCCESS! Pipeline started successfully.")
        print("Camera is initialized and streams are resolved.")

        # Get one frame to confirm it works
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Could not get frames, but pipeline started.")
        else:
            print(f"Successfully received frames: Color={color_frame.width}x{color_frame.height}, Depth={depth_frame.width}x{depth_frame.height}")

    except RuntimeError as e:
        print("\n-------------------------")
        print(">>> FAILED to start pipeline! <<<")
        print(f"RuntimeError: {e}")
        print("-------------------------")
        print("Possible Solutions:")
        print("1. Did you completely CLOSE the Intel RealSense Viewer application?")
        print("2. Is the camera plugged into a blue USB 3.0 port?")
        print("3. Do the WIDTH, HEIGHT, and FPS in this script exactly match a working combination from the Viewer?")

    finally:
        if pipeline:
            print("\nStopping pipeline.")
            pipeline.stop()
def record_manual_poses(robot_ip: str, num_poses: int = 10):
    """
    Interactively records a specified number of robot poses.
    
    The user manually moves the robot to a desired position and presses Enter
    to record the pose. This repeats for the specified number of iterations.
    
    Args:
        robot_ip (str): The IP address of the xArm.
        num_poses (int): The total number of poses to record.

    Returns:
        list: A list of 4x4 numpy arrays, where each array is a recorded pose.
              Returns an empty list if the connection fails or is interrupted.
    """
    
    # List to store the recorded 4x4 pose matrices
    recorded_poses = []
    
    # Initialize and connect to the robot arm
    arm = XArmAPI(robot_ip)
    arm.connect()
    if not arm.connected:
        print(f"Error: Failed to connect to xArm at IP: {robot_ip}")
        return recorded_poses

    try:
        # Enable manual mode on the robot so it can be moved by hand.
        # This will require you to press the "Freedrive" button on the arm.
        print("\nEnabling manual mode on the robot arm.")
        print("You will need to press and hold the 'Freedrive' button on the arm to move it.")
        arm.set_mode(2)  # Mode 2 is manual movement mode
        arm.set_state(0)
        time.sleep(1)

        print(f"\nReady to record {num_poses} poses.")
        print("="*40)

        # Loop for the specified number of iterations
        for i in range(num_poses):
            # Wait for the user to press a key
            input(f"--> Pose {i+1}/{num_poses}: Move the robot to the desired position, then press Enter to record...")
            
            # Read the robot's current position
            error_code, current_pose_list = arm.get_position(is_radian=False)
            
            if error_code != 0:
                print(f"Error reading robot position (error code: {error_code}). Skipping this pose.")
                continue

            # Extract XYZ and RPY values
            # The SDK returns position in millimeters, so we convert to meters
            x_meters = current_pose_list[0] 
            y_meters = current_pose_list[1] 
            z_meters = current_pose_list[2] 
            roll_degrees = current_pose_list[3]
            pitch_degrees = current_pose_list[4]
            yaw_degrees = current_pose_list[5]

            print(f"    ...Recording pose: X={x_meters:.4f}m, Y={y_meters:.4f}m, Z={z_meters:.4f}m, R={roll_degrees:.1f}°, P={pitch_degrees:.1f}°, Y={yaw_degrees:.1f}°")

            # Convert the pose to a 4x4 matrix
            pose_matrix = create_pose_matrix(
                x_mm=x_meters,
                y_mm=y_meters,
                z_mm=z_meters,
                roll_deg=roll_degrees,
                pitch_deg=pitch_degrees,
                yaw_deg=yaw_degrees
            )

            if pose_matrix is not None:
                # Store the matrix in our list
                recorded_poses.append(pose_matrix)
                print(f"    ...Pose {i+1} successfully recorded.")
            else:
                print(f"    ...Failed to create matrix for pose {i+1}. Skipping.")

    except Exception as e:
        print(f"\nAn error occurred during the process: {e}")
    finally:
        # Always make sure to disconnect from the robot
        print("\nDisconnecting from the robot arm.")
        arm.set_mode(0) # Set back to position control mode
        arm.set_state(0)
        arm.disconnect()
        
    return recorded_poses
def save_poses_to_file(poses: list, filename: str):
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
def load_poses_from_file(filename: str) -> list:
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

if __name__ == "__main__1":
    
    # !!!! REPLACE WITH YOUR ROBOT'S IP ADDRESS !!!!
    XARM_IP_ADDRESS = "192.168.1.XXX"
    NUMBER_OF_POSES_TO_RECORD = 10

    # Run the recording function
    my_poses = record_manual_poses(XARM_IP_ADDRESS, NUMBER_OF_POSES_TO_RECORD)

    # After the loop is done, print out all the recorded poses
    if my_poses:
        print("\n" + "="*50)
        print("          All Recorded Poses")
        print("="*50 + "\n")
        
        for i, pose_matrix in enumerate(my_poses):
            # Define a variable name for easy copy-pasting
            variable_name = f"target_pose{i+1}"
            
            # Use NumPy's print options for nice formatting
            np.set_printoptions(precision=8, suppress=True)
            
            print(f"# This is {variable_name}, generated from your manual position {i+1}")
            print(f"{variable_name} = np.array([")
            # Format each row to look clean
            for row in pose_matrix:
                print(f"    [ {row[0]:9.8f}, {row[1]:9.8f}, {row[2]:9.8f}, {row[3]:9.8f}],")
            print("])\n")
            
    else:
        print("\nNo poses were recorded.")

    print("Program finished.")

if __name__ == "__main__":
    ROBOT_IP = "192.168.1.184" 
    #checkerboard parameters 
    #poses_to_visit = record_manual_poses("192.168.1.184",10)
    #if poses_to_visit:
    #    save_poses_to_file(poses_to_visit, "test1")

    robot_poses_to_visit = load_poses_from_file("test1")

    CHARUCO_SQUARES_X = 6
    CHARUCO_SQUARES_Y = 7      # Number of squares in height
    CHARUCO_SQUARE_LENGTH_M = 0.0263 # Size of a square in METERS
    CHARUCO_MARKER_LENGTH_M = 0.0177 # Size of an ArUco marker in METERS
    CHARUCO_DICTIONARY = aruco.getPredefinedDictionary(aruco.DICT_6X6_250) # Example dictionary
    
    # Checkerboard corners 
    charuco_board = aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y), 
        CHARUCO_SQUARE_LENGTH_M, 
        CHARUCO_MARKER_LENGTH_M, 
        CHARUCO_DICTIONARY
    )
    aruco_params = aruco.DetectorParameters()

    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []


    #camera paramteres
    image_width_for_intrinsics = 1224
    image_height_for_intrinsics = 1024
    fx = 1050.0 # example
    fy = 1050.0 # example
    cx = image_width_for_intrinsics / 2.0
    cy = image_height_for_intrinsics / 2.0

    camera_K = np.array([
         [fx, 0, cx],
         [0, fy, cy],
         [0, 0,  1]
    ])

    cam_fx = 525.0
    cam_fy = 525.0
    cam_cx = 319.5
    cam_cy = 239.5
    example_camera_K = np.array([
        [cam_fx, 0, cam_cx],
        [0, cam_fy, cam_cy],
        [0, 0,  1]
    ])

    # target_pose1 = np.array([
    # [ 0.38600143, -0.92241696, -0.01831418,  0.3497],
    # [-0.92237976, -0.38607185, -0.03009714, -0.0701],
    # [-0.03502841,  0.00702671, -0.99935933,  0.4153],
    # [ 0.        ,  0.        ,  0.        ,  1.    ]
    # ])
    
    # target_pose2 = np.array([
    # [ 0.38596645, -0.92243403, -0.01285227,  0.344 ],
    # [-0.92240989, -0.38601614, -0.03009774, -0.1187],
    # [-0.03262602,  0.0152643 , -0.9993616 ,  0.4155],
    # [ 0.        ,  0.        ,  0.        ,  1.    ]
    # ])
    
    # target_pose3 = np.array([
    # [ 0.38596645, -0.92243403, -0.01285227,  0.352 ],
    # [-0.92240989, -0.38601614, -0.03009774, -0.0778],
    # [-0.03262602,  0.0152643 , -0.9993616 ,  0.3509],
    # [ 0.        ,  0.        ,  0.        ,  1.    ]
    # ])
    
    # target_pose4 = np.array([
    # [ 0.38596645, -0.92243403, - 0.01285227 ,  0.352 ],
    # [ -0.92240989 , - 0.38601614, - 0.03009774, -0.064],
    # [-0.03262602, 0.0152643, - 0.9993616,  0.330],
    # [ 0.        ,  0.        ,  0.        ,  1.    ]
    # ])
    
    # target_pose5 = np.array([
    # [ 0.38596645, -0.92243403, -0.01285227,  0.344 ],
    # [-0.92240989, -0.38601614, -0.03009774, -0.1187],
    # [-0.03262602,  0.0152643 , -0.9993616 ,  0.4155],
    # [ 0.        ,  0.        ,  0.        ,  1.    ]
    # ])

    # # --- Initialize Robot Arm ---
    print("Initializing UFactory xArm...")
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_mode(0) # Position control mode
    arm.set_state(state=0) # Ready state
    print("xArm Initialized.")

    for i, pose in enumerate(robot_poses_to_visit):
        #pocetak testa
        # --- Initialize RealSense Camera ---
        print("Initializing Intel RealSense Camera...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1920 , 1080, rs.format.bgr8, 30)
        pipeline.start(config)
        print("RealSense Camera Initialized.")
        time.sleep(2) # Give camera time to auto-adjust exposure, etc.

        color_intrinsics, _ = get_cam_intrinsics()
        camera_matrix = np.array([
        [color_intrinsics.fx, 0, color_intrinsics.ppx],
        [0, color_intrinsics.fy, color_intrinsics.ppy],
        [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.array(color_intrinsics.coeffs, dtype=np.float32)
        #kraj testa


        print(f"\n--- Processing Pose {i+1}/{len(robot_poses_to_visit)} ---")

        # 1. Move Robot
        print(f"Moving robot to pose: {pose}")
        move(arm,pose) #robot_poses_to_visit[i]
        print("Robot move complete.")
        time.sleep(2) # Wait a moment for vibrations to settle
        # 2. Get Robot Pose
        # Get the 4x4 matrix representing the pose of the end-effector (gripper) in the base fram
        T_gripper_in_base = get_tcp_pose(arm)
        print("Retrieved robot pose.")
        
        # 3. Capture Camera Image
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("Warning: Could not get color frame. Skipping this pose.")
            continue
        image = np.asanyarray(color_frame.get_data())
        print("Captured camera image.")
        
        # 4. Detect charuco board
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, CHARUCO_DICTIONARY, parameters=aruco_params)

        # 5. Calculate Target Pose in Camera Frame & Store Data
        if ids is not None and len(ids) > 0:
            print(f"Found {len(ids)} ArUco markers. Interpolating ChArUco corners...")
            
            # Interpolate to find the precise checkerboard corners from the detected ArUco markers
            retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, charuco_board
            )
            # If we found enough ChArUco corners, estimate the board's pose
            if retval and charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                print(f"Successfully interpolated {len(charuco_corners)} ChArUco corners. Estimating pose...")

                # Estimate the pose of the ChArUco board.
                # This function directly gives the pose relative to the camera frame.
                # It uses the 3D board definition and the found 2D corners.
                success, rvec, tvec = aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs, None, None
                )

                if success:
                    print("ChArUco pose estimated successfully!")
                    
                    # Convert to 4x4 matrix
                    T_target_in_cam = rtvec_to_matrix(rvec, tvec)
                    # Saving target and robot poses
                    R_gripper2base_list.append(T_gripper_in_base[0:3, 0:3]) 
                    t_gripper2base_list.append(T_gripper_in_base[0:3, 3])   
                    R_target2cam_list.append(T_target_in_cam[0:3, 0:3])
                    t_target2cam_list.append(T_target_in_cam[0:3, 3])


                    # Draw the detected corners and axes for visualization
                    image_with_detections = aruco.drawDetectedCornersCharuco(image.copy(), charuco_corners, charuco_ids)
                    cv2.drawFrameAxes(image_with_detections, camera_matrix, dist_coeffs, rvec, tvec, 0.1) # Draw a 10cm axis
                    cv2.imshow('ChArUco Detection', image_with_detections)
                    cv2.waitKey(500)
                    print("Successfully stored data pair for this pose.")
                else:
                    print("Warning: Pose estimation for ChArUco board failed.")
            else:
                print("Warning: Not enough ChArUco corners interpolated to estimate pose.")
        else:
            print("Warning: No ArUco markers found in this image. Skipping pose.")
            cv2.imshow('Detection Failed', image)

    # Clean up
    #cv2.destroyAllWindows()
    #pipeline.stop()
    print("\nCalibration loop finished. Collected data for test")

    # ==== 4. PERFORM CALIBRATION CALCULATION ====    

    print("\nPerforming Hand-Eye Calibration calculation...")
    # This function calculates the transformation from the robot base to the camera frame
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base_list,
        t_gripper2base=t_gripper2base_list,
        R_target2cam=R_target2cam_list,
        t_target2cam=t_target2cam_list,
        # PARK is a common method for calibration.
        # Try others for better results
        method=cv2.CALIB_HAND_EYE_PARK 
    )

    # ==== 5. SAVE AND DISPLAY THE RESULT ====
    
    # Combine rotation and translation into a single 4x4 matrix
    T_cam_in_base = np.eye(4)
    T_cam_in_base[0:3, 0:3] = R_cam2base
    T_cam_in_base[0:3, 3] = t_cam2base.flatten()

    print("\n--- Hand-Eye Calibration Result ---")
    print("Transformation Matrix (T_camera_in_base):")
    print(T_cam_in_base)

    # Save the result to a file for later use
    np.save("hand_eye_calibration_matrix.npy", T_cam_in_base)
    print("\nCalibration matrix saved to 'hand_eye_calibration_matrix.npy'")
    cv2.waitKey(10)
if __name__ == "null":
    
    #camera paramteres
    image_width_for_intrinsics = 1224
    image_height_for_intrinsics = 1024
    fx = 1050.0 # example
    fy = 1050.0 # example
    cx = image_width_for_intrinsics / 2.0
    cy = image_height_for_intrinsics / 2.0

    camera_K = np.array([
         [fx, 0, cx],
         [0, fy, cy],
         [0, 0,  1]
    ])

    cam_fx = 525.0
    cam_fy = 525.0
    cam_cx = 319.5
    cam_cy = 239.5
    example_camera_K = np.array([
        [cam_fx, 0, cam_cx],
        [0, cam_fy, cam_cy],
        [0, 0,  1]
    ])
    #some default extrinsic camera matrix, aligned with world axis
    example_camera_T_world_cam = np.identity(4)

    path_to_raw_rgb_for_sam = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\2trid.png" # Original RGB for SAM
    path_to_raw_depth = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\2depthmap.png" # Original Depth
    path_to_reference_model = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\ref_sample_true.ply" # Our best sample of an object we are picking

    generated_robot_waypoints = generate_waypoints(
            rgb_image_sam_path=path_to_raw_rgb_for_sam,
            depth_image_path=path_to_raw_depth,
            camera_intrinsics_k=example_camera_K, # Use the K matrix for your specific camera and depth resolution
            camera_extrinsics=example_camera_T_world_cam,
            reference_object_path=path_to_reference_model,
            sam_server_url= None, # Your SAM server
            sam_query="Segment the circular grey metallic caps,1 instance at a time, in order", # Your SAM query
            voxel_size_registration=0.001, # Adjust as needed
            depth_scale_to_meters=1000.0, # If depth values are in mm
            rerun_visualization=True,
            masks_output_dir= None,
            masks_input_dir=r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\mask2" # Where to load from
    )
    
    if generated_robot_waypoints:
        print(f"\n--- Generated {len(generated_robot_waypoints)} raw waypoints. Now filtering duplicates... ---")

        # Define how close two waypoints can be before being called a duplicate.
        MINIMUM_DISTANCE_BETWEEN_OBJECTS = 0.02  # 2 centimeters

        final_waypoints = filter_duplicate_waypoints(
            generated_robot_waypoints,
            min_distance=MINIMUM_DISTANCE_BETWEEN_OBJECTS
        )

    if final_waypoints:
            print("\nVisualizing final waypoints and path in Rerun under 'world/final_waypoints'...")

            # First, log the connecting line strip for the clean path
            final_path_positions = [wp[:3] for wp in final_waypoints]
            
            # Then, log each final waypoint as a coordinate system pose
            for idx, wp in enumerate(final_waypoints):
                translation = wp[:3]
                # The waypoint stores RPY angles; we need to convert them back to a rotation matrix for Rerun
                rotation_matrix = Rotation.from_euler('xyz', wp[3:], degrees=False).as_matrix()
                
                rr.log(
                    f"world/final_waypoints/{idx}",
                    rr.Transform3D(
                        translation=translation,
                        mat3x3=rotation_matrix,
                        axis_length=0.05 
                    )
                )
                # If using the workaround for older SDKs:
                # axis_length = 0.05
                # rr.log(f"world/final_waypoints/{idx}/axes", rr.Arrows3D(...))

    else:
        print("No waypoints were generated by the function.")