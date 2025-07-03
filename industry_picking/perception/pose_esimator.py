import rerun as rr
import numpy as np
import open3d as o3d
import cv2 # For loading images (pip install opencv-python)
import os # For os.path.exists if you use it, though not in current snippet
import copy 
from typing import List, Tuple, Optional # For type hinting
from industry_picking.perception.segmentation import getSegmentationMasksSAM
import industry_picking.utils.helper_functions as help
import matplotlib.pyplot as plt # For visualization (pip install matplotlib)

def visualize_fpfh(pcd, fpfh, voxel_size):
    # Color points by FPFH uniqueness (red = distinctive)
    fpfh_mean = np.mean(fpfh.data, axis=0)
    colors = plt.cm.viridis(fpfh_mean)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

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
        o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 0.999))
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
    masks_input_dir: Optional[str] = None,
    pcd_output_dir: Optional[str] = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\testSample",
    apply_mask: bool = False
) -> List[np.ndarray]:
    """
    Generates waypoints by segmenting objects, creating point clouds, and registering them.
    Uses a helper function to get masks from an online server or a local directory.
    """
    
    input_rgb_for_sam = cv2.imread(rgb_image_sam_path, cv2.IMREAD_UNCHANGED)
    depth_image_raw = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    
    # --- DEBUG: Print shapes and show images side by side ---
    if input_rgb_for_sam is not None:
        print(f"✅ DEBUG: RGB image resolution is: {input_rgb_for_sam.shape}")
    else:
        print("❌ ERROR: Failed to load RGB image.")

    if depth_image_raw is not None:
        print(f"✅ DEBUG: Depth image resolution is: {depth_image_raw.shape}")
    else:
        print("❌ ERROR: Failed to load depth image.")


    
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
    all_masks = getSegmentationMasksSAM(
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

    # Handle different depth image formats, including Zivid 4-channel depth images
    if len(depth_image_raw.shape) == 3:
        if depth_image_raw.shape[2] == 1:
            # Single channel depth (e.g., PNG 16-bit, loaded as (H, W, 1))
            processed_depth_image_original_resolution = depth_image_raw[:, :, 0]
        elif depth_image_raw.shape[2] == 3:
            # RealSense case: (H, W, 3) where depth is in the first channel
            # Sometimes RealSense saves depth as RGB where all channels are equal, or only the first channel is valid
            # We'll check if all channels are equal, otherwise use the first channel
            if np.all(depth_image_raw[:, :, 0] == depth_image_raw[:, :, 1]) and np.all(depth_image_raw[:, :, 0] == depth_image_raw[:, :, 2]):
                processed_depth_image_original_resolution = depth_image_raw[:, :, 0]
            else:
                print("Warning: Depth image has 3 channels but channels differ. Using the first channel as depth.")
                processed_depth_image_original_resolution = depth_image_raw[:, :, 0]
        elif depth_image_raw.shape[2] == 4:
            # Zivid case: (H, W, 4) where the first channel is depth, others may be confidence, noise, etc.
            print("Detected 4-channel (Zivid) depth image. Using the first channel as depth.")
            processed_depth_image_original_resolution = depth_image_raw[:, :, 0]
        else:
            print(f"Error: Depth image has an unsupported number of channels: {depth_image_raw.shape[2]}.")
            return waypoints
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
    
        
    target_down, target_fpfh = preprocess_point_cloud(target_pcd_reference, 0.001)
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
            help.save_mask(binary_mask_from_sam_resized, full_mask_path)
        
        if rerun_visualization:
            rr.log(f"world/instance_{instance_index}/sam_binary_mask", rr.SegmentationImage(binary_mask_from_sam_resized.astype(np.uint8)))

        # Do not apply mask to RGB or Depth images; use the full depth image for point cloud generation
        rgb_for_pcd_coloring = input_rgb_for_sam.copy()
        if apply_mask:
            depth_image_masked_instance = processed_depth_image_original_resolution.copy()
            depth_image_masked_instance[~binary_mask_from_sam_resized] = 0
        else:
            depth_image_masked_instance = processed_depth_image_original_resolution.copy()
        
        if np.count_nonzero(depth_image_masked_instance) == 0:
            print(f"Warning: Mask instance {instance_index} resulted in an empty depth map after masking. Skipping.")
            continue
            
        if rerun_visualization:
            rr.log(f"world/instance_{instance_index}/rgb_masked_for_pcd", rr.Image(cv2.cvtColor(rgb_for_pcd_coloring, cv2.COLOR_BGR2RGB)))
            rr.log(f"world/instance_{instance_index}/depth_masked", rr.DepthImage(depth_image_masked_instance.astype(np.float32), meter=depth_scale_to_meters))

        # Prepare for Open3D
        depth_for_o3d_instance = depth_image_masked_instance.astype(np.float32)
        depth_image_scaled_to_meters_instance = depth_for_o3d_instance / 1000


        print("[DEBUG] Depth SCALED image dtype:", depth_image_scaled_to_meters_instance.dtype)
        print("[DEBUG] Depth SCALED image min/max:", np.min(depth_image_scaled_to_meters_instance), np.max(depth_image_scaled_to_meters_instance))
        
        # --- DEBUG: Check X/Y spread of the depth mask (nonzero region) ---
        nonzero_indices = np.argwhere(depth_image_scaled_to_meters_instance > 0)
        if nonzero_indices.size > 0:
            y_min, x_min = np.min(nonzero_indices, axis=0)
            y_max, x_max = np.max(nonzero_indices, axis=0)
            print(f"[DEBUG] Depth mask nonzero X range: {x_min} to {x_max}, Y range: {y_min} to {y_max}")
        else:
            print("[DEBUG] No nonzero depth values after masking.")
        
        if rerun_visualization:
            rr.log(
                f"world/instance_{instance_index}/depth_scaled_to_meters",
                rr.DepthImage(depth_image_scaled_to_meters_instance.astype(np.float32), meter=1.0)
            )
        
        image_height, image_width = depth_image_masked_instance.shape[:2]
        o3d_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            image_width, image_height,
            camera_intrinsics_k[0, 0], camera_intrinsics_k[1, 1],
            camera_intrinsics_k[0, 2], camera_intrinsics_k[1, 2]
        )
        
        
        # Generate Point Cloud for this instance
        pcd_instance_camera_frame = help.convert_rgbd_to_pointcloud(
            rgb_for_pcd_coloring,
            depth_image_scaled_to_meters_instance,
            o3d_camera_intrinsics,
            np.identity(4)
        )
        

        # --- Rerun visualization of the generated point cloud from depth image ---
        if rerun_visualization:
            colors_for_rr = (np.asarray(pcd_instance_camera_frame.colors) * 255).astype(np.uint8) if pcd_instance_camera_frame.has_colors() else None
            rr.log(f"world/instance_{instance_index}/generated_pcd_from_depth", rr.Points3D(positions=np.asarray(pcd_instance_camera_frame.points), colors=colors_for_rr))

        if pcd_instance_camera_frame is None or not pcd_instance_camera_frame.has_points():
            print(f"Failed to generate point cloud for instance {instance_index}. Skipping.")
            continue
        
        
        # *** ADDED: Save the generated point cloud if an output directory is provided ***
        if pcd_output_dir:
            pcd_filename = f"instance_{instance_index}_point_cloud.ply"
            full_pcd_path = os.path.join(pcd_output_dir, pcd_filename)
            o3d.io.write_point_cloud(full_pcd_path, pcd_instance_camera_frame)
            print(f"Saved generated point cloud to: {full_pcd_path}")
        
        
        if rerun_visualization:
            colors_for_rr = (np.asarray(pcd_instance_camera_frame.colors) * 255).astype(np.uint8) if pcd_instance_camera_frame.has_colors() else None
            rr.log(f"world/instance_{instance_index}/segmented_pcd_camera_frame", rr.Points3D(positions=np.asarray(pcd_instance_camera_frame.points), colors=colors_for_rr))

        # Perform Registration against the reference model
        source_down, source_fpfh = preprocess_point_cloud(pcd_instance_camera_frame, voxel_size_registration)
        if not source_down.has_points():
            print(f"Downsampling segmented cloud for instance {instance_index} resulted in empty cloud. Skipping.")
            continue
        print(f"Source downsampled cloud has {len(source_down.points)} points.")
        
        coarse_reg_result = execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size_registration
        )
        # Normals are computed in preprocess_point_cloud() via pcd_down.estimate_normals()
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
        rpy_world = help.rotation_matrix_to_rpy(rotation_matrix_world)

        waypoint = np.concatenate((xyz_world, rpy_world))
        waypoints.append(waypoint)

        print(f"Instance {instance_index}: Waypoint in World Frame: XYZ=[{xyz_world[0]:.3f}, {xyz_world[1]:.3f}, {xyz_world[2]:.3f}], RPY(xyz)=[{rpy_world[0]:.3f}, {rpy_world[1]:.3f}, {rpy_world[2]:.3f}] rad")
        
        if rerun_visualization:
            #rr.log(f"world/instance_{instance_index}/waypoint_pose_world", rr.Transform3D(translation=xyz_world, mat3x3=rotation_matrix_world, axis_length=0.03))
            
            source_down_aligned_to_target = copy.deepcopy(source_down)
            source_down_aligned_to_target.transform(T_target_source)
            rr.log(f"world/instance_{instance_index}/registration_visualization/source_aligned_to_target", 
                   rr.Points3D(positions=np.asarray(source_down_aligned_to_target.points), 
                               colors=[255, 0, 0],
                               radii=0.0005))

    if not waypoints:
        print("\nNo waypoints were generated.")
    else:
        print(f"\nSuccessfully generated {len(waypoints)} waypoint(s).")
        
    return waypoints