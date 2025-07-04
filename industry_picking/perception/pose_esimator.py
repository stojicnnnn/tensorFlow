import rerun as rr
import numpy as np
import open3d as o3d
import cv2 # For loading images (pip install opencv-python)
import os # For os.path.exists if you use it, though not in current snippet
import copy 
from typing import List, Tuple, Optional # For type hinting
from industry_picking.perception.segmentation import getSegmentationMasksSAM
import industry_picking.utils.helper_functions as help


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

def filter_duplicate_waypoints(
    waypoints: List[np.ndarray], 
    min_distance: float,
    object_origin: np.ndarray = np.zeros(3)  # Default origin at [0,0,0]
) -> List[np.ndarray]:
    """
    Filters a list of 4x4 waypoints to remove duplicates based on XYZ proximity,
    keeping the waypoint closer to the object origin when duplicates are found.

    Args:
        waypoints: List of 4x4 transformation matrices.
        min_distance: Minimum Euclidean distance (meters) between two waypoints.
        object_origin: XYZ coordinates of the sample object's origin (default [0,0,0]).

    Returns:
        Filtered list of 4x4 matrices with duplicates removed (keeping closer waypoints).
    """
    if not waypoints:
        return []

    filtered_waypoints = []
    
    for waypoint in waypoints:
        current_translation = waypoint[:3, 3]
        current_distance = np.linalg.norm(current_translation - object_origin)
        
        # Check against already filtered waypoints
        duplicate_index = None
        for i, filtered_wp in enumerate(filtered_waypoints):
            distance = np.linalg.norm(current_translation - filtered_wp[:3, 3])
            if distance < min_distance:
                duplicate_index = i
                break
        
        if duplicate_index is not None:
            # Compare distances to origin
            existing_distance = np.linalg.norm(
                filtered_waypoints[duplicate_index][:3, 3] - object_origin
            )
            
            if current_distance < existing_distance:
                # Replace the existing waypoint with the closer one
                print(f"Replacing duplicate waypoint (distance to origin: {existing_distance:.4f}m) "
                      f"with closer one (distance: {current_distance:.4f}m)")
                filtered_waypoints[duplicate_index] = waypoint
            else:
                print(f"Keeping existing waypoint (distance to origin: {existing_distance:.4f}m), "
                      f"discarding new one (distance: {current_distance:.4f}m)")
        else:
            filtered_waypoints.append(waypoint)

    return filtered_waypoints

def load_and_validate_images(rgb_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate RGB and depth images."""
    rgb_image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if rgb_image is None:
        raise FileNotFoundError(f"Failed to load RGB image: {rgb_path}")
    if depth_image is None:
        raise FileNotFoundError(f"Failed to load depth image: {depth_path}")
        
    print(f"✅ RGB image resolution: {rgb_image.shape}")
    print(f"✅ Depth image resolution: {depth_image.shape}")
    
    return rgb_image, depth_image

def process_depth_image(depth_image: np.ndarray) -> np.ndarray:
    """Process depth image to handle different formats."""
    if len(depth_image.shape) == 3:
        if depth_image.shape[2] == 1:
            return depth_image[:, :, 0]
        elif depth_image.shape[2] == 3:
            if np.all(depth_image[:, :, 0] == depth_image[:, :, 1]) and np.all(depth_image[:, :, 0] == depth_image[:, :, 2]):
                return depth_image[:, :, 0]
            print("Warning: Using first channel of 3-channel depth image")
            return depth_image[:, :, 0]
        elif depth_image.shape[2] == 4:
            print("Detected 4-channel (Zivid) depth image")
            return depth_image[:, :, 0]
        else:
            raise ValueError(f"Unsupported depth image channels: {depth_image.shape[2]}")
    elif len(depth_image.shape) == 2:
        return depth_image
    else:
        raise ValueError(f"Unsupported depth image shape: {depth_image.shape}")

def load_reference_model(reference_path: str, voxel_size: float = 0.001) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Load and preprocess reference point cloud model."""
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference object not found at {reference_path}")
    
    target_pcd = o3d.io.read_point_cloud(reference_path)
    if not target_pcd.has_points():
        raise ValueError("Reference point cloud is empty")
    
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    if not target_down.has_points():
        raise ValueError("Downsampled reference cloud is empty")
    
    return target_down, target_fpfh

def process_mask_instance(
    mask: np.ndarray, 
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    apply_mask: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single mask instance and prepare depth/RGB data."""
    # Resize mask if needed
    if mask.shape != depth_image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8) * 255,
            (depth_image.shape[1], depth_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    
    # Apply mask if requested
    if apply_mask:
        masked_depth = depth_image.copy()
        masked_depth[~mask] = 0
    else:
        masked_depth = depth_image.copy()
    
    if np.count_nonzero(masked_depth) == 0:
        raise ValueError("Mask resulted in empty depth map")
    
    return masked_depth, rgb_image.copy()

def register_and_calculate_pose(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    target_fpfh: np.ndarray,
    voxel_size: float,
    camera_extrinsics: np.ndarray
) -> np.ndarray:
    """Perform registration and calculate world pose."""
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    if not source_down.has_points():
        raise ValueError("Source point cloud is empty after downsampling")
    
    # Coarse registration
    coarse_result = execute_global_registration(
        source_down, target_pcd, source_fpfh, target_fpfh, voxel_size
    )
    
    # Fine registration
    icp_result = refine_registration_icp(
        source_down, target_pcd, coarse_result.transformation, 
        voxel_size, use_point_to_plane=True
    )
    
    if icp_result.fitness < 0.3:
        print(f"Warning: Low registration fitness: {icp_result.fitness:.2f}")
    
    T_camera_object = np.linalg.inv(icp_result.transformation)
    T_world_object = camera_extrinsics @ T_camera_object
    
    return T_world_object

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
    pcd_output_dir: Optional[str] = None,
    apply_mask: bool = False
) -> List[np.ndarray]:
    """Main function to generate waypoints from segmented objects."""
    # Initialize visualization
    if rerun_visualization:
        try:
            rr.init("generate_waypoints_demo", spawn=True)
        except Exception as e:
            print(f"Rerun initialization failed: {e}")
            rerun_visualization = False

    # Load and validate input data
    rgb_image, depth_image = load_and_validate_images(rgb_image_sam_path, depth_image_path)
    processed_depth = process_depth_image(depth_image)
    target_down, target_fpfh = load_reference_model(reference_object_path, voxel_size_registration)

    # Get segmentation masks
    all_masks = getSegmentationMasksSAM(
        rgb_image_path=rgb_image_sam_path,
        sam_server_url=sam_server_url,
        sam_query=sam_query,
        masks_input_dir=masks_input_dir,
        input_rgb_shape=rgb_image.shape[:2]
    )
    if not all_masks:
        print("No segmentation masks found")
        return []

    # Process each mask instance
    waypoints = []
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=depth_image.shape[1],
        height=depth_image.shape[0],
        fx=camera_intrinsics_k[0, 0],
        fy=camera_intrinsics_k[1, 1],
        cx=camera_intrinsics_k[0, 2],
        cy=camera_intrinsics_k[1, 2]
    )

    for i, mask in enumerate(all_masks):
        try:
            print(f"\n--- Processing Instance {i} ---")
            
            # Process mask and prepare data
            masked_depth, rgb_coloring = process_mask_instance(
                mask, processed_depth, rgb_image, apply_mask
            )
            scaled_depth = masked_depth.astype(np.float32) / depth_scale_to_meters

            # Create point cloud
            pcd = help.convert_rgbd_to_pointcloud(
                rgb_coloring, scaled_depth, o3d_intrinsics, np.identity(4)
            )
            if pcd is None or not pcd.has_points():
                print(f"Skipping instance {i} - empty point cloud")
                continue

            # Save point cloud if requested
            if pcd_output_dir:
                os.makedirs(pcd_output_dir, exist_ok=True)
                o3d.io.write_point_cloud(
                    os.path.join(pcd_output_dir, f"instance_{i}_point_cloud.ply"), 
                    pcd
                )

            # Register and calculate pose
            pose = register_and_calculate_pose(
                pcd, target_down, target_fpfh,
                voxel_size_registration, camera_extrinsics
            )
            waypoints.append(pose)

            # Visualization
            if rerun_visualization:
                visualize_instance(i, pcd, pose, target_down)

        except Exception as e:
            print(f"Error processing instance {i}: {str(e)}")
            continue

    print(f"\nGenerated {len(waypoints)} valid waypoints")
    return waypoints

def visualize_instance(
    instance_id: int,
    pcd: o3d.geometry.PointCloud,
    pose: np.ndarray,
    target_pcd: o3d.geometry.PointCloud
):
    """Visualize registration results using Rerun."""
    # Visualize point cloud
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
    rr.log(f"world/instance_{instance_id}/pcd", 
           rr.Points3D(positions=np.asarray(pcd.points), colors=colors))
    
    # Visualize pose
    rr.log(f"world/instance_{instance_id}/pose",
           rr.Transform3D(
               translation=pose[:3, 3],
               mat3x3=pose[:3, :3]
           ))
    
    # Visualize aligned model
    aligned_target = copy.deepcopy(target_pcd)
    aligned_target.transform(np.linalg.inv(pose))
    rr.log(f"world/instance_{instance_id}/aligned_target",
           rr.Points3D(positions=np.asarray(aligned_target.points)))