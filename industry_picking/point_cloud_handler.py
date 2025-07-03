import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation
import zivid.experimental
import zivid.experimental.calibration # For RPY conversion (pip install scipy)
import industry_picking.perception.pose_esimator as pest
import cv2
import time
import zivid
import os
from pathlib import Path 
import matplotlib.pyplot as plt
import open3d as o3d
from zivid import PointCloud
from typing import List, Optional, Tuple
import pyvista as pv
from industry_picking.cameras.camera import RealSense
from industry_picking.utils import helper_functions as help
from industry_picking.robots.xarm import Xarm
from scipy.spatial.transform import Rotation as R


# if __name__ == "__main__": 
#     arm = Xarm("192.168.1.184")
#     arm.connect()
#     cam = Camera(1224,1024)
#     robot_poses_to_visit = help.loadPosesFile("poses")
#     robot_poses = []
#     target_poses = []
#     #arm.connect()
#     #for i, curr_pose in enumerate(robot_poses_to_visit):
#     for i, curr_pose in enumerate(robot_poses_to_visit):
#         cam.connect()
#         camera_matrix,dist_coeffs = cam.getIntrinsics()
#         print(f"\n--- Processing Pose {i+1}/{len(robot_poses_to_visit)} ---")
#         arm.move(pose=curr_pose)
#         pose_rob = arm.getPose()
#         robot_poses.append(pose_rob)
#         frames = cam.captureImage()
#         pose_cam = cam.capturePose(image=frames)
#         target_poses.append(pose_cam)
        
#     print(f"Transformation matrix after 10th iteration.")
#     help.calibrateHandEye(target_poses=target_poses,robot_poses=robot_poses_to_visit)
            
if __name__ == "__main__":
    
    
    cam_fx = 1244.1572265625
    cam_fy = 1243.88977050781
    cam_cx = 617.272047802408
    cam_cy = 522.501989648454
    example_camera_K = np.array([
          [cam_fx, 0, cam_cx],
          [0, cam_fy, cam_cy],
          [0, 0,  1]
      ])
    
    arm = Xarm("192.168.1.184")
    arm.connect()
    cam = RealSense(width=1280,height=720)
    cam.connect()
    example_camera_K, _= cam.getIntrinsics()
    img , depth = cam.captureImage()
    help.save_image_to_file(image_data=img , file_path=r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\data\imgNova.png")
    help.save_image_to_file(image_data=depth , file_path=r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\data\depthNova.png")
    path_to_raw_rgb_for_sam =  r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\data\imgNova.png" # Original RGB for SAM
    path_to_raw_depth = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\data\depthNova.png" # Original Depth
    # --- DEBUG: Print depth image dtype and min/max ---
    depth_image_raw = cv2.imread(path_to_raw_depth, cv2.IMREAD_UNCHANGED)
    if depth_image_raw is not None:
        print("[DEBUG] Depth image dtype:", depth_image_raw.dtype)
        print("[DEBUG] Depth image min/max:", np.min(depth_image_raw), np.max(depth_image_raw))
    else:
        print("[DEBUG] Failed to load depth image for debug print.")
    path_to_reference_model = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\2xmanjiCC.ply" # Our best sample of an object we are picking

    example_camera_T_world_cam =  [[-0.06480675 , 0.99710445, -0.03978454 , 0.42953945],
                                    [ 0.99665811 , 0.06268809, -0.05237195 , -0.0752032 ],
                                    [-0.04972628 , -0.04304564, -0.99783484 , 0.4529416 ],
                                    [ 0.          , 0.          , 0.          , 1.        ]] #iz handeye kalibracije
    #example_camera_T_world_cam = np.identity(4) # For debugging, use identity matrix to simulate no transformation
    divisor = 1/  0.0010000000474974513


    generated_robot_waypoints = pest.generate_waypoints(
            rgb_image_sam_path=path_to_raw_rgb_for_sam,
            depth_image_path=path_to_raw_depth,
            camera_intrinsics_k=example_camera_K, # Use the K matrix for your specific camera and depth resolution
            camera_extrinsics=example_camera_T_world_cam,
            reference_object_path=path_to_reference_model,
            sam_server_url= "http://192.168.2.168:8090/sam2", # Your SAM server
            sam_query="Segment the circular grey metallic caps,1 instance at a time, in order", # Your SAM query
            voxel_size_registration=0.001, #0.001, # Adjust as needed
            depth_scale_to_meters=1000.0, # If depth values are in mm
            rerun_visualization=True,
            masks_output_dir= None,
            masks_input_dir=r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\uglovi\3maske", # Where to load from
            apply_mask=True
    )
    if generated_robot_waypoints:
        print(f"\n--- Generated {len(generated_robot_waypoints)} raw waypoints. Now filtering duplicates... ---")

        

        # Define how close two waypoints can be before being called a duplicate.
        MINIMUM_DISTANCE_BETWEEN_OBJECTS = 0.1 # 2 centimeters

        final_waypoints = pest.filter_duplicate_waypoints(
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
    # Move the robot arm to each final waypoint with a Z offset
    
    Z_OFFSET = -0.083  # meters (adjust as needed)
    for idx, wp in enumerate(final_waypoints):
        
        x, y, z, rot_x, rot_y, rot_z = wp[0], wp[1], wp[2]+0.1, wp[3], wp[4], wp[5] 
        # Convert Euler angles to rotation matrix (assuming 'xyz' or 'zyx' order)
        rotation = R.from_euler('xyz', [rot_x, rot_y, rot_z]).as_matrix()
        # Build 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [x, y, z]
        
        target_pose = transform.copy()
        arm.move(pose=target_pose)
        time.sleep(1)  # Optional: wait for motion to complete
    else:
        print("No waypoints were generated by the function.")
    time.sleep(60)
    
    
    
    
    
# def _point_cloud_to_cv_z(point_cloud: zivid.PointCloud) -> np.ndarray:
#     """Get depth map from frame.

#     Args:
#         point_cloud: Zivid point cloud

#     Returns:
#         depth_map_color_map: Depth map (HxWx1 ndarray)

#     """
#     depth_map = point_cloud.copy_data("z")
#     depth_map_uint8 = ((depth_map - np.nanmin(depth_map)) / (np.nanmax(depth_map) - np.nanmin(depth_map)) * 255).astype(
#         np.uint8
#     )

#     depth_map_color_map = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_VIRIDIS)

#     # Setting nans to black
#     depth_map_color_map[np.isnan(depth_map)[:, :]] = 0

#     return depth_map_color_map


# def _point_cloud_to_cv_bgr(point_cloud: zivid.PointCloud) -> np.ndarray:
#     """Get bgr image from frame.

#     Args:
#         point_cloud: Zivid point cloud

#     Returns:
#         bgr: BGR image (HxWx3 ndarray)

#     """
#     bgra = point_cloud.copy_data("bgra_srgb")

#     return bgra[:, :, :3]

# def display_bgr(bgr: np.ndarray, title: str = "RGB image") -> None:
#     """Display BGR image using OpenCV.

#     Args:
#         bgr: BGR image (HxWx3 ndarray)
#         title: Name of the OpenCV window

#     """
#     cv2.imshow(title, bgr)
#     print("Press any key to continue")
#     cv2.waitKey(0)

# def _visualize_and_save_image(image: np.ndarray, image_file: str, title: str) -> None:
#     """Visualize and save image to file.

#     Args:
#         image: BGR image (HxWx3 ndarray)
#         image_file: File name
#         title: OpenCV Window name

#     """
#     display_bgr(image, title)
#     cv2.imwrite(image_file, image)
# def _main() -> None:
#     with zivid.Application():
#         filename_zdf = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\uglovi\2.zdf"

#        #print(f"Reading {filename_zdf} point cloud")

#         frame = zivid.Frame(filename_zdf)
#         point_cloud = frame.point_cloud()
#         xyz = point_cloud.copy_data("xyz")  # Get all XYZ coordinates

#         print("Converting to BGR image in OpenCV format")
#         #bgr = _point_cloud_to_cv_bgr(point_cloud)

#         bgr_image_file = "ImageRGB.png"
#         print(f"Visualizing and saving BGR image to file: {bgr_image_file}")
#         #_visualize_and_save_image(bgr, bgr_image_file, "BGR image")

#         print("Converting to Depth map in OpenCV format")
#         #z_color_map = _point_cloud_to_cv_z(point_cloud)

#         depth_map_file = "DepthMap.png"
#         print(f"Visualizing and saving Depth map to file: {depth_map_file}")
#         #_visualize_and_save_image(z_color_map, depth_map_file, "Depth map")
#         depth_map = xyz[:, :, 2]  # Z channel contains depth

#         # Save as 32-bit TIFF (preserves full precision)
#         cv2.imwrite("depth_32bit.tiff", depth_map)



# if __name__ == "__main__":
#     _main()

# import open3d as o3d
# import numpy as np
# from matplotlib import pyplot as plt

# def compare_point_clouds(cloud1_path, cloud2_path, voxel_size=0.001):
#     """Compare all key characteristics of two point clouds."""
#     # Load clouds
#     cloud1 = o3d.io.read_point_cloud(cloud1_path)
#     cloud2 = o3d.io.read_point_cloud(cloud2_path)
    
#     def normalize_scale(pcd, target_scale):
#         """Rescale point cloud to match target scale"""
#         points = np.asarray(pcd.points)
#         current_scale = np.max(points) - np.min(points)
#         scale_factor = target_scale / current_scale
#         pcd.scale(scale_factor, center=pcd.get_center())
#         return pcd
    
#     for pcd in [cloud1, cloud2]:
#         pcd.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(
#             radius=0.05,  # Adjust based on point spacing
#             max_nn=30
#         )
#     )
#     cloud1 = normalize_scale(cloud1, target_scale=0.1)  # ~cloud2's max extent

#     # Orient normals consistently
#     pcd.orient_normals_to_align_with_direction()

#     if np.all(np.asarray(cloud1.normals)[:, 2] == 1):  # If all Z-up
#         print("Warning: cloud1 appears to be a 2D plane - add artificial noise")
#     points = np.asarray(cloud1.points)
#     points += np.random.normal(0, 0.001, points.shape)  # Add minor noise
#     cloud1.points = o3d.utility.Vector3dVector(points)
#     cloud1.estimate_normals()  # Recompute normals    
#     cloud1.scale(0.5, center=cloud1.get_center())
  
    
#     ###################END OF NORMALS AND NOISE HANDLING###################
#     # Rescale cloud1 to match cloud2's scale (or vice versa)
#     o3d.io.write_point_cloud("2xmanji.ply", cloud1)  # PLY format
#     # Basic validation
#     cloud1 = cloud1.remove_non_finite_points()
#     cloud2 = cloud2.remove_non_finite_points()
    
#     # Initialize comparison dict
#     comparison = {
#         'basic_stats': {},
#         'normal_stats': {},
#         'feature_stats': {},
#         'overlap_analysis': None,
#         'visualizations': []
#     }
    
#     # 1. Basic statistics comparison
#     def get_basic_stats(pcd, name):
#         points = np.asarray(pcd.points)
#         nn_distances = np.asarray(pcd.compute_nearest_neighbor_distance())
#         return {
#         f'{name}_point_count': len(points),
#         f'{name}_extent': np.ptp(points, axis=0),  # Peak-to-peak (max-min) per axis
#         f'{name}_mean_spacing': float(np.mean(nn_distances)),  # Explicit conversion
#         f'{name}_has_normals': pcd.has_normals(),
#         f'{name}_has_colors': pcd.has_colors(),
#         f'{name}_nan_count': len(pcd.points) - len(points)
#     }
    
#     comparison['basic_stats'].update(get_basic_stats(cloud1, 'cloud1'))
#     comparison['basic_stats'].update(get_basic_stats(cloud2, 'cloud2'))
    
#     # 2. Normal vector analysis
#     def analyze_normals(pcd, name):
#         if not pcd.has_normals():
#             pcd.estimate_normals(
#                 o3d.geometry.KDTreeSearchParamHybrid(
#                     radius=voxel_size*2, 
#                     max_nn=30
#                 )
#             )
#         normals = np.asarray(pcd.normals)
#         return {
#             f'{name}_normal_consistency': np.mean(np.linalg.norm(normals, axis=1)),
#             f'{name}_normal_angle_var': np.var(np.arccos(normals[:, 2]))  # Variance from Z-axis
#         }
    
#     comparison['normal_stats'].update(analyze_normals(cloud1, 'cloud1'))
#     comparison['normal_stats'].update(analyze_normals(cloud2, 'cloud2'))
    
#     # 3. Feature quality comparison
#     def analyze_features(pcd, name):
#         if not pcd.has_normals():
#             pcd.estimate_normals()
#         fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#             pcd,
#             o3d.geometry.KDTreeSearchParamHybrid(
#                 radius=voxel_size*5,
#                 max_nn=100
#             )
#         )
#         uniqueness = np.mean(np.std(fpfh.data, axis=1))
#         return {
#             f'{name}_feature_uniqueness': uniqueness,
#             f'{name}_feature_dim': fpfh.data.shape[1]
#         }
    
#     comparison['feature_stats'].update(analyze_features(cloud1, 'cloud1'))
#     comparison['feature_stats'].update(analyze_features(cloud2, 'cloud2'))
    
#     # 4. Overlap analysis (requires downsampling)
#     cloud1_down = cloud1.voxel_down_sample(voxel_size)
#     cloud2_down = cloud2.voxel_down_sample(voxel_size)
    
#     # Compute rough overlap using nearest neighbors
#     def estimate_overlap(source, target):
#         dists = source.compute_point_cloud_distance(target)
#         overlap_ratio = np.mean(np.asarray(dists) < voxel_size * 1.5)
#         return overlap_ratio
    
#     comparison['overlap_analysis'] = {
#         'cloud1_to_cloud2': estimate_overlap(cloud1_down, cloud2_down),
#         'cloud2_to_cloud1': estimate_overlap(cloud2_down, cloud1_down),
#         'avg_overlap': (estimate_overlap(cloud1_down, cloud2_down) + 
#                         estimate_overlap(cloud2_down, cloud1_down)) / 2
#     }
    
#     # 5. Generate visualizations
#     def create_visualization(pcd1, pcd2, title):
#         pcd1.paint_uniform_color([1, 0, 0])  # Red
#         pcd2.paint_uniform_color([0, 1, 0])  # Green
#         vis = o3d.visualization.Visualizer()
#         vis.create_window()
#         vis.add_geometry(pcd1)
#         vis.add_geometry(pcd2)
#         vis.get_render_option().point_size = 3
#         vis.run()  # User must close window to continue
#         vis.destroy_window()
    
#     # Save visualization to list
#     comparison['visualizations'].append(
#         create_visualization(cloud1_down, cloud2_down, "Downsampled Comparison")
#     )
    
#     # 6. Format comparison report
#     def generate_report(comparison):
#         report = "=== POINT CLOUD COMPARISON REPORT ===\n"
        
#         # Basic stats
#         report += "\n[BASIC STATISTICS]\n"
#         for k, v in comparison['basic_stats'].items():
#             report += f"{k:30}: {v}\n"
            
#         # Normal stats
#         report += "\n[NORMAL VECTOR ANALYSIS]\n"
#         for k, v in comparison['normal_stats'].items():
#             report += f"{k:30}: {v:.4f}\n"
            
#         # Feature stats
#         report += "\n[FEATURE QUALITY]\n"
#         for k, v in comparison['feature_stats'].items():
#             report += f"{k:30}: {v:.4f}\n"
            
#         # Overlap
#         report += "\n[OVERLAP ANALYSIS]\n"
#         for k, v in comparison['overlap_analysis'].items():
#             report += f"{k:30}: {v:.2%}\n"
            
#         return report
    
#     comparison['report'] = generate_report(comparison)
    
#     return comparison

# # Usage example:
# result = compare_point_clouds( r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\uglovi\noviSample.xyz",
#                               r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\testSample\instance_2_point_cloud.ply")
# print(result['report'])

# # To view visualizations (blocks until window is closed)
# for vis in result['visualizations']:
#     pass  # Visualization windows already shown during creation
