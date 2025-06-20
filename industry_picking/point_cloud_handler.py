import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation # For RPY conversion (pip install scipy)
import industry_picking.perception.pose_esimator as pest
import pyrealsense2 as rs
import os
import open3d as o3d
import copy
import time

if __name__ == '__main__':
    # --- Setup Rerun and Dummy Data ---
    rr.init("object_model_fusion_demo", spawn=True)
    
    SCAN_DIR = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\testSample"
    VOXEL_SIZE = 0.002 # 5mm - This is the most important parameter to tune!

    # Create dummy data if it doesn't exist
    if not os.path.exists(SCAN_DIR):
        print("Creating dummy scan data...")
        os.makedirs(SCAN_DIR)
        # Create a base sphere
        base_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        base_mesh.compute_vertex_normals()
        
        # Generate several "scans" by rotating and sampling the sphere
        for i in range(4):
            rotation = o3d.geometry.get_rotation_matrix_from_xyz((
                np.random.rand() * 0.8,
                np.random.rand() * 0.8,
                i * np.pi / 2
            ))
            scan_mesh = copy.deepcopy(base_mesh).rotate(rotation, center=(0,0,0))
            pcd = scan_mesh.sample_points_uniformly(number_of_points=5000)
            pcd.paint_uniform_color(np.random.rand(3)) # Color each scan differently
            o3d.io.write_point_cloud(os.path.join(SCAN_DIR, f"scan_{i}.ply"), pcd)

    # --- Run the Fusion ---
    final_fused_pcd = pest.fuse_multiple_scans(
        scan_directory=SCAN_DIR,
        voxel_size=VOXEL_SIZE,
        rerun_entity_path="world/fused_object"
    )
    time.sleep(60)
        
        
    
if __name__ == "__main__1":
    
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

    generated_robot_waypoints = pest.generate_waypoints(
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

    else:
        print("No waypoints were generated by the function.")