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

# polinomalni koeficijenti look it up!
rr.init("rerun_demo", spawn=True) #inicijalizacija reruna


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
#funkcija koja na osnovu rgb i depth slike i unutrasnjih i spljnjih parametara generise waypointe za robot
# TODO napraviti instancu klase 
def generate_waypoints(
    
        rgb_image_sam_path : str, #path do slike koju dajemo SAMu da segmentuje 
        depth_image_path : str, #path do depth slike sa koje cemo da dobijamo point cloudove
        camera_intrinsics_k : np.ndarray, #unutrasnji parametri kamere
        camera_extrinsics : np.ndarray, #spoljasnji parametri kamere
        reference_object_path : str, #path do ref objekta kog cemo da skeniramo vise puta da bi imali sto bolji ply sample
        sam_server_url : str = "http://192.168.2.168:3001/sam", #url SAM servera
        sam_query : str = "Segment object,1 instance at a time, in order", #promt modelu sta da segmentuje
        voxel_size_registration : float = 0.001, #istrazi                                                                       TO DO
        depth_scale_to_meters : float = 1000.0, #skaliranje iz milimetara u metre, da zivid daje cm bilo bi 100.0 ? 
        rerun_visualization : bool = False, #ukljucivanje/iskljucivanje rerun-a
        masks_output_dir: Optional[str] = None
                    )->List[np.ndarray]: 
    waypoints = []
    #rerun on/off
    if rerun_visualization:
            try:
                rr.init("generate_waypoints_s ση מdemo", spawn=True) # Ensure unique app ID
              ##  rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True) # Example: Z up for world
            except Exception as e:
                print(f"Rerun initialization failed: {e}. Continuing without Rerun visualization in this function.")
                rerun_visualization = False
    #korak 1: ucitavanje rgb slike za SAM2 
    input_rgb_for_sam = cv2.imread(rgb_image_sam_path)
    if input_rgb_for_sam is None:
        print(f"Error: Could not read input_rgb_for_sam: {rgb_image_sam_path}")
        return waypoints
    if rerun_visualization:
        rr.log("world/input_images/rgb_for_sam", rr.Image(cv2.cvtColor(input_rgb_for_sam, cv2.COLOR_BGR2RGB)))
    #korak 2: pozovemo SAM2 da bi dobili segmentirane slike (ili YOLO najvjerovatnije u buducnosti)
    #pored slanja slike i citanja segmentirane slike, provjerava se da li je format tj rezolucija dobijenih slika
    #jednaka rezoluciji depth slike na koju ce se maska primjenjivati
    try:
        response = requests.post(
            sam_server_url,
            files={"image": cv2.imencode(".jpg", input_rgb_for_sam)[1].tobytes()},
            data={"query": sam_query},
            timeout=30 # Added timeout
        )
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"SAM connection failed: {e}")
        return waypoints
    
    decoded_response_image_stack = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
    #provjera da li je SAM vratio uopste sliku tj da li smo je ucitali
    if decoded_response_image_stack is None:
        print("Error: Could not decode image received from SAM server.")
        return waypoints
    H_orig, W_orig = input_rgb_for_sam.shape[:2]
    H_response, W_response = decoded_response_image_stack.shape[:2]
    #provjerava rezoluciju
    if W_response != W_orig:
        print(f"Warning: SAM Response image width ({W_response}) does not match input image width ({W_orig}). This might affect mask extraction if not a stacked image.")
        # Consider resizing decoded_response_image_stack width or erroring if critical

    num_full_segments = H_response // H_orig
    if num_full_segments <= 1: # Expect at least annotations + 1 mask
        print(f"Warning: SAM Response seems to contain only {num_full_segments-1} mask(s) (or only annotations). Check SAM query and output.")
        # If num_full_segments is 0 or 1, it means no actual masks might be present beyond annotations.
        if num_full_segments == 0: return waypoints
    #korak 3: loadujemo depth sliku koja ce sluziti za generisanje point clouda
    depth_image_raw = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    #provjerava da li je loadovao sliku
    if depth_image_raw is None:
        print(f"Failed to load ORIGINAL depth image from: {depth_image_path}.")
        return waypoints
    #provjerava da je depth slika 2d niz i na kom kanalu se nalaze depth podaci
    if len(depth_image_raw.shape) == 3 and depth_image_raw.shape[2] >= 1:
        print(f"Depth image has {depth_image_raw.shape[2]} channels. Using the first channel.")
        processed_depth_image_original_resolution = depth_image_raw[:,:,0]
    elif len(depth_image_raw.shape) == 2:
        processed_depth_image_original_resolution = depth_image_raw
    else:
        print(f"Error: Original depth image has an unsupported shape: {depth_image_raw.shape}.")
        return waypoints
    #loadujemo ref sample sa kojim poredimo sve ostale point cloudove
    if not os.path.exists(reference_object_path):
        print(f"Error: Reference object file not found at '{reference_object_path}'.")
        return waypoints
    target_pcd_reference = o3d.io.read_point_cloud(reference_object_path)
    if not target_pcd_reference.has_points():
        print("Error: Target reference point cloud is empty or failed to load.")
        return waypoints
    #preprocessovanje point clouda target objekta
    #voxel downsampling->smanjivanje broja tacaka zbog lakse obrade i smanjivanja suma, rezultat je target_down
    #fpfh fast point feature histogram opisuju lokalnu geometriju oko svake tacke i koriste se za matchovanje objekta sa targetom
    target_down, target_fpfh = preprocess_point_cloud(target_pcd_reference, voxel_size_registration)
    if not target_down.has_points():
        print("Error: Downsampling target reference cloud resulted in an empty cloud. Adjust voxel_size.")
        return waypoints
    if rerun_visualization:
        rr.log("world/reference_model/target_downsampled", rr.Points3D(positions=np.asarray(target_down.points), colors = [0,0,255] if not target_down.has_colors() else None))
    
    #korak 5: prolazenje kroz svaku masku koju je SAM (ili YOLO) vratio, maske krecu od indexa 1
    for i in range(1, num_full_segments): #num_full_segments
        instance_index = i - 1 # 0-indexed masks
        print(f"\n--- Processing Mask Instance {instance_index} ---")
        start_row = i * H_orig
        end_row = (i + 1) * H_orig     
        sam_instance_visual_bgr = decoded_response_image_stack[start_row:end_row, :, :]

        if len(sam_instance_visual_bgr.shape) == 3: # Color mask
            binary_mask_from_sam = np.any(sam_instance_visual_bgr > 10, axis=2) # True for non-black pixels
        else: # Grayscale mask
            binary_mask_from_sam = sam_instance_visual_bgr > 10 # True for non-background pixels

        if binary_mask_from_sam.shape != processed_depth_image_original_resolution.shape[:2]:
            print(f"Resizing SAM binary mask from {binary_mask_from_sam.shape} to depth image shape {processed_depth_image_original_resolution.shape[:2]}")
            binary_mask_from_sam_resized = cv2.resize(
                binary_mask_from_sam.astype(np.uint8) * 255, # Convert boolean to uint8 image for resize
                (processed_depth_image_original_resolution.shape[1], processed_depth_image_original_resolution.shape[0]), # (W, H)
                interpolation=cv2.INTER_NEAREST # Nearest neighbor for binary masks
            ).astype(bool) # Convert back to boolean
        else:
            binary_mask_from_sam_resized = binary_mask_from_sam

        if masks_output_dir:
            # Construct a unique filename for the mask
            mask_filename = f"mask_instance_{instance_index}.png"
            full_mask_path = os.path.join(masks_output_dir, mask_filename)
            
            # Call the save function
            save_mask(binary_mask_from_sam_resized, full_mask_path)
        
        if rerun_visualization:
             rr.log(f"world/instance_{instance_index}/sam_binary_mask", rr.SegmentationImage(binary_mask_from_sam_resized.astype(np.uint8)))

         # Mask the original RGB image using the SAM binary mask for point cloud coloring
        # Ensure input_rgb_for_sam is also resized to match depth if its original dimensions differ
        # (This assumes camera_intrinsics_K are for the processed_depth_image_original_resolution)
        rgb_for_pcd_coloring = input_rgb_for_sam.copy()
        if rgb_for_pcd_coloring.shape[:2] != processed_depth_image_original_resolution.shape[:2]:
            rgb_for_pcd_coloring = cv2.resize(
                rgb_for_pcd_coloring,
                (processed_depth_image_original_resolution.shape[1], processed_depth_image_original_resolution.shape[0]),
                interpolation=cv2.INTER_AREA
            )
        
        # Apply mask: set pixels outside the mask to black [0,0,0] in the RGB image
        rgb_for_pcd_coloring[~binary_mask_from_sam_resized] = [0, 0, 0]
        if rerun_visualization:
             rr.log(f"world/instance_{instance_index}/rgb_masked_for_pcd", rr.Image(cv2.cvtColor(rgb_for_pcd_coloring, cv2.COLOR_BGR2RGB)))


        # Apply the binary mask to the original resolution depth image
        depth_image_masked_instance = processed_depth_image_original_resolution.copy()
        depth_image_masked_instance[~binary_mask_from_sam_resized] = 0 # Set depth to 0 outside mask
        
        if np.count_nonzero(depth_image_masked_instance) == 0:
            print(f"Warning: Mask instance {instance_index} resulted in an empty depth map after masking. Skipping.")
            continue
        if rerun_visualization:
             rr.log(f"world/instance_{instance_index}/depth_masked", rr.DepthImage(depth_image_masked_instance.astype(np.float32), meter=depth_scale_to_meters))


        # Prepare depth for Open3D (convert to float32 and scale to meters)
        depth_for_o3d_instance = depth_image_masked_instance.astype(np.float32)
        depth_image_scaled_to_meters_instance = depth_for_o3d_instance / depth_scale_to_meters
        
        # Define O3D Camera Intrinsics using the provided K matrix and current image dimensions
        # (assuming K is for the resolution of processed_depth_image_original_resolution)
        image_height, image_width = depth_image_scaled_to_meters_instance.shape[:2]
        o3d_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            image_width, image_height,
            camera_intrinsics_k[0,0], # fx
            camera_intrinsics_k[1,1], # fy
            camera_intrinsics_k[0,2], # cx
            camera_intrinsics_k[1,2]  # cy
        )
        
        # Generate Point Cloud for this instance (in camera frame)
        pcd_instance_camera_frame = convert_rgbd_to_pointcloud(
            rgb_for_pcd_coloring, # Masked original RGB
            depth_image_scaled_to_meters_instance, # Masked and scaled depth
            o3d_camera_intrinsics,
            np.identity(4) # Point cloud generated in camera coordinate system
        )

        if pcd_instance_camera_frame is None or not pcd_instance_camera_frame.has_points():
            print(f"Failed to generate point cloud for instance {instance_index}. Skipping.")
            continue
        if rerun_visualization:
            colors_for_rr = (np.asarray(pcd_instance_camera_frame.colors)*255).astype(np.uint8) if pcd_instance_camera_frame.has_colors() else None
            rr.log(f"world/instance_{instance_index}/segmented_pcd_camera_frame", rr.Points3D(positions=np.asarray(pcd_instance_camera_frame.points), colors=colors_for_rr))


        # Perform Registration against the reference model
        source_down, source_fpfh = preprocess_point_cloud(pcd_instance_camera_frame, voxel_size_registration)
        if not source_down.has_points():
            print(f"Downsampling segmented cloud for instance {instance_index} resulted in empty cloud. Skipping.")
            continue

        coarse_reg_result = execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size_registration
        )
        if coarse_reg_result.fitness < 0.05: # Basic check for coarse alignment quality
             print(f"Warning: Low fitness ({coarse_reg_result.fitness:.4f}) for coarse registration on instance {instance_index}.")
        
        icp_reg_result = refine_registration_icp(
            source_down, target_down, coarse_reg_result.transformation, voxel_size_registration, use_point_to_plane=True
        )
        
        if icp_reg_result.fitness < 0.3: # Stricter check for ICP success
            print(f"Warning: Low fitness ({icp_reg_result.fitness:.4f}) after ICP for instance {instance_index}. Waypoint may be inaccurate or object not found.")
            # continue # Optionally skip if ICP is too bad, or log as failed attempt

        # final_transformation_source_to_target is T_target_model <- source_camera
        # This transforms points from the source (camera) frame to the target (model) frame.
        T_target_source = icp_reg_result.transformation

        # To get the pose of the object (defined by the target model's frame) in the camera's coordinate system:
        # T_camera_object = (T_target_model <- source_camera)^-1 = T_source_camera <- target_model
        try:
            T_camera_object = np.linalg.inv(T_target_source)
        except np.linalg.LinAlgError:
            print(f"Error: Could not compute inverse of registration matrix for instance {instance_index}. Skipping.")
            continue
            
        # Transform object pose from Camera Frame to World Frame
        # T_world_object = T_world_camera * T_camera_object
        T_world_object = camera_extrinsics @ T_camera_object

        # Extract XYZ position from the world frame pose
        xyz_world = T_world_object[0:3, 3]
        
        # Extract Rotation Matrix from the world frame pose and convert to Roll, Pitch, Yaw
        rotation_matrix_world = T_world_object[0:3, 0:3]
        rpy_world = rotation_matrix_to_rpy(rotation_matrix_world) # [roll, pitch, yaw] in radians

        waypoint = np.concatenate((xyz_world, rpy_world))
        waypoints.append(waypoint)

        print(f"Instance {instance_index}: Waypoint in World Frame: XYZ=[{xyz_world[0]:.3f}, {xyz_world[1]:.3f}, {xyz_world[2]:.3f}], RPY(xyz)=[{rpy_world[0]:.3f}, {rpy_world[1]:.3f}, {rpy_world[2]:.3f}] rad")
            
    # After the loop, if waypoints were generated and visualization is on:
    if not rerun_visualization and waypoints:
        # Extract only the XYZ coordinates from the waypoints list
        path_positions = [wp[:3] for wp in waypoints]

        # Log the waypoints as a connected path, grouping it with the waypoint poses
        rr.log(
            "world/waypoints/path",  # Log the path as a child of 'waypoints'
            rr.LineStrips3D(
                [path_positions],
                colors=[255, 215, 0],
                radii=0.002
            )
        )
        # We no longer need to log the individual yellow points.
        # The Transform3D logged in the loop serves as a much better marker.

    if not waypoints:
        print("No waypoints generated.")
    else:
        print(f"\nSuccessfully generated {len(waypoints)} waypoint(s).")
    return waypoints
        
   
if __name__ == "__main__":
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

    path_to_raw_rgb_for_sam = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\5trid.png" # Original RGB for SAM
    path_to_raw_depth = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\2depthmap.png" # Original Depth
    path_to_reference_model = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\ref_sample_true.ply" # Our best sample of an object we are picking

    generated_robot_waypoints = generate_waypoints(
            rgb_image_sam_path=path_to_raw_rgb_for_sam,
            depth_image_path=path_to_raw_depth,
            camera_intrinsics_k=example_camera_K, # Use the K matrix for your specific camera and depth resolution
            camera_extrinsics=example_camera_T_world_cam,
            reference_object_path=path_to_reference_model,
            sam_server_url="http://109.245.66.46:3001/sam", # Your SAM server
            sam_query="Segment the circular grey metallic caps,1 instance at a time, in order", # Your SAM query
            voxel_size_registration=0.001, # Adjust as needed
            depth_scale_to_meters=1000.0, # If depth values are in mm
            rerun_visualization=True,
            masks_output_dir= r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\mask5"
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
