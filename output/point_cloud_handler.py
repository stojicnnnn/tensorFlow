import open3d as o3d
import numpy as np
import cv2 # For loading images (pip install opencv-python)
import os # For os.path.exists if you use it, though not in current snippet
import copy 

# --- Main part: How to prepare inputs and call the function ---
def convert_rgbd_to_pointcloud(rgb, depth, intrinsic_matrix, extrinsic=None):
    if rgb is None or depth is None:
        print("RGB or depth image is None")
        return None

    if rgb.shape[:2] != depth.shape[:2]:
        print(f"RGB shape {rgb.shape[:2]} and depth shape {depth.shape[:2]} have different sizes")
        return None

    rgb_converted_to_rgb_format = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    o3d_color_image = o3d.geometry.Image(rgb_converted_to_rgb_format)
    o3d_depth_image = o3d.geometry.Image(depth) 

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color_image,
        o3d_depth_image,
        convert_rgb_to_intensity=False, 
        depth_scale=1.0, 
        depth_trunc=1000.0, 
    )

    if not isinstance(intrinsic_matrix, o3d.camera.PinholeCameraIntrinsic):
        print("Error: intrinsic_matrix must be an o3d.camera.PinholeCameraIntrinsic object.")
        return None

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic_matrix, extrinsic 
    )
    return point_cloud

if __name__ == "__main__":
    # 1. Load your RGB and Depth images
    # path_to_rgb should be your MODIFIED RGB image (with blacked-out parts)
    path_to_rgb = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\2.png" # CHANGE TO YOUR MODIFIED RGB
    path_to_depth = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\2depthmap.png" # ORIGINAL, UNMODIFIED DEPTH MAP

    try:
        rgb_image_modified_bgr_prev = cv2.imread(path_to_rgb, cv2.IMREAD_COLOR)
        depth_image_raw = cv2.imread(path_to_depth, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error loading images: {e}")
        exit()

    if rgb_image_modified_bgr_prev is None:
        print(f"Failed to load MODIFIED RGB image from: {path_to_rgb}. Check path and file.")
        exit()
    if depth_image_raw is None:
        print(f"Failed to load ORIGINAL depth image from: {path_to_depth}. Check path and file.")
        exit()

    print(f"DEBUG: Shape of MODIFIED rgb_image_bgr: {rgb_image_modified_bgr_prev.shape}, dtype: {rgb_image_modified_bgr_prev.dtype}")
    print(f"DEBUG: Shape of depth_image_raw just before checks: {depth_image_raw.shape}, dtype: {depth_image_raw.dtype}")
    rgb_image_modified_bgr = cv2.resize(
        rgb_image_modified_bgr_prev, 
        (1224, 1024), 
        interpolation=cv2.INTER_AREA # INTER_AREA is good for downscaling
    )
    print(f"DEBUG: Shape of MODIFIED rgb AFTER RESIZE!!: {rgb_image_modified_bgr.shape}, dtype: {rgb_image_modified_bgr.dtype}")
    print(f"DEBUG: Shape of depth_image_raw just before checks: {depth_image_raw.shape}, dtype: {depth_image_raw.dtype}")

    # --- Process original depth image shape (as before) ---
    processed_depth_image_original = None # This will be the full depth map initially
    # (Insert your robust depth image shape handling block here to get processed_depth_image_original as a 2D array)
    # For brevity, I'll assume it's done and results in 'processed_depth_image_original'
    # For example, if it was 4-channel:
    if len(depth_image_raw.shape) == 3 and depth_image_raw.shape[2] == 4:
        print("Assuming first channel of 4-channel raw depth is depth data.")
        processed_depth_image_original = depth_image_raw[:,:,0]
    elif len(depth_image_raw.shape) == 2:
        processed_depth_image_original = depth_image_raw
    else:
        print("Error: Original depth image not 2D or expected 4-channel. Please fix shape handling.")
        exit()
    
    if processed_depth_image_original is None or len(processed_depth_image_original.shape) != 2:
        print("Error: processed_depth_image_original is not a 2D array.")
        exit()
    print(f"DEBUG: processed_depth_image_original shape: {processed_depth_image_original.shape}")

    # --- Create a mask from your MODIFIED RGB image ---
    # Pixels are considered "object" if they are not pure black (0,0,0).
    # You might need to adjust the threshold slightly if Paint doesn't save perfect black.
    if len(rgb_image_modified_bgr.shape) == 3:
        # Summing channels: if sum > a small threshold, it's not black.
        # A more robust way for "not black" might be to check if any channel value > threshold
        object_mask_from_rgb = np.any(rgb_image_modified_bgr > 10, axis=2) # True for non-black pixels
    else: # Assuming grayscale modified image if not 3 channels
        object_mask_from_rgb = rgb_image_modified_bgr > 10
    print(f"DEBUG: object_mask_from_rgb shape: {object_mask_from_rgb.shape}")

    # Ensure the mask and depth image have the same H, W dimensions before masking
    if object_mask_from_rgb.shape != processed_depth_image_original.shape:
        print(f"Warning: Mask shape {object_mask_from_rgb.shape} and original depth shape {processed_depth_image_original.shape} differ.")
        print("         Attempting to resize mask to depth image dimensions. Verify results.")
        # Resize mask to match depth image dimensions.
        # cv2.resize expects (width, height)
        object_mask_from_rgb_resized = cv2.resize(
            object_mask_from_rgb.astype(np.uint8) * 255, # Convert boolean to uint8 for resize
            (processed_depth_image_original.shape[1], processed_depth_image_original.shape[0]),
            interpolation=cv2.INTER_NEAREST # Use nearest neighbor to keep it binary
        ).astype(bool) # Convert back to boolean
        if object_mask_from_rgb_resized.shape != processed_depth_image_original.shape:
            print("Error: Mask resize failed to match depth dimensions. Exiting.")
            exit()
        object_mask_to_apply = object_mask_from_rgb_resized
    else:
        object_mask_to_apply = object_mask_from_rgb
    print(f"DEBUG: object_mask_to_apply shape: {object_mask_to_apply.shape}")


    # --- Apply the mask to the depth image ---
    # Create a copy of the original processed depth to modify
    depth_image_masked = processed_depth_image_original.copy()
    # Where the RGB mask says it's background (False), set depth to 0
    depth_image_masked[~object_mask_to_apply] = 0 
    print(f"DEBUG: Applied RGB mask to depth image. Non-zero depth values: {np.count_nonzero(depth_image_masked)}")

    depth_image_final_for_scaling = depth_image_masked # Use this for scaling

    # Now use 'depth_image_final_for_scaling' for the rest of the processing
    image_height, image_width = depth_image_final_for_scaling.shape[:2]
    print(f"DEBUG: image_height={image_height}, image_width={image_width} (from masked depth)")

    # 2. Define Camera Intrinsics (CRITICAL: Use YOUR camera's actual values)
    print("DEBUG: Defining camera intrinsics...")
    fx = 525.0  # Focal length x (in pixels) - REPLACE
    fy = 525.0  # Focal length y (in pixels) - REPLACE
    cx = image_width / 2.0   # Principal point x 
    cy = image_height / 2.0  # Principal point y 

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        int(image_width), int(image_height), fx, fy, cx, cy
    )
    print(f"DEBUG: Camera intrinsics defined: \n{camera_intrinsics.intrinsic_matrix}")

    # 3. Prepare Depth Data (Scaling to Meters if necessary)
    print("DEBUG: Preparing depth data for scaling...")
    depth_values_are_in_millimeters = True # SET THIS ACCORDING TO YOUR DEPTH IMAGE UNITS
    
    if depth_image_final_for_scaling.dtype != np.float32:
        depth_for_o3d = depth_image_final_for_scaling.astype(np.float32)
    else:
        depth_for_o3d = depth_image_final_for_scaling

    if depth_values_are_in_millimeters:
        depth_image_scaled_to_meters = depth_for_o3d / 1000.0
    else:
        depth_image_scaled_to_meters = depth_for_o3d
    print(f"DEBUG: Masked depth image scaled. Min: {np.min(depth_image_scaled_to_meters)}, Max: {np.max(depth_image_scaled_to_meters)}")
    
    # 4. Define Camera Extrinsics (Optional)
    extrinsic_matrix = np.identity(4)
    print("DEBUG: Extrinsic matrix defined (identity).")

    # 5. Call your function
    print("DEBUG: Calling convert_rgbd_to_pointcloud function...")
    # Use your MODIFIED RGB image and the MASKED (and scaled) depth image
    pcd = convert_rgbd_to_pointcloud(
        rgb_image_modified_bgr, 
        depth_image_scaled_to_meters,
        camera_intrinsics,
        extrinsic_matrix
    )
    print("DEBUG: Returned from convert_rgbd_to_pointcloud function.")

    # 6. Use the resulting point cloud
    print("DEBUG: Checking the generated point cloud 'pcd' object...")
    # (Insert the detailed pcd checking and visualization block from previous response here)
    if pcd is not None:
        print(f"  DEBUG: 'pcd' object is not None. Type: {type(pcd)}")
        if pcd.has_points():
            num_points = len(pcd.points)
            print(f"  Successfully generated point cloud with {num_points} points.")
            # ... (rest of the print and visualization code)
            output_filename_ply = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\ref_sample.ply"
            o3d.io.write_point_cloud(output_filename_ply, pcd , write_ascii=False)
            o3d.visualization.draw_geometries([pcd], window_name="Segmented Point Cloud from RGB-D")
            
        else:
            print("  'pcd' object was generated BUT HAS NO POINTS.")
    else:
        print("  'pcd' object is None.")