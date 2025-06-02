import open3d as o3d
import numpy as np
import cv2 # For loading images (pip install opencv-python)

# Your provided function (I've made a slight adjustment to use the converted RGB)
def convert_rgbd_to_pointcloud(rgb, depth, intrinsic_matrix, extrinsic=None):
    if rgb is None or depth is None:
        print("RGB or depth image is None")
        return None

    if rgb.shape[:2] != depth.shape[:2]:
        print(f"RGB shape {rgb.shape[:2]} and depth shape {depth.shape[:2]} have different sizes")
        return None

    # Assuming input 'rgb' is BGR from cv2.imread.
    # If your input is already RGB, you might adjust this.
    rgb_converted_to_rgb_format = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    o3d_color_image = o3d.geometry.Image(rgb_converted_to_rgb_format)
    o3d_depth_image = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color_image,
        o3d_depth_image,
        convert_rgb_to_intensity=False, # Keep it as color
        depth_scale=1.0,  # IMPORTANT: Assumes input 'depth' array is already in target units (e.g., meters)
        depth_trunc=1000.0, # IMPORTANT: Truncates depth values beyond this (in target units)
    )

    # Ensure intrinsic_matrix is an Open3D PinholeCameraIntrinsic object
    if not isinstance(intrinsic_matrix, o3d.camera.PinholeCameraIntrinsic):
        print("Error: intrinsic_matrix must be an o3d.camera.PinholeCameraIntrinsic object.")
        return None

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic_matrix, extrinsic # extrinsic is optional
    )
    return point_cloud

# --- Main part: How to prepare inputs and call the function ---
if __name__ == "__main__":
    # 1. Load your RGB and Depth images
    # Replace with the actual paths to your images
    path_to_rgb = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\2trid.png"
    path_to_depth = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\2depthmap.png"

    try:
        rgb_image_bgr = cv2.imread(path_to_rgb, cv2.IMREAD_COLOR)
        # Load depth image as is. Common formats are 16-bit single-channel grayscale.
        depth_image_raw = cv2.imread(path_to_depth, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error loading images: {e}")
        exit()

    if rgb_image_bgr is None:
        print(f"Failed to load RGB image from: {path_to_rgb}. Check path and file.")
        exit()
    if depth_image_raw is None:
        print(f"Failed to load depth image from: {path_to_depth}. Check path and file.")
        exit()
  # --- New, more robust depth image shape handling ---
    processed_depth_image = None


    if len(depth_image_raw.shape) == 2:
        # This is already a 2D grayscale depth image, perfect.
        print(f"Depth image is 2D. Shape: {depth_image_raw.shape}")
        processed_depth_image = depth_image_raw
    elif len(depth_image_raw.shape) == 3:
        num_channels = depth_image_raw.shape[2]
        print(f"DEBUG: Depth image has 3 dimensions. num_channels = {num_channels}") # Another debug print

        if num_channels == 1: 
            print(f"Depth image has 3 dimensions but only 1 channel. Taking the single channel. Shape: {depth_image_raw.shape}")
            processed_depth_image = depth_image_raw[:, :, 0]
        elif num_channels == 3:
            # Check if it's a grayscale image saved as 3 channels (R=G=B)
            if np.array_equal(depth_image_raw[:,:,0], depth_image_raw[:,:,1]) and \
               np.array_equal(depth_image_raw[:,:,0], depth_image_raw[:,:,2]):
                print("Depth image loaded with 3 identical channels. Taking the first channel as depth.")
                processed_depth_image = depth_image_raw[:,:,0] # Take the first channel
            else:
                # This is the error you are seeing.
                print(f"Error: Depth image loaded as a 3-channel color image (channels are different) with shape: {depth_image_raw.shape}. Expected a single channel depth map or 3 identical channels.")
                exit()
        elif num_channels == 4: # This block should handle your case
            print(f"DEBUG: Entering num_channels == 4 block.")
            print(f"Warning: Depth image loaded with 4 channels (shape: {depth_image_raw.shape}).")
            print("         Assuming depth information is in the FIRST channel (index 0).")
            print("         Please verify this assumption for your specific depth image format.")
            processed_depth_image = depth_image_raw[:,:,0] # Taking the first channel as depth
        else: # This handles cases like 2 or 5+ channels for a 3-dimensionally shaped array
            print(f"Error: Depth image (3D shape) loaded with an unexpected number of channels: {num_channels} in shape: {depth_image_raw.shape}.")
            exit()
    else: # This handles cases where shape is not 2D or 3D (e.g. 1D, 4D+)
        print(f"Error: Depth image has an unexpected number of dimensions: {len(depth_image_raw.shape)}. Shape: {depth_image_raw.shape}.")
        exit()

    # At this point, processed_depth_image should be a 2D array
    if processed_depth_image is None or len(processed_depth_image.shape) != 2:
        print(f"Error: Depth image processing failed. Resulting depth map is not 2D. Shape: {processed_depth_image.shape if processed_depth_image is not None else 'None'}")
        exit()
    
    depth_image_final_for_scaling = processed_depth_image
    # --- End of new depth image shape handling ---

    # Now use 'depth_image_final_for_scaling' for the rest of the processing
    image_height, image_width = depth_image_final_for_scaling.shape[:2]

    # ... (rest of your code for defining intrinsics, scaling depth, calling convert_rgbd_to_pointcloud) ...
    # Make sure to use 'depth_image_final_for_scaling' when you prepare it for the convert_rgbd_to_pointcloud function:

    # Example:
    # if depth_values_are_in_millimeters:
    #     depth_image_scaled_to_meters = depth_image_final_for_scaling.astype(np.float32) / 1000.0
    # else:
    #     depth_image_scaled_to_meters = depth_image_final_for_scaling.astype(np.float32)
    
    # pcd = convert_rgbd_to_pointcloud(
    #     rgb_image_bgr, # Your loaded BGR image
    #     depth_image_scaled_to_meters, # The correctly scaled 2D depth map
    #     camera_intrinsics,
    #     extrinsic_matrix
    # )