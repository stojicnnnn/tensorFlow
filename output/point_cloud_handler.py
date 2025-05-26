#importing all the needed libraries for zivid camera, point cloud processing and general python 
import zivid
import numpy as np
import open3d as o3d
import os
from datetime import datetime

#function for capturing point clouds from zivid camera
def capture_point_cloud_from_zivid(
    settings_file_path=None, output_zdf_path=None, output_ply_path=None
):
    
    #Arguments:
    #    settings_file_path (str, optional): Path to a .yml camera settings file.  If None, default or last-used settings might be applied.
    #  output_zdf_path (str, optional): If provided, saves the raw Zivid frame to this .zdf file path.
    #  output_ply_path (str, optional): If provided, saves the point cloud to this .ply file path.

    #Returns:
    #    open3d.geometry.PointCloud: The captured point cloud as an Open3D object.Returns None if capture fails.
    
    app = None #default
    try: 
        print("Init Zivid app")
        app = zivid.Application() #Creates a instance of zivid app,needed for everything related to zivid SDK
        #cleaned at the end of the function

        print("Connecting to camera")
        camera = app.connect_camera() #Attempting to connect to camera

        if not camera: #Maybe not needed since if camera is not found error will occur and trigger exception block
            print("Error: Could not connect to camera.")
            return None

        print(f"Connected to camera: {camera.info.serial_number}")  

        # Acquisition settings
        if settings_file_path:
            if os.path.exists(settings_file_path):
                print(f"Loading settings from: {settings_file_path}")
                camera_settings = zivid.Settings.load(settings_file_path)
                #Checks if we provided path to .yml file
                #.yml file usually contains camera settings from Zivid studio
                
            else:
                print(f"Warning: Settings file not found at {settings_file_path}. Using default/current settings.")
                # Create a default settings object if needed for a single capture
                camera_settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])
        else:
            print("No settings file provided. Using default/current camera settings.")
            # Create a default settings object for a single capture
            # To do: make a better default settings !!!                                                                                              TO DO
            camera_settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])
            # Example: Adjusting exposure time for the first acquisition
            # if camera_settings.acquisitions:
            #     camera_settings.acquisitions[0].exposure_time = datetime.timedelta(microseconds=20000)


        print("Capturing frame...")
        with camera.capture(camera_settings) as frame: #tells the camera to capture an image, using camera_settings we set up earlier
            # "as frame" stores the captured data in the frame variable
            print("Frame captured.")

            if output_zdf_path:
                print(f"Saving frame to Zivid Data File (ZDF): {output_zdf_path}")
                frame.save(output_zdf_path)
                print(f"Frame saved to {output_zdf_path}")
                # saves full raw capture including point cloud,snr,color,depth

            print("Extracting point cloud data (XYZ)...")
            point_cloud_xyz = frame.point_cloud().copy_data("xyz") # from the object frame which holds multiple formats, gets 3d coords,xyz
            #stores it into numpy array point_cloud_xyz
            #organized point cloud, TO DO: explore more about different point cloud formats                                                         TO DO
            point_cloud_rgba = frame.point_cloud().copy_data("rgba") # Get color data   

            # Zivid point clouds are organized (height x width x 3/4).
            # For Open3D, we often need a flat list of points (N x 3), where N is total number of points.
            points = point_cloud_xyz.reshape(-1, 3)
            # 'reshape(-1, 3)' changes the shape of the 'point_cloud_xyz' array.
            # The '-1' tells NumPy to automatically calculate the number of rows
            # needed to make it have 3 columns (for X, Y, Z).
            # So, if you had 100x100 points, this becomes a 10000x3 array.
            
            # Extract colors (RGB) 0-255 in Zivid and normalize to [0,1] for Open3D
            # RGBA is (height x width x 4)
            colors_rgba = point_cloud_rgba.reshape(-1, 4)
            colors_rgb = colors_rgba[:, :3] / 255.0  # Take RGB, normalize

            valid_indices = ~np.isnan(points).any(axis=1)
            points_valid = points[valid_indices]
            colors_valid = colors_rgb[valid_indices]

            # 'np.isnan(points)' creates a boolean array: True where a coordinate is NaN, False otherwise.
            # '.any(axis=1)' checks across each row (axis=1, i.e., for each point's X,Y,Z). 
            #  If ANY coordinate in a point is NaN, it returns True for that point.
            # '~' is the bitwise NOT operator, used here to invert the boolean values.
            # So, 'valid_indices' is True for points where ALL coordinates are valid numbers, and False otherwise.

            if points_valid.shape[0] == 0:
                print("Warning: No valid points found in the point cloud.")
                return None

            print(f"Number of valid points: {points_valid.shape[0]}")

            # Create Open3D point cloud object
            o3d_point_cloud = o3d.geometry.PointCloud() # Creates an empty open3d pointcloud object
            o3d_point_cloud.points = o3d.utility.Vector3dVector(points_valid) 
            # Converts NumPy array of valid point into the specific C++ vector format Open3D uses internally for points.
            o3d_point_cloud.colors = o3d.utility.Vector3dVector(colors_valid)
            # Same for color data

            if output_ply_path:
                print(f"Saving point cloud to PLY: {output_ply_path}")
                o3d.io.write_point_cloud(output_ply_path, o3d_point_cloud)
                print(f"Point cloud saved to {output_ply_path}")
                # This saves the 'o3d_point_cloud' object to a .ply file.
                # PLY is a common, simple format for 3D data, widely supported.

            return o3d_point_cloud

    except RuntimeError as e:
        # If a 'RuntimeError' occurs in the 'try' block (e.g., camera not found, capture fails),
        # this 'except' block will catch it. 'e' will hold the error object.
        print(f"Zivid Runtime Error: {e}")
        return None # Return None indicating failure.
    except Exception as e:
        # This is a more general error handler. 'Exception' catches almost any type of error
        # that wasn't caught by more specific 'except' blocks above it.
        print(f"An unexpected error occurred: {e}")
        return None # Return None indicating failure.
    finally:
        # The 'finally' block contains code that will *always* run,
        # whether the 'try' block succeeded, or an error occurred and was caught by 'except'.
        # It's often used for cleanup actions.
        # In this script, the Zivid 'app' object is created locally within the function.
        # When the function exits, 'app' will be automatically cleaned up by Python's garbage collector.
        # If 'app' were a more persistent object (e.g., part of a class), you might do explicit
        # disconnection or cleanup here. For now, this print statement just signals completion.
        print("Camera capture function finished.")


def load_point_cloud_from_file(file_path):
   
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    try:
        print(f"Loading point cloud from: {file_path}")
        o3d_point_cloud = o3d.io.read_point_cloud(file_path)
        if not o3d_point_cloud.has_points():
            print(f"Warning: No points found in the loaded file: {file_path}")
            return None # Or return empty point cloud based on desired behavior
        print(f"Point cloud loaded successfully. Number of points: {len(o3d_point_cloud.points)}")
        return o3d_point_cloud
    except Exception as e:
        print(f"Error loading point cloud from file: {e}")
        return None

if __name__ == "__main__": 


    # --- Example Usage ---

    # --- Option 1: Capture from Zivid Camera ---
    # Note: This requires a Zivid camera to be connected and SDK properly installed.
    
    # Define a path for camera settings (optional, create a .yml file with Zivid Studio)
    # example_settings_file = "path/to/your/camera_settings.yml" 
    example_settings_file = None # Set to None to use default/current settings

    # Timestamp to avoid overwriting files during tests.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zdf_output = f"zivid_capture_{timestamp}.zdf"
    ply_output_live = f"zivid_capture_{timestamp}.ply"

    print("\n--- Attempting to capture from Zivid camera ---")
    # live_pcd = capture_point_cloud_from_zivid(
    #     settings_file_path=example_settings_file,
    #     output_zdf_path=zdf_output,
    #     output_ply_path=ply_output_live
    # )

    # if live_pcd:
    #     print("Successfully captured point cloud from Zivid camera.")
    #     print(f"Number of points in live_pcd: {len(live_pcd.points)}")
    #     # You can now visualize or process live_pcd
    #     # o3d.visualization.draw_geometries([live_pcd])
    # else:
    #     print("Failed to capture point cloud from Zivid camera.")
    print("Camera capture example commented out. Uncomment to run if a camera is connected.")

    # --- Option 2: Load from a PLY file ---
    # First, ensure you have a .ply file. If you ran the capture above,
    # 'ply_output_live' would be a candidate.
    # For testing, let's assume you have a file named "test_cloud.ply"
    # You might need to create a dummy PLY file or use one from a previous capture.

    # Create a dummy PLY file for testing if you don't have one
    # This is just for the example to run without a camera.
    # In a real scenario, you'd use a PLY file from a Zivid capture or other source.
    dummy_ply_file_path = "dummy_test_cloud.ply"
    if not os.path.exists(dummy_ply_file_path):
        print(f"\nCreating a dummy PLY file for testing: {dummy_ply_file_path}")
        # Create a simple cube point cloud
        dummy_points = np.array([
            [0,0,0], [1,0,0], [0,1,0], [1,1,0],
            [0,0,1], [1,0,1], [0,1,1], [1,1,1]
        ])
        dummy_colors = np.array([ # RGB, normalized
            [1,0,0], [0,1,0], [0,0,1], [1,1,0],
            [1,0,1], [0,1,1], [0.5,0.5,0.5], [0,0,0]
        ])
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(dummy_points)
        temp_pcd.colors = o3d.utility.Vector3dVector(dummy_colors)
        o3d.io.write_point_cloud(dummy_ply_file_path, temp_pcd)
        print(f"Dummy PLY file created: {dummy_ply_file_path}")


    print("\n--- Attempting to load point cloud from file ---")
    # Replace 'dummy_test_cloud.ply' with the path to your actual .ply file
    # For example, if you successfully ran the capture part:
    # loaded_pcd = load_point_cloud_from_file(ply_output_live) 
    
    loaded_pcd = load_point_cloud_from_file(dummy_ply_file_path)

   # ... (loading part remains the same up to the 'if loaded_pcd:' check) ...
    if loaded_pcd and loaded_pcd.has_points(): # Ensure pcd is not None and has points
        print("Successfully loaded point cloud from file.")
        print(f"Number of points in loaded_pcd: {len(loaded_pcd.points)}")
        
        print("Visualizing loaded point cloud using o3d.visualization.Visualizer...")
        vis = o3d.visualization.Visualizer()
        try:
            vis.create_window(window_name="Loaded Point Cloud - Visualizer", width=800, height=600)
            
            # Add the geometry to the visualizer
            vis.add_geometry(loaded_pcd)
            
            # Get rendering options
            render_option = vis.get_render_option()
            
            # Increase point size (default is often 1.0 which can be tiny)
            render_option.point_size = 25.0 
            
            # Optionally set a background color (e.g., light grey)
            # Default is a shade of grey, but being explicit can help.
            render_option.background_color = np.asarray([0.8, 0.8, 0.8]) 

            # Attempt to reset the view to frame the geometry automatically
            # This is usually quite effective.
            vis.reset_view_point(True)

            # --- For more manual control if reset_view_point isn't sufficient (optional) ---
            # view_control = vis.get_view_control()
            # print("Setting manual view control parameters (example values)...")
            # For the dummy cube (0,0,0) to (1,1,1):
            # view_control.set_front([-0.5, -0.5, -1])  # Direction camera is pointing
            # view_control.set_lookat([0.5, 0.5, 0.5]) # Point camera is looking at (center of cube)
            # view_control.set_up([0, 1, 0])          # Orientation of 'up'
            # view_control.set_zoom(0.8)              # Zoom level

            print("Starting visualizer. Close the window to continue script.")
            vis.run()
            
        except Exception as e:
            print(f"Error during visualization: {e}")
        finally:
            if 'vis' in locals() and vis: # Ensure vis was created
                vis.destroy_window()
            
    elif loaded_pcd is None: # Explicitly check if loading failed
        print(f"Failed to load point cloud from {dummy_ply_file_path}.")
    else: # Loaded but has no points
        print(f"Point cloud loaded from {dummy_ply_file_path} but contains no points.")