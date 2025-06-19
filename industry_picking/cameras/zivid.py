import zivid
import numpy as np
import cv2
import zivid.experimental
import zivid.experimental.calibration

def connect_to_zivid_camera():
    """
    Connects to the first available Zivid camera.

    Returns:
        zivid.Camera: The connected camera object.
    """
    try:
        app = zivid.Application()
        print("Connecting to camera...")
        camera = app.connect_camera()
        print(f"Connected to camera: {camera.info.model_name}")
        return camera
    except RuntimeError as e:
        print(f"Error connecting to camera: {e}")
        return None


def get_camera_intrinsics_from_capture(camera: zivid.Camera):
    """
    Captures a frame and retrieves the camera intrinsics from that frame.

    Args:
        camera: An active Zivid camera object.
    """
    if not camera:
        print("No camera connected.")
        return

    print("\n--- Capturing a frame to get intrinsics ---")
    
    # Define capture settings. You can adjust these as needed.
    # For a Zivid 2+ M60, you might want to specify an aperture.
    # If no f-number is specified, the SDK will choose one.
    settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])

    # Capture a frame
    with camera.capture(settings) as frame:
        print("Frame captured. Retrieving intrinsics...")
        
        # Intrinsics are an attribute of the captured frame
        intrinsics = zivid.experimental.calibration.estimate_intrinsics(frame)
        #intrinsics = zivid.experimental.calibration.estimate_intrinsics(frame)

        print(intrinsics)

        # The intrinsics object contains the camera matrix and distortion coefficients
        camera_matrix = intrinsics.camera_matrix
        distortion_coeffs = intrinsics.distortion

        print("\nCamera Matrix (OpenCV format):")
        print(camera_matrix)
        print("\nDistortion Coefficients (OpenCV format):")
        print(distortion_coeffs)