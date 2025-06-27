import numpy as np
import time
import cv2 # For loading images (pip install opencv-python)
import pyrealsense2 as rs
import cv2.aruco as aruco
import zivid 
import industry_picking.utils.helper_functions as help


class Camera:
    """A base class for different camera types."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        print(f"Camera instance created with a resolution of {width}x{height}.")

    def _rtvec_to_matrix(self, rvec, tvec):
        """Converts a rotation vector and a translation vector to a 4x4 transformation matrix."""
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        transformation_matrix = np.eye(4)
        transformation_matrix[0:3, 0:3] = rotation_matrix
        transformation_matrix[0:3, 3] = tvec.flatten()
        return transformation_matrix

    def capturePose(self, image, CHARUCO_SQUARES_X=6, CHARUCO_SQUARES_Y=7, CHARUCO_SQUARE_LENGTH_M=0.0263, CHARUCO_MARKER_LENGTH_M=0.0177, charuco_dict_name="DICT_6X6_250"):
        """Estimates the pose of a ChArUco board from an image."""
        try:
            charuco_dict = getattr(aruco, charuco_dict_name)
        except AttributeError:
            print(f"Error: The ArUco dictionary '{charuco_dict_name}' is not valid.")
            return None

        charuco_dictionary = aruco.getPredefinedDictionary(charuco_dict)
        charuco_board = aruco.CharucoBoard(
            (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
            CHARUCO_SQUARE_LENGTH_M,
            CHARUCO_MARKER_LENGTH_M,
            charuco_dictionary
        )
        aruco_params = aruco.DetectorParameters()

        camera_matrix, dist_coeffs = self.getIntrinsics()
        if camera_matrix is None:
            print("Error: Could not get camera intrinsics.")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, charuco_dictionary, parameters=aruco_params)

        if ids is not None and len(ids) > 0:
            retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, charuco_board
            )
            if retval and charuco_corners is not None and len(charuco_corners) > 3:
                success, rvec, tvec = aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs, None, None
                )
                if success:
                    T_target_in_cam = self._rtvec_to_matrix(rvec, tvec)
                    # Visualization
                    image_with_detections = aruco.drawDetectedCornersCharuco(image.copy(), charuco_corners, charuco_ids)
                    cv2.drawFrameAxes(image_with_detections, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                    cv2.imshow('ChArUco Detection', image_with_detections)
                    cv2.waitKey(500)
                    return T_target_in_cam
                else:
                    print("Warning: Pose estimation for ChArUco board failed.")
            else:
                print("Warning: Not enough ChArUco corners interpolated.")
        else:
            print("Warning: No ArUco markers found.")
            cv2.imshow('Detection Failed', image)
            cv2.waitKey(500)
        return None

class Zivid(Camera):
    """A subclass for Zivid cameras."""
    def __init__(self, width, height):
        super().__init__(width, height)
        self.app = zivid.Application()
        print("Zivid camera instance created.")

    def connect(self):
        try:
            print("Connecting to camera...")
            self.camera = self.app.connect_camera()
            print(f"Connected to camera: {self.camera.info.model_name}")
            return self.camera
        except RuntimeError as e:
            print(f"Error connecting to camera: {e}")
            return None            

    def getIntrinsics(self):
        if not self.camera:
            print("No camera connected.")
            return

        print("\n--- Capturing a frame to get intrinsics ---")
        settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])

        # Capture a frame
        with self.camera.capture(settings) as frame:
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

    def captureImage(self):
        pass

class RealSense(Camera):
    """A subclass for Intel RealSense cameras."""
    def __init__(self, width, height):
        super().__init__(width, height)
        self.pipeline = None
        self.config = None
        print("RealSense instance created.")

    def connect(self):
        print("Initializing Intel RealSense Camera...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)

        try:
            self.profile = self.pipeline.start(self.config)            
            print("RealSense Camera Initialized and pipeline started.")
            time.sleep(2)  # Allow for auto-adjustment
        except RuntimeError as e:
            print(f"Error starting RealSense pipeline: {e}")
            self.pipeline = None

    def getIntrinsics(self):
        if not self.pipeline:
            print("Error: Pipeline not active. Call connect() first.")
            return None
        try:
            profile = self.pipeline.get_active_profile()
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            color_intrinsics = color_profile.get_intrinsics()
            camera_matrix = np.array([
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float32)
            dist_coeffs = np.array(color_intrinsics.coeffs, dtype=np.float32)
            return camera_matrix, dist_coeffs
        except Exception as e:
            print(f"An error occurred while getting intrinsics: {e}")
            return None, None

    def captureImage(self):
        if not self.pipeline:
            print("Error: Pipeline not active. Call connect() first.")
            return None
        
        frames = self.pipeline.wait_for_frames()
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 0.2 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Streaming loop
        try:
            while True:
                # Get frameset of color and depth
                frames = self.pipeline.wait_for_frames()
                # frames.get_depth_frame() is a 640x360 depth image

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Remove background - Set pixels further than clipping_distance to grey
                grey_color = 153
                depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                images = np.hstack((bg_removed, depth_colormap))
                

                cv2.waitKey(1)
                return color_image, depth_image

        finally:
            self.pipeline.stop()
