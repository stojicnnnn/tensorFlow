import numpy as np
import time
import cv2 # For loading images (pip install opencv-python)
import pyrealsense2 as rs
import cv2.aruco as aruco



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



        print("Zivid camera instance created.")

    def connect(self):




        print("Connecting to Zivid camera...")
        pass

    def getIntrinsics(self):
        pass




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
        try:
            self.pipeline.start(self.config)
            print("RealSense Camera Initialized and pipeline started.")
            time.sleep(2)  # Allow for auto-adjustment
        except RuntimeError as e:
            print(f"Error starting RealSense pipeline: {e}")
            self.pipeline = None

    def getIntrinsics(self):
        if not self.pipeline:
            print("Error: Pipeline not active. Call connect() first.")
            return None, None
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
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("Warning: Could not get color frame.")
                return None
            image = np.asanyarray(color_frame.get_data())
            print("Captured RealSense camera image.")
            return image
        except Exception as e:
            print(f"Failed to capture image: {e}")
            return None

    def __del__(self):
        if self.pipeline:
            self.pipeline.stop()
            print("RealSense pipeline stopped.")