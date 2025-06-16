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
import glob
from xarm.wrapper import XArmAPI
import pyrealsense2 as rs
import math 
from transforms3d import euler
import glob # We'll use this for finding mask files
from transforms3d import affines 
from transforms3d.euler import euler2mat 
import cv2.aruco as aruco



class Camera:
    def __init__(self,width,height):
        self.width = width
        self.height = height
        print(f"RealSense instance created with a resolution of {width}x{height}.")
    def test(self):
        print("Test success!")
    def connect(self):
        print("Initializing Intel RealSense Camera...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width , self.height, rs.format.bgr8, 30)
        pipeline.start(config)
        print("RealSense Camera Initialized.")
        time.sleep(2) # Give camera time to auto-adjust exposure, etc. 
    def _rtvec_to_matrix(self,rvec, tvec):
        """Converts a rotation vector and a translation vector to a 4x4 transformation matrix."""
        rotation_matrix, _ = cv2.Rodrigues(rvec) # _ to ignore the Jakobian that cv2.Rodrigues returnes
        transformation_matrix = np.eye(4) # Square, neutral matrix, no rotation and no translatio
        transformation_matrix[0:3, 0:3] = rotation_matrix
        transformation_matrix[0:3, 3] = tvec.flatten() #flatten makes sure that we get a simple 2d array 1x3 and not 3x1
        return transformation_matrix
    def getIntrinsics(self):
        
        # Returns camera matrix with color intrinsics as well as distrortion coeffs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # 3. Call start() on the pipeline INSTANCE, passing the config INSTANCE
        profile = pipeline.start(config)
        try:
            # Get the active stream profiles from the started profile object
            depth_profile = profile.get_stream(rs.stream.depth)
            color_profile = profile.get_stream(rs.stream.color)

            # Downcast the profile to a video_stream_profile and get intrinsics
            depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

            camera_matrix = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.array(color_intrinsics.coeffs, dtype=np.float32)

            return camera_matrix,dist_coeffs

        finally:
            # 4. Call stop() on the pipeline INSTANCE
            pipeline.stop()
            print("\nGot camera intrinsics.")
    def captureImage(self):
        frames = rs.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("Warning: Could not get color frame. Skipping this pose.")
            return
        image = np.asanyarray(color_frame.get_data())
        print("Captured camera image.")
        return image        
    def capturePose(self,image,CHARUCO_SQUARES_X=6,CHARUCO_SQUARES_Y=7,CHARUCO_SQUARE_LENGTH_M=0.0263,CHARUCO_MARKER_LENGTH_M=0.0177,charuco_dict_name="DICT_6X6_250"):
        self.image = image
        
        try:
            charuco_dict=charuco_dict_name
            dictionary_constant = getattr(aruco, charuco_dict)
        except AttributeError:
            print(f"Error: The ArUco dictionary '{charuco_dict}' is not valid.")
        charuco_dictionary = aruco.getPredefinedDictionary(dictionary_constant)

        charuco_board = aruco.CharucoBoard(
            ( CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y ), 
            CHARUCO_SQUARE_LENGTH_M, 
            CHARUCO_MARKER_LENGTH_M, 
            charuco_dictionary
            )
        aruco_params = aruco.DetectorParameters()
        # Lists for storing robot and board poses
        target_poses = []
        # Getting the camera intrinsics
        camera_matrix,dist_coeffs = self.get_cam_intrinsics()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, self.CHARUCO_DICTIONARY, parameters=aruco_params)

        if ids is not None and len(ids) > 0:
            print(f"Found {len(ids)} ArUco markers. Interpolating ChArUco corners...")
                
            # Interpolate to find the precise checkerboard corners from the detected ArUco markers
            retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, charuco_board
            )
            # If we found enough ChArUco corners, estimate the board's pose
            if retval and charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                print(f"Successfully interpolated {len(charuco_corners)} ChArUco corners. Estimating pose...")

                  # Estimate the pose of the ChArUco board.
                  # This function directly gives the pose relative to the camera frame.
                  # It uses the 3D board definition and the found 2D corners.
                success, rvec, tvec = aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs, None, None
                )
                if success:
                    print("ChArUco pose estimated successfully!")     
                    # Convert to 4x4 matrix
                    T_target_in_cam = self.rtvec_to_matrix(rvec, tvec)

                    # Saving target pose
                    target_poses.append(T_target_in_cam)

                    # Draw the detected corners and axes for visualization
                    image_with_detections = aruco.drawDetectedCornersCharuco(image.copy(), charuco_corners, charuco_ids)
                    cv2.drawFrameAxes(image_with_detections, camera_matrix, dist_coeffs, rvec, tvec, 0.1) # Draw a 10cm axis
                    cv2.imshow('ChArUco Detection', image_with_detections)
                    cv2.waitKey(500)
                    print("Successfully stored data pair for this pose.")
                else:
                    print("Warning: Pose estimation for ChArUco board failed.")
            else:
                print("Warning: Not enough ChArUco corners interpolated to estimate pose.")
        else:
            print("Warning: No ArUco markers found in this image. Skipping pose.")
            cv2.imshow('Detection Failed', image)
        return target_poses
        
