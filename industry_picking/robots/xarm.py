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

class Xarm:
    def __init__(self,ip):
        self.ROBOT_IP = ip
        self.arm = None
    def connect(self):
        # # --- Initialize Robot Arm ---
        print("Initializing UFactory xArm...")
        print(f"with the ip {self.ROBOT_ip}")
        self.arm = XArmAPI(self.ROBOT_IP)
        self.arm.connect()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0) # Position control mode
        self.arm.set_state(state=0) # Ready state
        print("xArm Initialized.")
    def move(self,pose: np.ndarray):
        x = pose[0, 3] * 1000
        y = pose[1, 3] * 1000
        z = pose[2, 3] * 1000
        roll, pitch, yaw = euler.mat2euler(pose[:3, :3])
        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        error = self.arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=80,
            wait=True,
        )
        time.sleep(1)
    def getPose(self):
        ok, pose = self.arm.get_position()
        if ok != 0:
            return None

        translation = np.array(pose[:3]) / 1000
        eulers = np.array(pose[3:]) * math.pi / 180
        rotation = euler.euler2mat(
            eulers[0], eulers[1], eulers[2], 'sxyz')
        pose = affines.compose(translation, rotation, np.ones(3))
        return pose
    def create_pose_matrix(self,x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg):
        # Convert the Euler angles from degrees to radians
        roll_rad = np.deg2rad(roll_deg)
        pitch_rad = np.deg2rad(pitch_deg)
        yaw_rad = np.deg2rad(yaw_deg)

        # Create the 3x3 rotation matrix from the Euler angles
        # 'sxyz' is a common convention for roll, pitch, yaw
        rotation_matrix = euler2mat(roll_rad, pitch_rad, yaw_rad, axes='sxyz')

        # ---- 2. Handle the Position ----
        # Create the position vector and convert it from millimeters to meters
        position_vector_meters = np.array([x_mm, y_mm, z_mm]) / 1000.0

        # ---- 3. Assemble the final 4x4 Pose Matrix ----
        # Start with a 4x4 identity matrix (a clean slate)
        pose_matrix = np.identity(4)

        # Place the 3x3 rotation matrix in the top-left corner
        pose_matrix[:3, :3] = rotation_matrix

        # Place the converted position vector in the last column
        pose_matrix[:3, 3] = position_vector_meters

        return pose_matrix 
    def record_manual_poses(self, num_poses: int = 10):
        """
        Interactively records a specified number of robot poses.
        
        The user manually moves the robot to a desired position and presses Enter
        to record the pose. This repeats for the specified number of iterations.
        
        Args:
            robot_ip (str): The IP address of the xArm.
            num_poses (int): The total number of poses to record.

        Returns:
            list: A list of 4x4 numpy arrays, where each array is a recorded pose.
                Returns an empty list if the connection fails or is interrupted.
        """
        
        # List to store the recorded 4x4 pose matrices
        recorded_poses = []
        
        # Initialize and connect to the robot arm
        arm = XArmAPI(self.ROBOT_IP)
        arm.connect()
        if not arm.connected:
            print(f"Error: Failed to connect to xArm at IP: {self.ROBOT_IP}")
            return recorded_poses

        try:
            # Enable manual mode on the robot so it can be moved by hand.
            # This will require you to press the "Freedrive" button on the arm.
            print("\nEnabling manual mode on the robot arm.")
            print("You will need to press and hold the 'Freedrive' button on the arm to move it.")
            arm.set_mode(2)  # Mode 2 is manual movement mode
            arm.set_state(0)
            time.sleep(1)

            print(f"\nReady to record {num_poses} poses.")
            print("="*40)

            # Loop for the specified number of iterations
            for i in range(num_poses):
                # Wait for the user to press a key
                input(f"--> Pose {i+1}/{num_poses}: Move the robot to the desired position, then press Enter to record...")
                
                # Read the robot's current position
                error_code, current_pose_list = arm.get_position(is_radian=False)
                
                if error_code != 0:
                    print(f"Error reading robot position (error code: {error_code}). Skipping this pose.")
                    continue

                # Extract XYZ and RPY values
                # The SDK returns position in millimeters, so we convert to meters
                x_meters = current_pose_list[0] 
                y_meters = current_pose_list[1] 
                z_meters = current_pose_list[2] 
                roll_degrees = current_pose_list[3]
                pitch_degrees = current_pose_list[4]
                yaw_degrees = current_pose_list[5]

                print(f"    ...Recording pose: X={x_meters:.4f}m, Y={y_meters:.4f}m, Z={z_meters:.4f}m, R={roll_degrees:.1f}°, P={pitch_degrees:.1f}°, Y={yaw_degrees:.1f}°")

                # Convert the pose to a 4x4 matrix
                pose_matrix = self.create_pose_matrix(
                    x_mm=x_meters,
                    y_mm=y_meters,
                    z_mm=z_meters,
                    roll_deg=roll_degrees,
                    pitch_deg=pitch_degrees,
                    yaw_deg=yaw_degrees
                )

                if pose_matrix is not None:
                    # Store the matrix in our list
                    recorded_poses.append(pose_matrix)
                    print(f"    ...Pose {i+1} successfully recorded.")
                else:
                    print(f"    ...Failed to create matrix for pose {i+1}. Skipping.")

        except Exception as e:
            print(f"\nAn error occurred during the process: {e}")
        finally:
            # Always make sure to disconnect from the robot
            print("\nDisconnecting from the robot arm.")
            arm.set_mode(0) # Set back to position control mode
            arm.set_state(0)
            arm.disconnect()
            
        return recorded_poses