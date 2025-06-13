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

class xarm:
    def __init__(self,ip):
        self.ROBOT_IP = ip
        pass
    def _connect_arm(self):
        # # --- Initialize Robot Arm ---
        print("Initializing UFactory xArm...")
        arm = XArmAPI(self.ROBOT_IP)
        arm.connect()
        arm.motion_enable(enable=True)
        arm.set_mode(0) # Position control mode
        arm.set_state(state=0) # Ready state
        print("xArm Initialized.")
    def _move_arm(self,pose: np.ndarray):
        x = pose[0, 3] * 1000
        y = pose[1, 3] * 1000
        z = pose[2, 3] * 1000
        roll, pitch, yaw = euler.mat2euler(pose[:3, :3])
        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        error = self._arm.set_position(
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
    def get_tcp_pose(self):
        ok, pose = self._arm.get_position()
        if ok != 0:
            return None

        translation = np.array(pose[:3]) / 1000
        eulers = np.array(pose[3:]) * math.pi / 180
        rotation = euler.euler2mat(
            eulers[0], eulers[1], eulers[2], 'sxyz')
        pose = affines.compose(translation, rotation, np.ones(3))
        return pose
        