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
import zivid


class Zivid:
    def __init__(self, width = 1224, height = 1024):
        # --- Store the core camera properties ---
        self.width = width
        self.height = height
        # --- You can initialize other things here ---
        self.camera_handle = None # Placeholder for the actual Zivid camera connection
        self.is_connected = False

        print(f"ZividCamera instance created for a {width}x{height} camera.")
    def subsampledSettingsForCamera(camera: zivid.Camera) -> zivid.Settings:
        settings_subsampled = zivid.Settings(
        acquisitions=[zivid.Settings.Acquisition()],
        color=zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()]),
        )
        model = camera.info.model
        if (
        model is zivid.CameraInfo.Model.zividTwo
        or model is zivid.CameraInfo.Model.zividTwoL100
        or model is zivid.CameraInfo.Model.zivid2PlusM130
        or model is zivid.CameraInfo.Model.zivid2PlusM60
        or model is zivid.CameraInfo.Model.zivid2PlusL110
        ):
            settings_subsampled.sampling.pixel = zivid.Settings.Sampling.Pixel.blueSubsample2x2
            settings_subsampled.color.sampling.pixel = zivid.Settings2D.Sampling.Pixel.blueSubsample2x2
        elif (
        model is zivid.CameraInfo.Model.zivid2PlusMR130
        or model is zivid.CameraInfo.Model.zivid2PlusMR60
        or model is zivid.CameraInfo.Model.zivid2PlusLR110
        ):
            settings_subsampled.sampling.pixel = zivid.Settings.Sampling.Pixel.by2x2
            settings_subsampled.color.sampling.pixel = zivid.Settings2D.Sampling.Pixel.by2x2
        else:
            raise ValueError(f"Unhandled enum value {model}")

        return settings_subsampled        
    def connect(self):
        app = zivid.Application()
        print("Connecting to camera")
        camera = app.connect_camera()

        print("Getting camera intrinsics")
        intrinsics = zivid.experimental.calibration.intrinsics(camera)
        print(intrinsics)

        output_file = "Intrinsics.yml"
        print(f"Saving camera intrinsics to file: {output_file}")
        intrinsics.save(output_file)