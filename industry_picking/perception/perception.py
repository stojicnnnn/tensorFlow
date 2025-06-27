import numpy as np
import cv2
from industry_picking.cameras.camera import RealSense

depth_image = cv2.imread(r'C:\Users\Nikola\OneDrive\Desktop\zividSlike\data\depth_image_16bit.png', cv2.IMREAD_UNCHANGED)

if depth_image is not None:
    print(f"--- Statistical Summary of Depth Image ---")
    print(f"Data Type (dtype): {depth_image.dtype}")
    print(f"Shape (height, width): {depth_image.shape}")

        # Max value - useful for seeing the furthest point
    max_value = np.max(depth_image)
    print(f"Maximum depth value: {max_value}")

    # Min value of non-zero pixels - useful for seeing the closest point
    # We ignore zeros because they represent invalid pixels with no depth data.
    min_value_nonzero = np.min(depth_image[np.nonzero(depth_image)])
    print(f"Minimum non-zero depth value: {min_value_nonzero}")

    # Mean value - gives a sense of the average distance of objects in the scene
    mean_value = np.mean(depth_image[np.nonzero(depth_image)])
    print(f"Mean non-zero depth value: {mean_value:.2f}")

# cam = RealSense(width=640,height=480)
# cam.connect()
# example_camera_K, _= cam.getIntrinsics()
# img , depth = cam.captureImage()


