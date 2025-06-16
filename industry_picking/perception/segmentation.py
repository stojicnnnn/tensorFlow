import numpy as np
import cv2 # For loading images (pip install opencv-python)
import os # For os.path.exists if you use it, though not in current snippet
import requests
from typing import List, Tuple, Optional # For type hinting
import glob

def getSegmentationMasksSAM(
    rgb_image_path: str,
    sam_server_url: Optional[str],
    sam_query: str,
    masks_input_dir: Optional[str],
    input_rgb_shape: Tuple[int, int]
) -> List[np.ndarray]:
    """
    Gets segmentation masks either from an online SAM server or a local directory.

    Tries the SAM server first. If it fails (due to network error, server down, etc.)
    or if no URL is provided, it falls back to loading masks from the local directory.

    Args:
        rgb_image_path: Path to the original RGB image (for the online server).
        sam_server_url: URL of the SAM server. Can be None to force local mode.
        sam_query: The text prompt for SAM.
        masks_input_dir: Path to the directory containing pre-saved mask images.
        input_rgb_shape: The (height, width) of the original RGB image.

    Returns:
        A list of boolean numpy arrays, where each array is a segmentation mask.
        Returns an empty list if both methods fail.
    """
    masks = []
    H_orig, W_orig = input_rgb_shape

    # --- Mode 1: Try Online SAM Server ---
    if sam_server_url:
        print(f"Attempting to get masks from online SAM server: {sam_server_url}")
        try:
            input_rgb_for_sam = cv2.imread(rgb_image_path)
            if input_rgb_for_sam is None:
                raise ValueError(f"Could not read RGB image at {rgb_image_path}")

            response = requests.post(
                sam_server_url,
                files={"image": cv2.imencode(".jpg", input_rgb_for_sam)[1].tobytes()},
                data={"query": sam_query},
                timeout=15 # A shorter timeout to fail faster if the server is offline
            )
            response.raise_for_status() # Check for HTTP errors like 404 or 500

            decoded_stack = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
            if decoded_stack is None:
                raise ValueError("Could not decode image received from SAM server.")

            # Process the stacked image returned by the server
            num_segments = decoded_stack.shape[0] // H_orig
            for i in range(1, num_segments):
                start_row = i * H_orig
                end_row = (i + 1) * H_orig
                instance_img = decoded_stack[start_row:end_row, :, :]
                # Convert to boolean mask
                binary_mask = np.any(instance_img > 10, axis=2) if len(instance_img.shape) == 3 else instance_img > 10
                masks.append(binary_mask)

            print(f"Successfully retrieved {len(masks)} masks from the online server.")
            return masks

        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"\nOnline SAM server failed: {e}")
            print("--- Falling back to local mask directory. ---\n")
            # If it fails, the function will proceed to the offline mode below

    # --- Mode 2: Fallback to Local Directory ---
    print(f"Loading masks from local directory: {masks_input_dir}")
    if not masks_input_dir or not os.path.isdir(masks_input_dir):
        print(f"Error: Local mask directory not found or not specified: {masks_input_dir}")
        return []

    # Find all png/jpg/jpeg files, sort them to ensure consistent order
    mask_files = sorted(glob.glob(os.path.join(masks_input_dir, "*.png")))
    mask_files.extend(sorted(glob.glob(os.path.join(masks_input_dir, "*.jpg"))))
    mask_files.extend(sorted(glob.glob(os.path.join(masks_input_dir, "*.jpeg"))))

    if not mask_files:
        print(f"Warning: No mask files found in {masks_input_dir}")
        return []

    for mask_file in mask_files:
        mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask_image is not None:
            # Convert to boolean mask (True for any pixel value > 10)
            binary_mask = mask_image > 10
            masks.append(binary_mask)

    print(f"Successfully loaded {len(masks)} masks from the local directory.")
    return masks
def getSegmentationMasksYOLO():
    pass

