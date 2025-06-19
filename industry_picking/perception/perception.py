import cv2
import numpy as np
from ultralytics import YOLO


# --- Configuration ---
# You can change the model here. Common choices for segmentation:
# 'yolov8n-seg.pt' (fastest, lowest accuracy)
# 'yolov8s-seg.pt'
# 'yolov8m-seg.pt'
# 'yolov8l-seg.pt'
# 'yolov8x-seg.pt' (slowest, highest accuracy)
MODEL_PATH = 'yolov8n-seg.pt'

# The path to the image you want to process.
# Replace this with your image path or a webcam index (e.g., 0).
IMAGE_PATH = r"C:\Users\Nikola\OneDrive\Desktop\zividSlike\sampleObj\1color.png"
# Example of a real image URL you could use for testing:
# IMAGE_PATH = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"


# --- Main Application Logic ---

def run_segmentation():
    """
    Loads a YOLO segmentation model and processes an image to detect objects
    and generate their segmentation masks.
    """
    try:
        # 1. Load the YOLO Model
        # This will download the model if it's not already available.
        print(f"Loading model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")

        # 2. Load the Image
        # Using a placeholder if the path is the default one
        if IMAGE_PATH.startswith("https://placehold.co"):
             print(f"Loading placeholder image from: {IMAGE_PATH}")
             # In a real scenario, you'd handle file not found errors
             # For this example, we create a dummy image if the placeholder is used
             # to ensure the script can run.
             img_resp = cv2.imdecode(np.frombuffer(cv2.imencode('.png', np.zeros((720, 1280, 3), dtype=np.uint8))[1], np.uint8), cv2.IMREAD_COLOR)
             cv2.putText(img_resp, "Your Image Here", (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        # Handle local file paths
        elif not (IMAGE_PATH.startswith("http://") or IMAGE_PATH.startswith("https://")):
            print(f"Loading image from local path: {IMAGE_PATH}")
            img_resp = cv2.imread(IMAGE_PATH)
            if img_resp is None:
                raise FileNotFoundError(f"Image not found at path: {IMAGE_PATH}")
        # Handle URLs
        else:
             # In a real application, you would download the image
             print(f"Note: Using online images is best handled with libraries like requests and Pillow.")
             print(f"For this script, please download the image and provide a local path.")
             # As a fallback for this script to run, create a placeholder
             img_resp = cv2.imdecode(np.frombuffer(cv2.imencode('.png', np.zeros((720, 1280, 3), dtype=np.uint8))[1], np.uint8), cv2.IMREAD_COLOR)


        # Create a copy for drawing so the original is preserved
        original_image = img_resp
        
        # Create a black color image to draw the colored masks on
        colored_masks_image = np.zeros(original_image.shape, dtype=np.uint8)


        # 3. Perform Inference
        print("Performing inference...")
        # The model returns a list of results objects.
        results = model(original_image)
        print("Inference complete.")

        # 4. Process the Results
        # Check if any masks were detected
        if results[0].masks is not None:
            # The number of detected objects with masks
            num_masks = len(results[0].masks)
            print(f"Found {num_masks} masks.")

            # Get the masks tensor
            # Shape: (num_masks, height, width) at inference size
            masks_data = results[0].masks.data.cpu().numpy()

            # Generate a list of random colors, one for each detected object
            np.random.seed(42) # for consistent colors
            colors = np.random.randint(0, 256, size=(num_masks, 3), dtype=np.uint8)

            # Loop through each detected object
            for i in range(num_masks):
                # Get the binary mask for the current object at inference size
                mask_raw = masks_data[i]

                # Resize the mask to the original image's dimensions
                target_height, target_width = original_image.shape[:2]
                mask_i = cv2.resize(mask_raw, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                
                # Get the color for the current instance
                color = colors[i]
                color_tuple = (int(color[0]), int(color[1]), int(color[2]))
                
                # Set the pixels of the colored mask image to the instance color
                # We use a threshold of 0.5 because the resized mask has float values.
                colored_masks_image[mask_i > 0.5] = color_tuple


            # 5. Display the result
            print("Displaying results. Press any key to exit.")
            cv2.imshow("Original Image", original_image)
            cv2.imshow("Colored Masks", colored_masks_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Optionally, save the result
            save_path = "segmentation_mask_result.png"
            cv2.imwrite(save_path, colored_masks_image)
            print(f"Result saved to {save_path}")

        else:
            print("No masks were detected in the image.")
            cv2.imshow("YOLOv8 Segmentation", original_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # To run this script, you need to install the required libraries:
    # pip install ultralytics opencv-python numpy
    run_segmentation()
