import cv2
import numpy as np
import os

def process_image(image_path):
    # Load the image in grayscale
    x = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded correctly
    if x is None:
        print(f"Error loading image from {image_path}")
        return
    
    # Compute the mean brightness of the image
    mean_brightness = np.mean(x)
    # You can also compute the standard deviation if needed
    # std_deviation = np.std(x)
    
    # Convert image to double precision (float64)
    x = x.astype(np.float64)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    x = clahe.apply(x.astype(np.uint8))
    
    # Adjust the multiplication factor based on mean brightness
    factor = 2 if mean_brightness > 128 else 3  # Example condition, adjust as necessary

    # Multiply x by the determined factor
    xb = factor * x

    # Convert xb to uint8
    y = np.clip(xb, 0, 255).astype(np.uint8)

    # Perform the operation and convert the result to uint8
    # Subtract a value based on mean brightness, multiply by a factor, and clip the result between 0 and 255
    subtract_value = 250 if mean_brightness > 128 else 200  # Example condition, adjust as necessary
    result = np.clip(40 * (y.astype(np.float64) - subtract_value), 0, 255).astype(np.uint8)

    # Display the image
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define the folder path to your images
folder_path = 'Project2_Data/MIRAGE/ProtonSpatial/20210501/run04'

# Process each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.tiff') or filename.endswith('.TIFF'):
        image_path = os.path.join(folder_path, filename)
        process_image(image_path)


