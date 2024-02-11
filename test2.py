import cv2
import numpy as np

def find_and_fill_pill(gray_image):
    # Apply a threshold to get a binary image
    _, binary_image = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the pill is the largest contour by area that is not the full size of the image
    pill_contour = max(
        (c for c in contours if cv2.contourArea(c) < gray_image.size),
        key=cv2.contourArea,
        default=None
    )
    
    # If we didn't find a suitable contour, return None
    if pill_contour is None:
        return None
    
    # Create a mask for the pill shape
    pill_mask = np.zeros_like(gray_image)
    cv2.drawContours(pill_mask, [pill_contour], -1, 255, thickness=cv2.FILLED)
    
    # Fill the pill shape with white on the mask
    pill_filled = np.zeros_like(gray_image)
    pill_filled[pill_mask == 255] = 255
    return pill_filled

# Read the image from the provided path
image_path = 'Project2_Data/MIRAGE/ProtonSpatial/20210501/run04/shot7.tiff'  # Replace with your image path
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find the pill shape and fill it with white
pill_filled_image = find_and_fill_pill(gray_image)

# If we found and filled a pill, show it
if pill_filled_image is not None:
    # Convert single channel mask to a 3 channel image to concatenate with the original
    pill_filled_image_colored = cv2.cvtColor(pill_filled_image, cv2.COLOR_GRAY2BGR)
    
    # Show the original and processed images side by side
    combined = np.concatenate((image, pill_filled_image_colored), axis=1)
    cv2.imshow('Original and Processed Image', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No suitable pill contour found.")






