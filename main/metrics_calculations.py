import cv2
import numpy as np
from image_processing import ImageProcessor

def calculate_light_intensity(image_path, contour):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # If the image couldn't be loaded, return None
    if image is None:
        return None
    
    # Create a mask image that is the same size as the original and fill it with zeros (black)
    mask = np.zeros_like(image)
    
    # Fill the contour on the mask with white color
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Bitwise AND the mask and the original image to isolate the blob
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Calculate the sum of all pixel values in the masked image
    # Assuming pixel value corresponds to intensity
    total_intensity = np.sum(masked_image)
    
    # Calculate the number of pixels within the blob to find the average intensity
    area = cv2.contourArea(contour)
    average_intensity = total_intensity / area if area != 0 else 0
    
    # Optionally, calibration to physical units can be done here
    
    # Return the total and average intensity
    return total_intensity, average_intensity

if __name__ == "__main__":
    # Example usage - you'd replace 'path_to_image' and 'contour' with actual values
    path_to_image = 'path_to_your_image.png'
    contour = np.array([[x1, y1], [x2, y2], ...])  # replace with actual contour coordinates
    
    # Calculate the intensity
    total_intensity, average_intensity = calculate_light_intensity(path_to_image, contour)
    
    print(f"Total Intensity: {total_intensity}")
    print(f"Average Intensity: {average_intensity}")
