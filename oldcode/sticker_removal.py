import cv2
import numpy as np
import matplotlib.pyplot as plt
from Project2_Data import *

# Define the file path to your image
image_file_path = 'Project2_Data/MIRAGE/ProtonSpatial/20210501/run04/Shot7.tiff'  # Replace with your actual image file path

# Manually defined coordinates for the quadrilateral sticker
sticker_points = np.array([
    [266, 31],  # Top left (yellow)
    [237, 47],  # Bottom left (green)
    [555, 314], # Top right (pink)
    [542, 341], # Bottom right (blue)
], dtype=np.int32)

# Function to create and display a mask for the defined sticker area
def create_and_verify_sticker_mask(shape, sticker_points):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [sticker_points], 255)

    # Display the mask to verify its correctness
    plt.imshow(mask, cmap='gray')
    plt.title('Verification Mask')
    plt.axis('off')
    plt.show()

    return mask

# Function to load, mask, and inpaint the image
def inpaint_stickers(file_path, sticker_points):
    # Read the image
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Create and verify the mask for the sticker
    mask = create_and_verify_sticker_mask(img.shape, sticker_points)

    # Inpaint the image, adjust the radius if necessary
    inpainted_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return inpainted_img

# Process the image and display it
inpainted_image = inpaint_stickers(image_file_path, sticker_points)
plt.imshow(inpainted_image, cmap='gray')
plt.title('Inpainted Image')
plt.axis('off')
plt.show()