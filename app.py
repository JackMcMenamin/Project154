import cv2
import numpy as np
from PIL import Image
import os
import logging

logging.basicConfig(level=logging.DEBUG)

# Define the base directory
BASE_DIR = os.getcwd()
output_dir = os.path.join(BASE_DIR, 'output')

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to convert TIFF to PNG
def convert_tiff_to_png(input_path):
    tiff_image = Image.open(input_path)
    png_filename = os.path.splitext(os.path.basename(input_path))[0] + '.png'
    converted_image_path = os.path.join(BASE_DIR, png_filename)
    tiff_image.save(converted_image_path, 'PNG')
    tiff_image.close()
    logging.debug(f"Converted TIFF to PNG: {converted_image_path}")
    return converted_image_path

# Function to process the image
def process_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logging.error(f"Error loading image from {image_path}")
        return None

    # Perform thresholding
    _, thresholded_image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)

    # Morphological opening to remove noise
    kernel = np.ones((2, 2), np.uint8)
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw the largest contour in red
        color_image = cv2.cvtColor(opened_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(color_image, [largest_contour], -1, (0, 0, 255), 2)

        # Save the processed image
        processed_image_filename = os.path.splitext(os.path.basename(image_path))[0] + '_processed.png'
        processed_image_path = os.path.join(output_dir, processed_image_filename)
        cv2.imwrite(processed_image_path, color_image)
        logging.debug(f"Processed image saved to: {processed_image_path}")

        return processed_image_path

# Main code
if __name__ == '__main__':
    # Replace this with your TIFF image path
    input_file_path = 'Project2_Data/MIRAGE/ProtonSpatial/20210501/run04/shot7.tiff'

    # Check if the file is TIFF and convert to PNG
    if input_file_path.lower().endswith(('.tiff', '.tif')):
        input_file_path = convert_tiff_to_png(input_file_path)

    # Process the image
    output_file_path = process_image(input_file_path)

    if output_file_path:
        print(f"Processed image saved at: {output_file_path}")
    else:
        print("Failed to process the image.")

