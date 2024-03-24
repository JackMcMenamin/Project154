import cv2
import numpy as np
import os

def process_and_display_image(image_path):
    print(f"Starting to process image: {image_path}")

    # Load the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        print(f"Error loading image from {image_path}")
        return

    # Display the original image
    cv2.imshow('Original Image', original_image)
    cv2.waitKey(0)

    # Convert to grayscale and display
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.waitKey(0)

    # Perform adaptive thresholding to separate objects and display
    _, thresholded_image = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY)
    cv2.imshow('Thresholded Image', thresholded_image)
    cv2.waitKey(0)

    # Find contours
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image and display
    if contours:
        cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours Image', original_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

# Specify the path to your image
image_path = 'Project2_Data/MIRAGE/ProtonSpatial/20210501/run04/shot7.tiff' # Update this path to your specific image file path
process_and_display_image(image_path)

