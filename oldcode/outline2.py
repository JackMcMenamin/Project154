import cv2
import os

def process_image(image_path, threshold_value):
    # Load the image in grayscale
    x = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded correctly
    if x is None:
        print(f"Error loading image from {image_path}")
        return

    # Apply thresholding
    _, thresholded_image = cv2.threshold(x, threshold_value, 255, cv2.THRESH_BINARY)

    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Apply morphological opening (erosion followed by dilation)
    # Adjust the kernel size and shape as necessary for your images
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

    # Display the opened image
    cv2.imshow('Opened Image', opened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define the folder path to your images
folder_path = 'Project2_Data/MIRAGE/ProtonSpatial/20210501/run04'

# Define a threshold value
threshold_value = 100  # Adjust this value based on your requirement

# Process each image in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.tiff', '.tif')):
        image_path = os.path.join(folder_path, filename)
        process_image(image_path, threshold_value)



