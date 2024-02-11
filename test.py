import cv2
import numpy as np

def segment_pill_shape(gray_blurred):
    # Threshold the image to get the pill shape
    _, pill_mask = cv2.threshold(gray_blurred, 30, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    # Clean the mask with morphological operations
    pill_mask_cleaned = cv2.morphologyEx(pill_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return pill_mask_cleaned

def black_out_pill(image, pill_mask):
    # Black out the pill shape in the image
    image_with_pill_blacked_out = image.copy()
    image_with_pill_blacked_out[pill_mask == 255] = 0
    return image_with_pill_blacked_out

def improved_black_out_inside_lines(image, minLineLength, maxLineGap):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Detect lines using the Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    # Check if lines were detected
    if lines is not None:
        # Initialize an empty image for drawing lines
        line_image = np.zeros_like(image)
        
        # Draw the lines on the line image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Convert line image to grayscale
        line_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        # Threshold the line image to get the shapes
        _, line_thresh = cv2.threshold(line_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded line image
        contours, _ = cv2.findContours(line_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask for the areas to be blacked out
        fill_mask = np.zeros_like(gray)
        
        # Fill the contours to create the blackout areas
        cv2.drawContours(fill_mask, contours, -1, 255, cv2.FILLED)
        
        # Create the final mask to blackout the image
        final_mask = cv2.bitwise_not(fill_mask)
        
        # Apply the final mask to the image
        image_with_areas_blacked_out = cv2.bitwise_and(image, image, mask=final_mask)
        
        return image_with_areas_blacked_out
    else:
        # If no lines were detected, return the original image
        return image

# Read the image
image_path = 'Project2_Data/MIRAGE/ProtonSpatial/20210501/run04/shot7.tiff'  # Update the path to your image
image = cv2.imread(image_path)

# Convert to grayscale and apply a blur to reduce noise
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# Segment the pill shape
pill_mask = segment_pill_shape(gray_blurred)

# Black out the pill shape in the image
image_with_pill_blacked_out = black_out_pill(image, pill_mask)

# Black out the areas inside the lines in the image
final_image = improved_black_out_inside_lines(image_with_pill_blacked_out, minLineLength=100, maxLineGap=10)

# Display the original image and the final image with the areas inside the lines blacked out
combined = np.concatenate((image, final_image), axis=1)
cv2.imshow('Original and Final Image with Areas Inside Lines Blacked Out', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()


#works with overall circle and pill shape






