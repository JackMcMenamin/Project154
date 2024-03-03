from flask import Flask, request, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import cv2
import numpy as np
import os
import logging
from PIL import Image
from flask import jsonify
from werkzeug.utils import secure_filename
import re

logging.basicConfig(level=logging.ERROR) 

# Define the base directory of the Flask application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Define the uploads and processed directories paths
uploads_dir = os.path.join(BASE_DIR, 'static', 'uploads')
processed_dir = os.path.join(BASE_DIR, 'static', 'processed')

# Create the directories
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
    
# This helper function extracts numbers from a filename and returns it as an integer
def extract_number(filename):
    parts = re.findall(r'(\d+)', filename)
    return tuple(map(int, parts)) if parts else (0,)

@app.route('/image-detail')
def image_detail():
    # You can add any dynamic content to the template here.
    return render_template('image-detail.html')

@app.route('/process-images', methods=['POST'])
def process_images_endpoint():
    # This will store results for all images
    all_images_results = []
    overall_metrics = {
        'total_images': 0,
        'average_blob_area': 0,
        # Add more overall metrics as needed
    }

    # Check if any file is attached in the request
    if 'images' not in request.files:
        app.logger.error('No images part in the request.')
        return "No images provided", 400

    files = request.files.getlist('images')
    
    # Sort files by the numerical part of their filename
    files.sort(key=lambda x: extract_number(x.filename))
    
    # Loop over each file and process
    for file in files:
        print(f"Sorted filename: {file.filename}")
        if file.filename == '':
            app.logger.error('No selected file.')
            continue

        # Ensure the filename is secure
        filename = secure_filename(file.filename)
        original_image_path = os.path.join(uploads_dir, filename)
        file.save(original_image_path)
        
        # Process the image and get the processing results
        processing_results = process_image(original_image_path)
        
        # Skip files that couldn't be processed
        if processing_results is None:
            continue

        processed_image_path = processing_results['path']
        
        # Update overall metrics here if necessary
        
        # Prepare the response for the individual image
        image_results = {
            'original': os.path.basename(original_image_path),
            'processed': os.path.basename(processed_image_path),
            'metrics': {
                'beam_classification': processing_results['beam_classification'],
                'blob_area': processing_results['blob_area'],
                'blob_center': processing_results['blob_center'],
                'white_area_ratio': processing_results['white_area_ratio'],
            }
        }
        all_images_results.append(image_results)
        
    # Assume `all_images_results` is your list of image results
    all_images_results = sorted(all_images_results, key=lambda x: extract_number(x['original']))
        
    # Calculate overall metrics here if necessary
    
    # Before returning the response, log the results
    app.logger.debug('Overall metrics: %s', overall_metrics)
    app.logger.debug('Image results: %s', all_images_results)
    
    # Return the results for all images
    response = jsonify({
        'overall_metrics': overall_metrics,
        'images': all_images_results
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    app.logger.debug('Response: %s', response.get_data(as_text=True))
    return response

def extract_outline(processed_image_path):
    # Load the processed image
    processed_image = cv2.imread(processed_image_path, cv2.IMREAD_COLOR)
    if processed_image is None:
        app.logger.error(f"Error loading processed image from {processed_image_path}")
        return None

    # Convert to grayscale and threshold to get the mask
    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    return mask


def process_image(image_path):
    
    app.logger.debug(f"Starting to process image: {image_path}")
    
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        app.logger.error(f"Error loading image from {image_path}")
        return None

    # Initialize threshold value and image processing variables
    threshold_value = 230
    max_attempts = 10
    enhancement_factor = 1
    darkening_factor = -10
    beam_classification = None
    attempt = 0

    # Perform adaptive thresholding and classification
    while attempt < max_attempts:
        _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresholded_image)
        total_pixels = image.size
        white_area_ratio = (white_pixels / total_pixels) * 100
        app.logger.debug(f"Attempt {attempt}: White area ratio is {white_area_ratio}%")

        # Adjust image brightness based on classification
        if white_area_ratio < 15:
            beam_classification = 'image too dark'
            image = np.clip(image + enhancement_factor, 0, 255).astype(np.uint8)
        elif white_area_ratio > 40:
            beam_classification = 'image too bright'
            image = np.clip(image + darkening_factor, 0, 255).astype(np.uint8)
        else:
            beam_classification = 'normal beam'
            break  # Exit the loop if image is classified as normal

        attempt += 1
        
    app.logger.debug(f"Beam classification: {beam_classification}")

    if beam_classification == 'normal beam':
        # Further processing if the image is classified as normal
        # Define a kernel for morphological operations
        kernel = np.ones((2, 2), np.uint8)
        opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

        # Find contours and calculate blob properties
        contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            blob_area = cv2.contourArea(largest_contour)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                blob_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # Draw a red 'X' at the blob center
                x_center, y_center = blob_center
                size = 7
                cv2.line(image, (x_center - size, y_center - size), (x_center + size, y_center + size), (0, 0, 255), 2)
                cv2.line(image, (x_center + size, y_center - size), (x_center - size, y_center + size), (0, 0, 255), 2)
        else:
            blob_area, blob_center = None, None
    else:
        # If image is too dark or too bright even after max attempts, classify as bad image
        beam_classification = 'Bad image'
        blob_area, blob_center = None, None

    # Convert and save the processed image in PNG format
    processed_image_filename = os.path.splitext(os.path.basename(image_path))[0] + '.png'
    processed_image_path = os.path.join(processed_dir, processed_image_filename)

    # Perform the final processing and save as PNG
    if beam_classification == 'normal beam' and blob_center is not None:
        color_image = cv2.cvtColor(opened_image, cv2.COLOR_GRAY2BGR)
    else:
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(processed_image_path, color_image)  # Save the final image as PN

    return {
        'path': processed_image_path,
        'beam_classification': beam_classification,
        'blob_area': blob_area,
        'blob_center': blob_center,
        'white_area_ratio': white_area_ratio,
    }

def draw_blob_outline(image_path, processed_image_path):
    #app.logger.debug(f'Starting to draw outline on image: {processed_image_path}')

    # Load the processed image
    image = cv2.imread(processed_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        app.logger.error(f"Error loading processed image from {processed_image_path}")
        return None

    # First, dilate the image to close the gaps
    kernel_dilate = np.ones((15, 15), np.uint8)
    dilated_image = cv2.dilate(image, kernel_dilate, iterations=2)

    # Then, erode the dilated image to make the outline tighter
    kernel_erode = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(dilated_image, kernel_erode, iterations=1)

    # Find contours on the eroded image
    contours, hierarchy = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Load the original processed image in color to draw the red outline
    color_image = cv2.imread(processed_image_path)

    # Simplify contours before drawing them
    simplified_contours = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        simplified_contours.append(simplified_contour)

    # Draw the simplified contours
    cv2.drawContours(color_image, simplified_contours, -1, (0, 0, 255), 2)

    # Save the image with the red outline
    outline_image_filename = os.path.splitext(os.path.basename(processed_image_path))[0] + '_tight_outline.png'
    outline_image_path = os.path.join(processed_dir, outline_image_filename)
    cv2.imwrite(outline_image_path, color_image)
    #app.logger.debug(f'Image with tighter blob outline saved to {outline_image_path}')

    return outline_image_path

def apply_outline_to_original(original_image_path, processed_image_path):
    # Load the processed image and find the contour
    processed_image = cv2.imread(processed_image_path)
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Load the original image
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    
    # Resize the original image to the processed image's size if necessary
    if original_image.shape[:2] != processed_image.shape[:2]:
        original_image = cv2.resize(original_image, (processed_image.shape[1], processed_image.shape[0]))

    # Draw the largest contour and red 'X' onto the original image
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            size = 7  # Size of the 'X'
            cv2.drawContours(original_image, [largest_contour], -1, (0, 0, 255), 2)
            cv2.line(original_image, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 2)
            cv2.line(original_image, (cx + size, cy - size), (cx - size, cy + size), (0, 0, 255), 2)

    # Save the new "Step 3" image
    step3_image_filename = 'step3_' + os.path.basename(original_image_path)
    step3_image_path = os.path.join(processed_dir, step3_image_filename)
    cv2.imwrite(step3_image_path, original_image)

    return step3_image_filename


@app.route('/processed/<filename>')
def processed_file(filename):
    #app.logger.debug(f'Serving processed file: {filename}')
    return send_from_directory('static/processed', filename)

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f'An error occurred: {e}')
    response = jsonify({'error': str(e)})
    response.status_code = (e.code if isinstance(e, HTTPException) else 500)
    return response

def generate_histogram(image_path):
    # Construct the absolute path
    absolute_image_path = os.path.join(BASE_DIR, 'static', image_path)
    #app.logger.debug(f"Attempting to load image at path: {absolute_image_path}")

    # Check if the file exists
    if not os.path.isfile(absolute_image_path):
        app.logger.error(f"File does not exist at path: {absolute_image_path}")
        return []

    # Attempt to open the file with PIL as a sanity check
    try:
        with Image.open(absolute_image_path) as img:
            app.logger.debug(f"Successfully opened image with PIL: {absolute_image_path}")
    except IOError as e:
        app.logger.error(f"Failed to open image with PIL: {e}")
        return []

    # Attempt to read the image in grayscale
    image = cv2.imread(absolute_image_path, 0)
    if image is None:
        app.logger.error(f"Failed to load image with OpenCV at path: {absolute_image_path}")
        return []

    # Proceed with generating histogram
    histogram = np.histogram(image, bins=256, range=(0, 256))[0]
    return histogram.tolist()


@app.route('/')
def index():
    # This will render the index.html template when visiting the root URL
    return render_template('index.html')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True, use_reloader=True)
