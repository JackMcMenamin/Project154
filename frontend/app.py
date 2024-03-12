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
        if file.filename == '':
            app.logger.error('No selected file.')
            continue

        # Ensure the filename is secure
        filename = secure_filename(file.filename)
        original_image_path = os.path.join(uploads_dir, filename)
        file.save(original_image_path)
        
        # Process the image and overlay the blob outline on the original image
        processing_results = process_and_draw_image(original_image_path)
        
        if processing_results:
            # Prepare the response for the individual image
            image_results = {
                'original': os.path.basename(original_image_path),
                'processed': os.path.basename(processing_results['final_path']),  # This is the final image with the overlay
                'metrics': {
                    'beam_classification': processing_results['beam_classification'],
                    'blob_area': processing_results['blob_area'],
                    'blob_center': processing_results['blob_center'],
                    'white_area_ratio': processing_results['white_area_ratio'],
                }
            }
            all_images_results.append(image_results)
            overall_metrics['total_images'] += 1
            overall_metrics['average_blob_area'] += processing_results['blob_area'] or 0
        
    # Calculate overall metrics here if necessary
    if overall_metrics['total_images'] > 0:
        overall_metrics['average_blob_area'] = overall_metrics['average_blob_area'] / overall_metrics['total_images']
    
    # Sort the image results based on the numerical part of the original filename
    all_images_results = sorted(all_images_results, key=lambda x: extract_number(x['original']))
        
    # Log the overall metrics and image results
    app.logger.debug('Overall metrics: %s', overall_metrics)
    app.logger.debug('Image results: %s', all_images_results)
    
    # Return the results for all images
    response = jsonify({
        'overall_metrics': overall_metrics,
        'images': all_images_results
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


def process_and_draw_image(image_path):
    app.logger.debug(f"Starting to process image: {image_path}")

    # Load the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        app.logger.error(f"Error loading image from {image_path}")
        return None

    # Convert to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Initialize threshold value and image processing variables
    threshold_value = 90
    max_attempts = 110
    enhancement_factor = 1
    darkening_factor = 10
    beam_classification = None
    attempt = 0
    white_area_ratio = None

    # Perform adaptive thresholding and classification
    while attempt < max_attempts:
        _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresholded_image)
        total_pixels = gray_image.size
        white_area_ratio = (white_pixels / total_pixels) * 100
        app.logger.debug(f"Attempt {attempt}: White area ratio is {white_area_ratio}%")

        # Adjust image brightness based on classification
        if white_area_ratio < 15:
            beam_classification = 'image too dark'
            gray_image = np.clip(gray_image + enhancement_factor, 0, 255).astype(np.uint8)
        elif white_area_ratio > 40:
            beam_classification = 'image too bright'
            gray_image = np.clip(gray_image + darkening_factor, 0, 255).astype(np.uint8)
        else:
            beam_classification = 'normal beam'
            break  # Exit the loop if image is classified as normal

        attempt += 1

    blob_area = None
    blob_center = None

    # Further processing if the image is classified as normal
    if beam_classification == 'normal beam':
        # Define a kernel for morphological operations
        kernel = np.ones((5, 5), np.uint8)  # Size of the kernel adjusted for better contouring
        opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

        # Find contours and calculate blob properties
        contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # We can use cv2.convexHull to create a single outline around multiple contours
            all_contours = np.vstack([contours[i] for i in range(len(contours))])
            hull = cv2.convexHull(all_contours)
            
            # Draw the comprehensive outline onto the original image
            cv2.drawContours(original_image, [hull], -1, (0, 0, 255), 2)
            
            # Optionally, draw red 'X' at the centroid of the hull if needed
            M = cv2.moments(hull)
            if M["m00"] != 0:
                cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                size = 10
                cv2.line(original_image, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 2)
                cv2.line(original_image, (cx + size, cy - size), (cx - size, cy + size), (0, 0, 255), 2)
    else:
        # If image is too dark or too bright even after max attempts, classify as bad image
        beam_classification = 'Bad image'

    # Save the final image with the comprehensive red blob outline in PNG format
    final_image_filename = os.path.splitext(os.path.basename(image_path))[0] + '_processed.png'
    final_image_path = os.path.join(processed_dir, final_image_filename)
    cv2.imwrite(final_image_path, original_image)

    return {
        'final_path': final_image_path,
        'beam_classification': beam_classification,
        'blob_area': blob_area,
        'blob_center': blob_center,
        'white_area_ratio': white_area_ratio,
    }


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
