from flask import Flask, request, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import cv2
import numpy as np
import os
import logging
from PIL import Image
from flask import jsonify

logging.basicConfig(level=logging.DEBUG)

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

@app.route('/process-image', methods=['POST'])
def process_image_endpoint():
    app.logger.debug('Received a request to process an image.')
    if 'image' not in request.files:
        app.logger.error('No image part in the request.')
        return "No image provided", 400

    file = request.files['image']
    app.logger.debug(f'Received file: {file.filename}')

    # Check if the file is TIFF and convert to PNG
    original_image_path = os.path.join(uploads_dir, file.filename)
    converted_image_path = original_image_path 

    # Convert TIFF to PNG before processing
    if file.filename.lower().endswith(('.tiff', '.tif')):
        app.logger.debug('Converting TIFF to PNG')
        tiff_image = Image.open(file.stream)  # Open the image directly from the file stream
        png_filename = os.path.splitext(file.filename)[0] + '.png'
        converted_image_path = os.path.join(uploads_dir, png_filename)  # PNG file path
        tiff_image.save(converted_image_path)  # Save the PNG image
        tiff_image.close()
    else:
        # Save the original file
        file.save(original_image_path)

    app.logger.debug(f'Saving uploaded image to: {converted_image_path}')

    # Process the image and get the path of the processed image
    processed_image_path = process_image(converted_image_path)
    app.logger.debug(f'Processed image saved to: {processed_image_path}')

    # After saving the original image and before processing it
    original_histogram = generate_histogram(converted_image_path)  # Get histogram for the original image

    # Process the image and get the path of the processed image
    processing_results = process_image(converted_image_path)
    processed_image_path = processing_results['path']

    # After processing the image and before returning the response
    processed_histogram = generate_histogram(processed_image_path)  # Get histogram for the processed image

    # After processing the image and generating histograms
    response = jsonify({
        'original': os.path.basename(converted_image_path),
        'processed': os.path.basename(processed_image_path),  # Note: this uses the path from the dictionary
        'original_histogram': original_histogram,
        'processed_histogram': processed_histogram,
        # Metrics
        'beam_classification': processing_results['beam_classification'],
        'blob_area': processing_results['blob_area'],
        'blob_center': processing_results['blob_center'] if processing_results['blob_center'] else "Not applicable"
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

def process_image(image_path):
    app.logger.debug(f'Starting to process image: {image_path}')
    
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        app.logger.error(f"Error loading image from {image_path}")
        return None
    
    attempt = 0
    max_attempts = 5
    enhancement_factor = 1.5
    beam_classification = 'image too dark'
    
    while attempt < max_attempts:
        _, thresholded_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresholded_image)
        total_pixels = image.size
        white_area_ratio = (white_pixels / total_pixels) * 100

        beam_classification = 'normal beam' if white_area_ratio >= 20 else 'image too dark'
        app.logger.debug(f'Attempt {attempt}: Beam Classification: {beam_classification}, White Area Ratio: {white_area_ratio:.2f}%')

        if beam_classification == 'normal beam':
            # If the image is already bright enough, break the loop and do not enhance
            break
        else:
            # If the image is too dark, enhance the image brightness
            image = np.clip(image * enhancement_factor, 0, 255).astype(np.uint8)
            app.logger.debug(f'Enhanced image brightness on attempt {attempt+1}')

        attempt += 1
    
    if attempt == max_attempts and beam_classification == 'image too dark':
        # If after the maximum attempts the image is still too dark, classify it as 'Bad image'
        beam_classification = 'Bad image'
    # Define a kernel for morphological operations to clean up the image - noise suppression
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

    # Calculate blob area and center only if the image has been classified as 'normal beam'
    blob_area, blob_center = None, None
    if beam_classification == 'normal beam':
        blob_area = white_pixels
        contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                blob_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # Save the processed image regardless of classification
    processed_image_filename = os.path.basename(image_path)
    processed_image_path = os.path.join(processed_dir, processed_image_filename)
    cv2.imwrite(processed_image_path, opened_image)
    app.logger.debug(f'Processed image saved to {processed_image_path}')

    return {
        'path': os.path.join('processed', processed_image_filename),
        'beam_classification': beam_classification,
        'blob_area': blob_area,
        'blob_center': blob_center
    }


@app.route('/processed/<filename>')
def processed_file(filename):
    app.logger.debug(f'Serving processed file: {filename}')
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
    app.logger.debug(f"Attempting to load image at path: {absolute_image_path}")

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
    app.run(debug=True)