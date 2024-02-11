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

# Create the directories if they don't exist
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
    converted_image_path = original_image_path  # Assume no conversion needed by default

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
    processed_image_path = process_image(converted_image_path)

    # After processing the image and before returning the response
    processed_histogram = generate_histogram(processed_image_path)  # Get histogram for the processed image

    # After processing the image and generating histograms
    response = jsonify({
        'original': os.path.basename(converted_image_path),
        'processed': os.path.basename(processed_image_path),
        'original_histogram': original_histogram,
        'processed_histogram': processed_histogram
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

def process_image(image_path):
    
    app.logger.debug(f'Starting to process image: {image_path}')
    
    # Load the image in grayscale
    x = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded correctly
    if x is None:
        print(f"Error loading image from {image_path}")
        return
    
    # Compute the mean brightness of the image
    mean_brightness = np.mean(x)
    # You can also compute the standard deviation if needed
    # std_deviation = np.std(x)
    
    # Convert image to double precision (float64)
    x = x.astype(np.float64)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    x = clahe.apply(x.astype(np.uint8))
    
    # Adjust the multiplication factor based on mean brightness
    factor = 2 if mean_brightness > 128 else 3  # Example condition, adjust as necessary

    # Multiply x by the determined factor
    xb = factor * x

    # Convert xb to uint8
    y = np.clip(xb, 0, 255).astype(np.uint8)

    # Perform the operation and convert the result to uint8
    # Subtract a value based on mean brightness, multiply by a factor, and clip the result between 0 and 255
    subtract_value = 250 if mean_brightness > 128 else 200  # Example condition, adjust as necessary
    result = np.clip(40 * (y.astype(np.float64) - subtract_value), 0, 255).astype(np.uint8)
    
    app.logger.debug(f'Finished processing image: {image_path}')
    
    # Use the 'processed_dir' for the processed image path
    processed_image_filename = os.path.basename(image_path)
    processed_image_path = os.path.join(processed_dir, processed_image_filename)


    app.logger.debug(f'Saving processed image to {processed_image_path}')
    cv2.imwrite(processed_image_path, result)
    app.logger.debug(f'Processed image saved successfully.')
    
    # Return the relative path from the 'static' directory for the frontend
    return os.path.join('processed', processed_image_filename)

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
    return render_template('index.html')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)