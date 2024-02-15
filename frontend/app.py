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

    # Process the image and get the processing results
    processing_results = process_image(converted_image_path)
    processed_image_path = processing_results['path']  # This is the path without the outline

    # Now, we update the processed_image_path with the outline image path
    if 'outline_image' in processing_results and processing_results['outline_image']:
        processed_image_path = os.path.join(processed_dir, processing_results['outline_image'])

    app.logger.debug(f'Processed image saved to: {processed_image_path}')

    # After saving the original image and before returning the response
    original_histogram = generate_histogram(converted_image_path)
    processed_histogram = generate_histogram(processed_image_path)

    # Prepare the response with the path of the image with the outline
    response = jsonify({
        'original': os.path.basename(converted_image_path),
        'processed': os.path.basename(processed_image_path),  # Updated to use the outline image
        'original_histogram': original_histogram,
        'processed_histogram': processed_histogram,
        # Metrics
        'beam_classification': processing_results['beam_classification'],
        'blob_area': processing_results['blob_area'],
        'blob_center': processing_results['blob_center'] if processing_results['blob_center'] else "Not applicable",
        'white_area_ratio': round(processing_results['white_area_ratio'], 2)
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
    
    threshold_value = 230
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    print("Unique pixel values after thresholding:", np.unique(thresholded_image))
    
    # Check if the thresholding is as expected
    if np.any(thresholded_image[thresholded_image < threshold_value]):
        app.logger.error("Thresholding not performed as expected")
    
    attempt = 0
    max_attempts = 110
    enhancement_factor = 1
    darkening_factor = 10  # Less than 1 to darken the image
    beam_classification = None
    
    while attempt < max_attempts:
        _, thresholded_image = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresholded_image)
        total_pixels = image.size
        white_area_ratio = (white_pixels / total_pixels) * 100

        if white_area_ratio < 15:
            beam_classification = 'image too dark'
            image = np.clip(image + enhancement_factor, 0, 255).astype(np.uint8)  # Enhance brightness
        elif white_area_ratio > 40:
            beam_classification = 'image too bright'
            image = np.clip(image + darkening_factor, 0, 255).astype(np.uint8)  # Darken image
        else:
            beam_classification = 'normal beam'
            break

        app.logger.debug(f'Attempt {attempt}: Beam Classification: {beam_classification}, White Area Ratio: {white_area_ratio:.2f}%')
        attempt += 1
    
    if attempt == max_attempts and (beam_classification == 'image too dark' or beam_classification == 'image too bright'):
        beam_classification = 'Bad image'
        
    # Define a kernel for morphological operations to clean up the image - noise suppression
    kernel = np.ones((2,2),np.uint8)
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
                
                
    if beam_classification == 'normal beam' and blob_center is not None:
        # Load the image in color to put a colored 'X'
        color_image = cv2.cvtColor(opened_image, cv2.COLOR_GRAY2BGR)

        # Calculate the coordinates for the red 'X'
        x_center, y_center = blob_center
        size = 7  # Size of the 'X'
        # Draw two intersecting lines to form an 'X'
        cv2.line(color_image, (x_center - size, y_center - size), (x_center + size, y_center + size), (0, 0, 255), 2)
        cv2.line(color_image, (x_center + size, y_center - size), (x_center - size, y_center + size), (0, 0, 255), 2)

        # Save the color image with the red 'X'
        processed_image_filename = os.path.splitext(os.path.basename(image_path))[0] + '_marked.png'
        processed_image_path = os.path.join(processed_dir, processed_image_filename)
        cv2.imwrite(processed_image_path, color_image)  # Save the image with the red 'X'
        app.logger.debug(f'Processed image with blob center marked as red \'X\' saved to {processed_image_path}')
    else:
        # Save the grayscale image if no blob center is marked
        processed_image_filename = os.path.basename(image_path)
        processed_image_path = os.path.join(processed_dir, processed_image_filename)
        cv2.imwrite(processed_image_path, opened_image)
        app.logger.debug(f'Processed image saved to {processed_image_path}')


    white_pixels = cv2.countNonZero(thresholded_image)
    total_pixels = image.size
    white_area_ratio = (white_pixels / total_pixels) * 100
    print(white_area_ratio)
    
    # Call the new function to draw the blob outline
    outline_image_path = draw_blob_outline(image_path, processed_image_path)

    return {
        'path': os.path.join('processed', processed_image_filename),
        'beam_classification': beam_classification,
        'blob_area': white_pixels if beam_classification == 'normal beam' else None,
        'blob_center': blob_center if beam_classification == 'normal beam' else None,
        'white_area_ratio': white_area_ratio,
        'outline_image': os.path.basename(outline_image_path) if outline_image_path else None
    }
    
def draw_blob_outline(image_path, processed_image_path):
    app.logger.debug(f'Starting to draw outline on image: {processed_image_path}')

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
    app.logger.debug(f'Image with tighter blob outline saved to {outline_image_path}')

    return outline_image_path


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