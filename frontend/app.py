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
from math import sqrt

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
    app.logger.debug(f"Full URL requested: {request.url}")
    image_name_with_ext = request.args.get('name', '')
    base_name = os.path.splitext(image_name_with_ext)[0]  # This should give you "run04_Shot8" from "run04_Shot8.TIFF"
    app.logger.debug(f"Image name received: {base_name}")
    
    intermediate_dir = os.path.join(processed_dir, base_name)

    if not base_name:
        # Handle the error when image_name is None
        app.logger.error('No image name provided.')
        return "Error: No image name provided.", 400

    # Based on the image name, find the folder with the intermediate images
    intermediate_dir = os.path.join(processed_dir, base_name)

    # List all files in the directory
    image_files = [file for file in os.listdir(intermediate_dir) if file.endswith('.png')]

    # Sort the files by step (if you have a numbering system in the filenames)
    image_files.sort(key=lambda x: extract_number(x))

    # Return a template with the list of image file paths
    return render_template('image-detail.html', image_files=image_files)

@app.route('/get-intermediate-images')
def get_intermediate_images():
    image_name = request.args.get('name')
    if not image_name:
        return jsonify({'error': 'No image name provided'}), 400

    base_name = os.path.splitext(image_name)[0]
    intermediate_dir = os.path.join(processed_dir, base_name)
    if not os.path.exists(intermediate_dir):
        return jsonify({'error': 'Directory not found'}), 404

    # Assuming all the intermediate images are in this directory
    intermediate_images = [f for f in os.listdir(intermediate_dir) if f.endswith('.png')]
    intermediate_images.sort()  # Optionally sort the files if needed

    return jsonify({'intermediate_images': intermediate_images})

@app.route('/process-images', methods=['POST'])
def process_images_endpoint():
    # This will store results for all images
    all_images_results = []
    overall_metrics = {
        'total_images': 0,
        'bad_images': 0,
        'average_blob_area': 0,
        'blob_area_std_dev': 0,  # To calculate standard deviation
        'average_intensity': 0,
        'intensity_std_dev': 0,  # To calculate standard deviation
        'average_blob_center_offset': 0,  # If there is a known target center
        # Add more overall metrics as needed
    }
    
    # You would also need to accumulate data for each image to calculate these
    blob_areas = []
    intensities = []
    center_offsets = []

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
                'contour': os.path.basename(processing_results.get('contour_path', '')),
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
            
            #blob_areas.append(processing_results['blob_area'])
            #intensities.append(calculate_average_intensity(some_image_data))
            #center_offsets.append(calculate_center_offset(processing_results['blob_center']))
        
    #if overall_metrics['total_images'] > 0:
        #overall_metrics['average_blob_area'] = sum(blob_areas) / overall_metrics['total_images']
        #overall_metrics['blob_area_std_dev'] = sqrt(sum((x - overall_metrics['average_blob_area'])**2 for x in blob_areas) / overall_metrics['total_images'])
        #overall_metrics['average_intensity'] = sum(intensities) / overall_metrics['total_images']
        #overall_metrics['intensity_std_dev'] = sqrt(sum((x - overall_metrics['average_intensity'])**2 for x in intensities) / overall_metrics['total_images'])
        #overall_metrics['average_blob_center_offset'] = sum(center_offsets) / overall_metrics['total_images']
    
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

    # Prepare a directory for intermediate images
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    intermediate_dir = os.path.join(processed_dir, base_name)
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)

    # Save the original image
    original_image_path = os.path.join(intermediate_dir, base_name + '_original.png')
    cv2.imwrite(original_image_path, original_image)

    # Convert to grayscale and save
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_image_path = os.path.join(intermediate_dir, base_name + '_gray.png')
    cv2.imwrite(gray_image_path, gray_image)

    # Initialize threshold value and image processing variables
    threshold_value = 100 #90
    max_attempts = 110 #110
    enhancement_factor = 1
    darkening_factor = 1 #10
    beam_classification = None
    attempt = 0
    white_area_ratio = None

    # Perform adaptive thresholding and classification
    last_brightness_adjusted_image_path = None  # Initialize variable to hold the path of the last brightness-adjusted image
    
    # Initialize a variable to hold the path of the final thresholded image
    final_thresholded_image_path = None
    contour_overlay_image = None  # Initialize here to ensure it exists outside the if block
    
    while attempt < max_attempts:
        _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        
        thresholded_image_path = os.path.join(intermediate_dir, f'{base_name}_thresholded_{attempt}.png')
        cv2.imwrite(thresholded_image_path, thresholded_image)
        
        # Update the path for the last attempt (which will be the final threshold)
        final_thresholded_image_path = thresholded_image_path

        white_pixels = cv2.countNonZero(thresholded_image)
        total_pixels = gray_image.size
        white_area_ratio = (white_pixels / total_pixels) * 100
        app.logger.debug(f"Attempt {attempt}: White area ratio is {white_area_ratio}%")

        if white_area_ratio < 15:
            beam_classification = 'image too dark'
            gray_image = np.clip(gray_image + enhancement_factor, 0, 255).astype(np.uint8)
        elif white_area_ratio > 40:
            beam_classification = 'image too bright'
            gray_image = np.clip(gray_image - darkening_factor, 0, 255).astype(np.uint8)
        else:
            beam_classification = 'normal beam'
            # Save the last brightness-adjusted image before exiting loop
            if last_brightness_adjusted_image_path:
                cv2.imwrite(last_brightness_adjusted_image_path, gray_image)
                
            # Save the final thresholded image here
            final_thresholded_image_path = os.path.join(intermediate_dir, f'{base_name}_final_threshold.png')
            cv2.imwrite(final_thresholded_image_path, thresholded_image)
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

            # Create a copy of the thresholded image to draw on
            contour_overlay_image = thresholded_image.copy()

            # Convert to BGR to draw colored contours
            contour_overlay_image = cv2.cvtColor(contour_overlay_image, cv2.COLOR_GRAY2BGR)

            # Draw the comprehensive outline onto the contour_overlay_image
            cv2.drawContours(contour_overlay_image, [hull], -1, (0, 0, 255), 2)

            # Optionally, draw red 'X' at the centroid of the hull
            M = cv2.moments(hull)
            if M["m00"] != 0:
                cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                size = 10
                cv2.line(contour_overlay_image, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 2)
                cv2.line(contour_overlay_image, (cx + size, cy - size), (cx - size, cy + size), (0, 0, 255), 2)
    else:
        # If image is too dark or too bright even after max attempts, classify as bad image
        beam_classification = 'Bad image'
        # Handling for non-'normal beam' cases...
        contour_overlay_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)

    # Save the contour image (or final processed image in your case)
    contour_image_path = os.path.join(intermediate_dir, f'{base_name}_contour.png')
    if contour_overlay_image is not None:
        cv2.imwrite(contour_image_path, contour_overlay_image)
        final_image_path = contour_image_path  # Assign the contour image path as the final image path
    else:
        # Handle cases where no contour image is generated
        # This could include saving the original, thresholded, or a blank image as a placeholder
        # For the purpose of illustration, we're defaulting to the original image's path
        final_image_path = original_image_path

    return {
        'final_path': final_image_path,
        'final_thresholded_image': os.path.basename(final_thresholded_image_path) if final_thresholded_image_path else None,
        'beam_classification': beam_classification,
        'blob_area': None,  # Calculate and fill this if needed
        'blob_center': None,  # Calculate and fill this if needed
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
