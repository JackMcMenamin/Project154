from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import logging
from image_processing import ImageProcessor
from file_handling import FileManager
from utils import extract_number
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
logging.basicConfig(level=logging.INFO)

file_manager = FileManager()
image_processor = ImageProcessor()

@app.route('/process-images', methods=['POST'])
def process_images_endpoint():
    logging.info("Processing images endpoint called.")
    files = request.files.getlist('images')
    if not files:
        logging.warning("No images provided in the request.")
        return jsonify({"error": "No images provided"}), 400
    
    processed_images_info = image_processor.process_images(files)
    
    # The processed_images_info should now contain the metrics
    return jsonify({'images': processed_images_info})

@app.route('/get-intermediate-images')
def get_intermediate_images():
    image_name = request.args.get('name')
    logging.info(f"Fetching intermediate images for: {image_name}")

    # Assuming that the folder name is the same as the image name
    image_directory = os.path.join(app.static_folder, 'processed', image_name)
    logging.info(f"Looking for images in: {image_directory}")

    if not os.path.isdir(image_directory):
        logging.error(f"Directory not found: {image_directory}")
        return jsonify({"error": "Directory not found"}), 404

    # List all image files in the directory
    try:
        files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]
        logging.info(f"Found files: {files}")
        return jsonify({"intermediate_images": files})
    except Exception as e:
        logging.error(f"Error retrieving files: {str(e)}")
        return jsonify({"error": "Error retrieving files"}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image-detail')
def image_detail():
    image_name = request.args.get('name')
    # Use the new get_directory_for_image method to get the directory path
    image_directory = file_manager.get_directory_for_image(image_name)

    # Assuming you want to list all files in this directory
    files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

    # Pass the files to your template
    return render_template('image-detail.html', image_name=image_name, files=files)

if __name__ == '__main__':
    app.run(debug=True)
