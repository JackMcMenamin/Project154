import os
import logging
from werkzeug.utils import secure_filename

class FileManager:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.uploads_dir = self.create_dir('uploads')
        self.processed_dir = self.create_dir('processed')
        self.logger = logging.getLogger('FileManager')
        self.logger.setLevel(logging.INFO)

    def create_dir(self, dir_name):
        """
        Create a directory if it does not exist and return its path.
        """
        dir_path = os.path.join(self.base_dir, 'static', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            self.logger.info(f"Created directory at {dir_path}")
        return dir_path

    def save_uploaded_file(self, file):
        if file.filename == '':
            self.logger.error('No selected file.')
            return None
        filename = secure_filename(file.filename)
        file_path = os.path.join(self.uploads_dir, filename)
        file.save(file_path)
        self.logger.info(f"File saved to {file_path}")
        return file_path

    def create_directory_for_image(self, base_name):
        """
        Create a directory for storing all files related to a specific image and return its path.
        """
        dir_path = os.path.join(self.processed_dir, base_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            self.logger.info(f"Created directory for image {base_name} at {dir_path}")
        return dir_path

    def get_processed_file_path(self, base_name, suffix):
        """
        Construct and return a file path for a processed file based on its base name and a suffix.
        """
        filename = f"{base_name}_{suffix}.png"
        return os.path.join(self.processed_dir, base_name, filename)
    
    def get_directory_for_image(self, base_name):
        """
        Get the directory path for storing all files related to a specific image.
        If the directory does not exist, it will create one.
        """
        dir_path = os.path.join(self.processed_dir, base_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            self.logger.info(f"Created directory for image {base_name} at {dir_path}")
        return dir_path

