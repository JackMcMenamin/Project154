import unittest
import cv2
import os
from image_processing import ImageProcessor
from metrics_calculations import BeamMetricsCalculator
import numpy as np

class TestImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialise the image processor just once for all tests
        cls.processor = ImageProcessor()

        # Set up the path to the test image
        cls.script_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_image_path = os.path.join(cls.script_dir, 'static', 'uploads', 'run04_Shot7.png')
        cls.test_corrupt = os.path.join(cls.script_dir, 'static', 'test', 'run04_Shot7.TIFF')
        cls.test_exist = os.path.join(cls.script_dir, 'static', 'test', 'run04_Shot20.png')

        # Load the test image
        cls.test_image = cv2.imread(cls.test_image_path, cv2.IMREAD_COLOR)
        if cls.test_image is None:
            raise FileNotFoundError(f"Test image not found at path: {cls.test_image_path}")

    def setUp(self):
        # Set up for each individual test
        self.processor = ImageProcessor()
        self.gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        self.base_name = 'test_image'  # Defining a base name for the test
        self.intermediate_dir = self.processor.file_manager.create_directory_for_image(self.base_name)
        self.thresholded_image, _, _, _ = self.processor.apply_adaptive_thresholding(
            self.test_image, self.gray_image, self.base_name, self.intermediate_dir
        )
        
    def test_load_corrupted_image(self):
        """Test loading a corrupted or unreadable image file."""
        processor = ImageProcessor()
        result = processor.process_and_draw_image(self.test_corrupt)
        self.assertIsNone(result, "Failed to handle corrupted image correctly.")

    def test_non_existing_file(self):
        """Test handling of non-existing file."""
        processor = ImageProcessor()
        result = processor.process_and_draw_image(self.test_exist)
        self.assertIsNone(result, "Failed to handle non-existing image file correctly.")    

    def test_load_image(self):
        """Test image loading functionality."""
        image = cv2.imread(self.test_image_path, cv2.IMREAD_COLOR)
        self.assertIsNotNone(image, "Loading image failed.")

    def test_convert_to_grayscale(self):
        """Test grayscale conversion."""
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        self.assertEqual(gray_image.ndim, 2, "Grayscale conversion failed: Image not 2D.")

    def test_thresholding(self):
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        base_name = 'test_image' 
        intermediate_dir = self.processor.file_manager.create_directory_for_image(base_name)
        thresholded_image, _, _, _ = self.processor.apply_adaptive_thresholding(
            self.test_image, gray_image, base_name, intermediate_dir
        )
        self.assertEqual(thresholded_image.ndim, 2, "Thresholding failed: Resultant image is not 2D.")

    def test_contour_detection(self):
        """Test contour detection."""
        contours = self.processor.normal_image_processing(
            self.thresholded_image, 
            self.test_image, 
            self.base_name, 
            self.intermediate_dir
        )
        self.assertIsNotNone(contours, "Contour detection failed: No contours detected.")

    def test_beam_classification(self):
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
        classification = self.processor.beam_classification(thresholded_image)
        self.assertIn(classification, ['normal beam', 'bad image'], "Beam classification failed.")

    def test_blob_extraction(self):
        """Test blob extraction from the image."""
        extracted_blob_path = self.processor.extract_blob_contents(
            self.test_image_path, 
            self.test_image_path, 
            'normal beam'
        )
        self.assertTrue(os.path.exists(extracted_blob_path), "Blob extraction failed: No file created.")
        
class TestAdaptiveThresholding(unittest.TestCase):
    def test_adaptive_thresholding_dark_image(self):
        processor = ImageProcessor()
        dark_image = np.zeros((100, 100, 3), dtype=np.uint8)
        dark_gray_image = cv2.cvtColor(dark_image, cv2.COLOR_BGR2GRAY)
        dark_image_path = os.path.join(processor.file_manager.create_directory_for_image('dark_image'), 'dark_image.png')
        cv2.imwrite(dark_image_path, dark_image)
        _, _, classification, _ = processor.apply_adaptive_thresholding(dark_image, dark_gray_image, 'dark_image', processor.file_manager.create_directory_for_image('dark_image'))
        self.assertEqual(classification, 'bad image', "Dark image not classified as 'bad image'.")

    def test_adaptive_thresholding_bright_image(self):
        processor = ImageProcessor()
        bright_image = np.full((100, 100, 3), 255, dtype=np.uint8)  # Creating a bright image
        bright_gray_image = cv2.cvtColor(bright_image, cv2.COLOR_BGR2GRAY)
        bright_image_path = os.path.join(processor.file_manager.create_directory_for_image('bright_image'), 'bright_image.png')
        cv2.imwrite(bright_image_path, bright_image)
        _, _, classification, _ = processor.apply_adaptive_thresholding(bright_image, bright_gray_image, 'bright_image', processor.file_manager.create_directory_for_image('bright_image'))
        self.assertEqual(classification, 'bad image', "Bright image not classified as 'bad image'.")

class TestBlobExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_image_path = os.path.join(cls.script_dir, 'static', 'test', 'full_black.png')
        
    def test_blob_extraction_bad_classification(self):
        processor = ImageProcessor()
        bad_classification = 'bad image'
        extracted_blob_path = processor.extract_blob_contents(self.test_image_path, self.test_image_path, bad_classification)
        extracted_blob = cv2.imread(extracted_blob_path, cv2.IMREAD_COLOR)
        original_image = cv2.imread(self.test_image_path, cv2.IMREAD_COLOR)
        comparison = original_image == extracted_blob
        self.assertTrue(comparison.all(), "Blob extraction with bad classification did not fall back to original image.")


class TestBeamMetricsCalculator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up the path to the test image
        cls.script_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_image_path = os.path.join(cls.script_dir, 'static', 'processed','run04_Shot6', 'run04_Shot6_final_processed.png')
        
        cls.calculator = BeamMetricsCalculator(cls.test_image_path)

        # Load the test image
        cls.test_image = cv2.imread(cls.test_image_path, cv2.IMREAD_COLOR)
        if cls.test_image is None:
            raise FileNotFoundError(f"Test image not found at path: {cls.test_image_path}")

    def test_load_and_normalize_image(self):
        """Ensure the image loads and normalizes correctly."""
        normalized_image = self.calculator.load_and_normalize_image()
        self.assertIsNotNone(normalized_image, "Image loading failed.")
        self.assertEqual(normalized_image.max(), 1, "Image normalization failed.")

    def test_fit_gaussian(self):
        """Test the Gaussian fit on synthetic data."""
        # Create a synthetic image with known parameters
        data = np.zeros((100, 100), dtype=float)
        x, y = np.meshgrid(np.linspace(0, 99, 100), np.linspace(0, 99, 100))
        data += np.exp(-((x-50)**2 + (y-50)**2) / (2*10**2))

        # Use this data to perform a Gaussian fit
        fitted_params = self.calculator.fit_gaussian(data)
        self.assertIsNotNone(fitted_params, "Gaussian fitting failed.")

    def test_calculate_metrics(self):
        """Test the calculation of metrics from fitted parameters."""
        # Assuming we have a synthetic test similar to fit_gaussian
        data = np.zeros((100, 100), dtype=float)
        x, y = np.meshgrid(np.linspace(0, 99, 100), np.linspace(0, 99, 100))
        data += np.exp(-((x-50)**2 + (y-50)**2) / (2*10**2))
        self.calculator.image_path = self.test_image_path  # Mock the path if necessary
        cv2.imwrite(self.test_image_path, data)

        metrics = self.calculator.calculate_metrics()
        self.assertIn('intensity', metrics, "Metrics calculation missing 'intensity'.")
        self.assertIn('center_x', metrics, "Metrics calculation missing 'center_x'.")

if __name__ == '__main__':
    unittest.main()

