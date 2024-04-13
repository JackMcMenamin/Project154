import unittest
import os
from image_processing import ImageProcessor
from metrics_calculations import BeamMetricsCalculator

class TestImageProcessingIntegration(unittest.TestCase):
    def setUp(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_image_path = os.path.join(self.script_dir, 'static', 'test', 'run04_Shot7.png')
        self.processor = ImageProcessor()
 
    def test_image_upload_and_processing_integration(self):
        # Simulate uploading an image
        test_image_path = self.test_image_path
        
        # Pass the uploaded image path to the processor
        processing_result = self.processor.process_and_draw_image(test_image_path)
        
        # Assert that the result is not None
        self.assertIsNotNone(processing_result)
        
        # Further assertions can be made to check if processing_result is as expected
        

class TestImageUploadAndMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_image_path = os.path.join(cls.script_dir, 'static', 'test', 'run04_Shot7.png')
        cls.final_image_path = os.path.join(cls.script_dir, 'static', 'test', 'final.png')
        cls.processor = ImageProcessor()
    
    def test_image_upload_and_metrics_calculation(self):
        # Simulate uploading an image and processing it
        uploaded_image_path = self.test_image_path  # This would be the path after uploading
        processing_result = self.processor.process_and_draw_image(uploaded_image_path)
        
        # Check if processing_result is not None
        self.assertIsNotNone(processing_result)
        
        # Assuming processing_result contains the path to the final processed image
        final_image_path = self.final_image_path
        
        # Pass the path of the processed image to the metrics calculator
        metrics_calculator = BeamMetricsCalculator(final_image_path)
        metrics = metrics_calculator.calculate_metrics()
        
        # Check if metrics are calculated
        self.assertIsNotNone(metrics)
        self.assertIn('intensity', metrics)
        self.assertIn('center_x', metrics)
        self.assertIn('center_y', metrics)
        # Add other assertions as per your requirements

        # Optionally, check if the calculated metrics are within expected ranges
        self.assertGreater(metrics['intensity'], 0, "Intensity should be greater than 0")
        self.assertGreater(metrics['center_x'], 0, "Center X should be within the image bounds")
        self.assertGreater(metrics['center_y'], 0, "Center Y should be within the image bounds")
        
class TestUIIntegration(unittest.TestCase):
    def setUp(self):
        # Assuming you have a way to simulate the UI interaction
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_image_dir = os.path.join(self.script_dir, 'static', 'processed', 'run04_Shot8')
        self.processor = ImageProcessor()
        self.metrics_calculator = BeamMetricsCalculator(None)  # Path will be set later

    def test_ui_display_of_processed_images_and_metrics(self):
        # Simulate the action of a user uploading an image through the UI
        for filename in os.listdir(self.test_image_dir):
            if filename.endswith(".png"):
                test_image_path = os.path.join(self.test_image_dir, filename)

                # Simulate processing the image
                processing_result = self.processor.process_and_draw_image(test_image_path)
                self.assertIsNotNone(processing_result, "Processing of image failed")

                # Simulate calculating metrics
                final_image_path = processing_result['final_image_path']
                self.metrics_calculator.image_path = final_image_path
                metrics = self.metrics_calculator.calculate_metrics()
                self.assertIsNotNone(metrics, "Metrics calculation failed")

                # Check if the UI can display the processed image and metrics
                # Here you would normally have UI logic, which might involve rendering the image
                # and displaying metrics on the screen. Since this is an integration test without an actual UI,
                # we are assuming that the "display" action is successful if the image exists and metrics are calculated.
                self.assertTrue(os.path.exists(final_image_path), "Processed image file does not exist")
                self.assertGreater(metrics['intensity'], 0, "Intensity should be greater than 0")
                # Add other UI-related assertions as required


if __name__ == '__main__':
    unittest.main()
