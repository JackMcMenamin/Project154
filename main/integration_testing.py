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
        
        final_image_path = self.final_image_path
        
        # Pass the path of the processed image to the metrics calculator
        metrics_calculator = BeamMetricsCalculator(final_image_path)
        metrics = metrics_calculator.calculate_metrics()
        
        # Check if metrics are calculated
        self.assertIsNotNone(metrics)
        self.assertIn('intensity', metrics)
        self.assertIn('center_x', metrics)
        self.assertIn('center_y', metrics)

        self.assertGreater(metrics['intensity'], 0, "Intensity should be greater than 0")
        self.assertGreater(metrics['center_x'], 0, "Center X should be within the image bounds")
        self.assertGreater(metrics['center_y'], 0, "Center Y should be within the image bounds")
        
class TestUIIntegration(unittest.TestCase):
    def setUp(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_image_dir = os.path.join(self.script_dir, 'static', 'processed', 'run04_Shot8')
        self.processor = ImageProcessor()
        self.metrics_calculator = BeamMetricsCalculator(None) 

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

                self.assertTrue(os.path.exists(final_image_path), "Processed image file does not exist")
                self.assertGreater(metrics['intensity'], 0, "Intensity should be greater than 0")



if __name__ == '__main__':
    unittest.main()
