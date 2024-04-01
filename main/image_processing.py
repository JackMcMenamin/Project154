import cv2
import numpy as np
import os
import logging
from file_handling import FileManager

class ImageProcessor:
    def __init__(self):
        self.file_manager = FileManager()
        self.logger = logging.getLogger('ImageProcessor')
        self.logger.setLevel(logging.INFO)

    def process_images(self, files):
        self.logger.info(f"Starting processing of {len(files)} images.")
        results = []
        for file in files:
            original_image_path = self.file_manager.save_uploaded_file(file)
            self.logger.info(f"Saved uploaded file to {original_image_path}.")
            processing_result = self.process_and_draw_image(original_image_path)
            if processing_result:
                results.append(processing_result)
                self.logger.info(f"Image processed successfully: {original_image_path}")
        self.logger.info("All images processed.")
        return results
    
    @staticmethod
    def scale_contour(cnt, scale):
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            return cnt
        cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        cnt_scaled = cnt_scaled.astype(np.int32)
        return cnt_scaled
    
    def beam_classification(self, thresholded_image):
        """
        Classify the image based on the percentage of white pixels.
        
        Args:
        - thresholded_image: The image after applying thresholding.

        Returns:
        - A string indicating the classification of the image.
        """
        white_pixels = cv2.countNonZero(thresholded_image)
        total_pixels = thresholded_image.size
        white_area_ratio = (white_pixels / total_pixels) * 100
        
        if white_area_ratio > 10:
            return 'normal beam'
        else:
            return 'bad image'

    def process_and_draw_image(self, image_path):
        self.logger.info(f"Starting to process image: {image_path}")

        # Load the original image
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original_image is None:
            self.logger.error(f"Error loading image from {image_path}")
            return None

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        intermediate_dir = self.file_manager.create_directory_for_image(base_name)

        # Process steps
        original_image_path = self.save_intermediate_image(original_image, intermediate_dir, base_name, 'original')
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        gray_image_path = self.save_intermediate_image(gray_image, intermediate_dir, base_name, 'gray')

        # Thresholding
        thresholded_image, final_thresholded_image_path = self.apply_adaptive_thresholding(gray_image, base_name, intermediate_dir)
        
        # Classify the image
        classification = self.beam_classification(thresholded_image)
        
        # Save the final processed image
        if classification == 'normal beam':
            contour_overlay_image, contour_image_path = self.normal_image_processing(thresholded_image, original_image, base_name, intermediate_dir)
            final_image_path = self.combine_contour_with_original(original_image, contour_overlay_image, base_name, intermediate_dir)
        else:
            contour_image_path = thresholded_image
            final_image_path = original_image_path

        self.logger.info(f"Image processing completed for: {image_path}")

        result = {
            'original_image_path': original_image_path,
            'gray_image_path': gray_image_path,
            'final_thresholded_image_path': final_thresholded_image_path,
            'contour_image_path': contour_image_path,
            'final_image_path': final_image_path,
        }
        logging.info(f"Processing result for {image_path}: {result}")
        return result

    def save_intermediate_image(self, image, dir_path, base_name, suffix):
        image_path = os.path.join(dir_path, f"{base_name}_{suffix}.png")
        cv2.imwrite(image_path, image)
        return image_path

    def apply_adaptive_thresholding(self, gray_image, base_name, intermediate_dir):
        # Bring over the thresholding and brightness adjustment logic from your old code
        threshold_value = 100  # Start with a baseline threshold value
        max_attempts = 110  # Allow a certain number of attempts to adjust brightness
        enhancement_factor = 1  # How much to brighten the image if it's too dark
        darkening_factor = 1  # How much to darken the image if it's too bright
        attempt = 0  # Keep track of the number of attempts
        white_area_ratio = None  # Track the white area ratio for diagnostics
        
        while attempt < max_attempts:
            _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
            # Save each thresholding attempt if necessary for debugging
            # self.save_intermediate_image(thresholded_image, intermediate_dir, base_name, f'thresholded_{attempt}')
            
            white_pixels = cv2.countNonZero(thresholded_image)
            total_pixels = gray_image.size
            white_area_ratio = (white_pixels / total_pixels) * 100
            
            if white_area_ratio < 15:
                gray_image = np.clip(gray_image + enhancement_factor, 0, 255).astype(np.uint8)
            elif white_area_ratio > 40:
                gray_image = np.clip(gray_image - darkening_factor, 0, 255).astype(np.uint8)
            else:
                # If the white area ratio is within acceptable bounds, break from the loop
                break
            
            attempt += 1
        
        final_thresholded_image_path = self.save_intermediate_image(thresholded_image, intermediate_dir, base_name, 'final_threshold')
        return thresholded_image, final_thresholded_image_path

    def normal_image_processing(self, thresholded_image, original_image, base_name, intermediate_dir):
        # Make a color version of the thresholded image to draw colored contours on
        # This converts the binary image back to a BGR image so that contours can be colored
        contour_overlay_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)

        # Use morphological opening to clean up the thresholded image
        kernel = np.ones((5,5), np.uint8)
        opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

        # Find contours on the opened (cleaned up) thresholded image
        contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Proceed if contours were found
            all_contours = np.vstack([contour for contour in contours])
            hull = cv2.convexHull(all_contours)
            scaled_hull = ImageProcessor.scale_contour(hull, 1.1)
            # Draw the scaled hull as a comprehensive outline onto the contour_overlay_image
            cv2.drawContours(contour_overlay_image, [scaled_hull], -1, (0, 0, 255), 2)

            # Optionally, mark the centroid of the hull
            M = cv2.moments(hull)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                size = 10
                cv2.line(contour_overlay_image, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 2)
                cv2.line(contour_overlay_image, (cx + size, cy - size), (cx - size, cy + size), (0, 0, 255), 2)
            else:
                self.logger.error("No contours found in image. Using thresholded image as fallback.")
        
        # Save the contour overlay image which is the thresholded image with contours
        contour_image_path = self.save_intermediate_image(contour_overlay_image, intermediate_dir, base_name, 'contour')

        return contour_overlay_image, contour_image_path
    
    def combine_contour_with_original(self, original_image, contour_overlay_image, base_name, intermediate_dir):
        # Create a mask where the contours are drawn
        contour_mask = cv2.inRange(contour_overlay_image, (0, 0, 255), (0, 0, 255))

        # Create an all-black image with the same dimensions as the original image
        black_background = np.zeros_like(original_image)

        # Draw the red contours onto the black background using the mask
        black_background[contour_mask != 0] = (0, 0, 255)

        # Now add this image with red contours onto the original image
        combined_image = cv2.addWeighted(original_image, 1, black_background, 1, 0)
        
        # Save the combined image
        combined_image_path = self.save_intermediate_image(combined_image, intermediate_dir, base_name, 'final_processed')
        
        return combined_image_path



    def save_final_processed_image(self, image, dir_path, base_name):
    # Assuming 'image' is the image with contours drawn on it
        return self.save_intermediate_image(image, dir_path, base_name, 'final_processed')



