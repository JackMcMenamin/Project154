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
        
        # Perform contouring on the thresholded image, passing in the original image for overlay
        contour_overlay_image, contour_image_path = self.find_and_draw_contours(thresholded_image, original_image, base_name, intermediate_dir)

        # Final image is the one with contours drawn
        final_image_path = self.save_final_processed_image(contour_overlay_image, intermediate_dir, base_name)
        
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

    def find_and_draw_contours(self, thresholded_image, original_image, base_name, intermediate_dir):
        # Use morphological opening as in the old code to clean up the image
        kernel = np.ones((5,5), np.uint8)
        opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

        # Find contours on the opened image
        contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contours were found
        if not contours:
            self.logger.error(f"No contours found in image: {opened_image}")
            return original_image, None

        # Combine all contours to create a single comprehensive outline using convex hull
        # Convert contours to a list first, if it's not already one
        all_contours = np.vstack(list(contours[i] for i in range(len(contours))))
        hull = cv2.convexHull(all_contours)

        # Scale the hull by 1.1 times (make it 10% larger)
        scaled_hull = ImageProcessor.scale_contour(hull, 1.1)

        # Draw the comprehensive outline onto the original image
        cv2.drawContours(original_image, [scaled_hull], -1, (0, 0, 255), 2)

        # Optionally, draw red 'X' at the centroid of the hull if needed
        M = cv2.moments(hull)
        if M["m00"] != 0:
            cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            size = 10
            cv2.line(original_image, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 2)
            cv2.line(original_image, (cx + size, cy - size), (cx - size, cy + size), (0, 0, 255), 2)

        # Save the contour image
        contour_image_path = self.save_intermediate_image(original_image, intermediate_dir, base_name, 'contour')

        return original_image, contour_image_path


    def save_final_processed_image(self, image, dir_path, base_name):
    # Assuming 'image' is the image with contours drawn on it
        return self.save_intermediate_image(image, dir_path, base_name, 'final_processed')



