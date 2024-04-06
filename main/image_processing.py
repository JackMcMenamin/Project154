import cv2
import numpy as np
import os
import logging
from file_handling import FileManager
from metrics_calculations import BeamMetricsCalculator


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
    
    def extract_blob_contents(self, final_processed_image_path, original_image_path, classification):
        # This method now accepts classification as a parameter
        base_name = os.path.splitext(os.path.basename(original_image_path))[0]
        extracted_blob_path = os.path.join(os.path.dirname(final_processed_image_path), f"{base_name}_extracted_blob.png")

        if classification == 'normal beam':
            # Proceed with extracting the blob from the image with red outline
            final_image = cv2.imread(final_processed_image_path)
            lower_red = np.array([0, 0, 200])
            upper_red = np.array([50, 50, 255])
            red_mask = cv2.inRange(final_image, lower_red, upper_red)
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                blob_mask = np.zeros_like(red_mask)
                cv2.fillPoly(blob_mask, [largest_contour], 255)
                original_image = cv2.imread(original_image_path)
                extracted_blob = cv2.bitwise_and(original_image, original_image, mask=blob_mask)
            else:
                # If no contours found, use the original image
                extracted_blob = cv2.imread(original_image_path)
        else:
            # For 'bad' images, use the original image
            extracted_blob = cv2.imread(original_image_path)

        # Save the extracted blob using the new path
        cv2.imwrite(extracted_blob_path, extracted_blob)
        
        return extracted_blob_path
    
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
        
        if white_area_ratio > 15:
            return 'normal beam'
        else:
            return 'bad image'

    def process_and_draw_image(self, image_path):
        self.logger.info(f"Starting to process image: {image_path}")

        # Load the original image
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original_image is None:
            self.logger.error(f"Error loading image from {image_path}")
            return None, None  # Return None for both result and metrics when there's an error

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        intermediate_dir = self.file_manager.create_directory_for_image(base_name)

        # Process steps
        original_image_path = self.save_intermediate_image(original_image, intermediate_dir, base_name, 'original')
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        gray_image_path = self.save_intermediate_image(gray_image, intermediate_dir, base_name, 'gray')

        # Thresholding
        thresholded_image, final_thresholded_image_path, classification, preserved_brightness_image_path = self.apply_adaptive_thresholding(original_image, gray_image, base_name, intermediate_dir)

        final_image_path = None
        beam_metrics = None  # Initialize beam metrics here
        
        contour_overlay_image, contour_image_path, classification = self.normal_image_processing(thresholded_image, original_image, base_name, intermediate_dir)

        if classification == 'normal beam':
            if contour_overlay_image is not None:
                final_image_path = self.combine_contour_with_original(original_image, contour_overlay_image, base_name, intermediate_dir)
                contour_image_path = final_thresholded_image_path  # For 'normal beam', contour image is the one with drawn contours

                # Initialize BeamMetricsCalculator with the path to the preserved brightness image
                beam_metrics_calculator = BeamMetricsCalculator(preserved_brightness_image_path)

                # Calculate the metrics for the image
                beam_metrics = beam_metrics_calculator.calculate_metrics()

                # Log or print the metrics for debugging
                self.logger.info(f"Beam metrics for {image_path}: {beam_metrics}")
        else:
            # For 'bad' images:
            contour_image_path = self.save_intermediate_image(thresholded_image, intermediate_dir, base_name, 'contour')
            final_image_path = os.path.join(intermediate_dir, f"{base_name}_final_processed.png")
            cv2.imwrite(final_image_path, original_image)
            beam_metrics = None

        result = {
            'original_image_path': original_image_path,
            'gray_image_path': gray_image_path,
            'final_thresholded_image_path': final_thresholded_image_path,
            'contour_image_path': contour_image_path,
            'final_image_path': final_image_path,
            'blob_extraction_path': preserved_brightness_image_path,
            'metrics': beam_metrics,
            'image_classification': classification
        }

        self.logger.info(f"Image processing completed for: {image_path}")
        
        # Return result and beam metrics separately
        return result



    def save_intermediate_image(self, image, dir_path, base_name, suffix):
        image_path = os.path.join(dir_path, f"{base_name}_{suffix}.png")
        cv2.imwrite(image_path, image)
        return image_path

    def apply_adaptive_thresholding(self, original_image, gray_image, base_name, intermediate_dir):
        threshold_value = 100  # Start with a baseline threshold value
        max_attempts = 110  # Allow a certain number of attempts to adjust brightness
        enhancement_factor = 1  # How much to brighten the image if it's too dark
        darkening_factor = -1  # How much to darken the image if it's too bright
        attempt = 0  # Keep track of the number of attempts
        white_area_ratio = None  # Track the white area ratio for diagnostics
        beam_classification = None
        original_color_image = original_image.copy()  # Make a copy of the original color image
        
        while attempt < max_attempts:
            _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
            white_pixels = cv2.countNonZero(thresholded_image)
            total_pixels = gray_image.size
            white_area_ratio = (white_pixels / total_pixels) * 100

            # Use the old code's logic for adjusting brightness and classification
            if white_area_ratio < 15:
                beam_classification = 'image too dark'
                gray_image = cv2.add(gray_image, enhancement_factor)
            elif white_area_ratio > 40:
                beam_classification = 'image too bright'
                gray_image = cv2.add(gray_image, darkening_factor)
            else:
                beam_classification = 'normal beam'
                break  # Exit the loop if image is classified as normal

            attempt += 1

        if beam_classification != 'normal beam':
            # If image is too dark or too bright even after max attempts, classify as bad image
            beam_classification = 'bad image'

        # Convert the thresholded binary image back to a 3-channel image
        color_thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)

        # Create a mask for the original bright areas
        mask = color_thresholded_image.astype(bool)

        # Apply the mask to the original image to preserve the brightness of non-black areas
        preserved_brightness_image = np.where(mask, original_color_image, 0)

        # Save the thresholded image with preserved brightness
        preserved_brightness_image_path = self.save_intermediate_image(preserved_brightness_image, intermediate_dir, base_name, 'preserved_brightness')
        
        final_thresholded_image_path = self.save_intermediate_image(thresholded_image, intermediate_dir, base_name, 'final_threshold')
        return thresholded_image, final_thresholded_image_path, beam_classification, preserved_brightness_image_path


    def normal_image_processing(self, thresholded_image, original_image, base_name, intermediate_dir):
        # Make a color version of the thresholded image to draw colored contours on
        # This converts the binary image back to a BGR image so that contours can be colored
        contour_overlay_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)

        # Use morphological opening to clean up the thresholded image
        kernel = np.ones((5,5), np.uint8)
        opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

        # Find contours on the opened (cleaned up) thresholded image
        contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Call analyze_contours to determine if the image is 'normal' or 'bad'
        classification = self.analyse_contours(contours, original_image.shape)
        
        if classification == 'normal beam':
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
            
            # Save the contour overlay image which is the thresholded image with contours
            contour_image_path = self.save_intermediate_image(contour_overlay_image, intermediate_dir, base_name, 'contour')

            return contour_overlay_image, contour_image_path, 'normal beam'
        
        else:
            # If analyze_contours classified the image as 'bad', handle it accordingly
            self.logger.info(f"Contour analysis classified {base_name} as 'bad image'.")
            return None, None, 'bad image'
    
    def analyse_contours(self, contours, image_shape):
        if not contours:
            return 'bad image'
        
        min_contour_area = 100

        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < min_contour_area:  # self.min_contour_area to be defined based on your images
            return 'bad image'
        
        return 'normal beam'

    
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
    
    def apply_color_preserved_thresholding(self, original_image, gray_image, base_name, intermediate_dir):
        threshold_value = 100  # Define the threshold value
        _, binary_thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        # This mask will have the shape of the original image but will contain the thresholded results
        color_preserved_mask = cv2.cvtColor(binary_thresholded_image, cv2.COLOR_GRAY2BGR)

        # Where the mask is white, we use the original image; where it is black, we assign black pixels.
        color_preserved_thresholded_image = np.where(color_preserved_mask == np.array([255, 255, 255]), original_image, 0)

        # Save the color-preserved thresholded image
        color_preserved_thresholded_image_path = self.save_intermediate_image(color_preserved_thresholded_image, intermediate_dir, base_name, 'color_preserved_threshold')
        
        return color_preserved_thresholded_image, color_preserved_thresholded_image_path


    def save_final_processed_image(self, image, dir_path, base_name):
    # Assuming 'image' is the image with contours drawn on it
        return self.save_intermediate_image(image, dir_path, base_name, 'final_processed')
    
    def check_for_red_lines(self, contour_overlay_image):
        # Count the number of red pixels, which are pixels where the red channel is 255 and the other two are less than a threshold.
        red_pixels = np.sum((contour_overlay_image[:, :, 2] == 255) &
                            (contour_overlay_image[:, :, 1] < 200) &
                            (contour_overlay_image[:, :, 0] < 200))
        total_pixels = contour_overlay_image.shape[0] * contour_overlay_image.shape[1]
        red_pixel_ratio = (red_pixels / total_pixels) * 100

        # You may need to adjust the percentage based on your specific case.
        if red_pixel_ratio < 1:  # If less than 1% of the image contains red lines, it might be a bad image.
            return False
        return True




