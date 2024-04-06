import numpy as np
from scipy.optimize import curve_fit
import cv2

class BeamMetricsCalculator:
    def __init__(self, image_path):
        self.image_path = image_path

    def gaussian(self, x, y, amplitude, xo, yo, sigma_x, sigma_y, theta):
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude * np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        return g.ravel()

    def load_and_normalize_image(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"No file found at {self.image_path}")
        normalized_image = image / np.max(image)
        return normalized_image

    def create_meshgrid(self, data):
        x = np.linspace(0, data.shape[1]-1, data.shape[1])
        y = np.linspace(0, data.shape[0]-1, data.shape[0])
        x, y = np.meshgrid(x, y)
        return x, y

    def fit_gaussian(self, data):
        x, y = self.create_meshgrid(data)
        x_data, y_data = x.ravel(), y.ravel()
        z_data = data.ravel()
        initial_guess = (1, data.shape[1]/2, data.shape[0]/2, data.shape[1]/4, data.shape[0]/4, 0)
        xy_data = np.vstack((x_data, y_data)).T
        try:
            popt, _ = curve_fit(lambda xy, amplitude, xo, yo, sigma_x, sigma_y, theta: 
                                self.gaussian(xy[:, 0], xy[:, 1], amplitude, xo, yo, sigma_x, sigma_y, theta),
                                xy_data, z_data, p0=initial_guess)
        except RuntimeError as e:
            raise RuntimeError(f"Error fitting Gaussian: {e}")
        return popt

    def calculate_metrics(self):
        processed_data = self.load_and_normalize_image()
        fitted_params = self.fit_gaussian(processed_data)
        return {
            'intensity': fitted_params[0],
            'center_x': fitted_params[1],
            'center_y': fitted_params[2],
            'width_x': fitted_params[3],
            'width_y': fitted_params[4],
            'aspect_ratio': fitted_params[3] / fitted_params[4],
            'orientation': np.degrees(fitted_params[5]) % 360
        }

# Example usage
if __name__ == "__main__":
    calculator = BeamMetricsCalculator('Z:\VSCode/Project154-3/main/static/processed/20210501_run04_Shot7/20210501_run04_Shot7_preserved_brightness.png')
    try:
        metrics = calculator.calculate_metrics()
        print(metrics)
    except Exception as e:
        print(e)


