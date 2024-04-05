from scipy.optimize import curve_fit
import numpy as np

def gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

# Assuming 'data' is a 2D array from the thresholded beam profile, with 0 for background and 1 for beam pixels.
x = np.linspace(0, data.shape[1], data.shape[1])
y = np.linspace(0, data.shape[0], data.shape[0])
x, y = np.meshgrid(x, y)

# Initial guesses for Gaussian parameters
initial_guess = (3, data.shape[1]/2, data.shape[0]/2, 20, 20, 0)

# Perform the fitting
params, pcov = curve_fit(gaussian, (x, y), data.ravel(), p0=initial_guess)

# Extract the parameters
amplitude, xo, yo, sigma_x, sigma_y, theta = params

# Now you can use amplitude, xo, yo, sigma_x, sigma_y, theta to calculate fit metrics.

