# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
    
    r = [i for i in range(-3*int(sigma), 3*int(sigma)+1)]
    return np.array([1 / (sigma * math.sqrt(2*math.pi)) * math.exp(-float(x)**2/(2*sigma**2)) for x in r]), r



"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
    
    kernel_1D = gauss(sigma)[0] 
    """
    kernel = np.outer(kernel_1D, kernel_1D) # mul kernel to have 2D matrix
    
    return conv2(img, kernel, mode='valid') # return blurred image.
    """
    
    newImage = np.zeros((len(img), len(img[0])))
    
    for row in range(len(img)):
        newImage[row,:] = np.convolve(img[row], kernel_1D, mode='same')
    
    for col in range(len(img[0])):
        newImage[:,col] = np.convolve(newImage[:,col], kernel_1D, mode='same')
    
    return newImage



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

    r = [i for i in range(-3*int(sigma), 3*int(sigma)+1)]
    return np.array([1 / (sigma**3 * math.sqrt(2*math.pi)) * x * math.exp(-float(x)**2/(2*sigma**2)) for x in r]), r

def gaussderiv(img, sigma):

    #return imgDx, imgDy
    pass