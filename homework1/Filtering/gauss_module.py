# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
import scipy.ndimage as ndimage



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
    
    #...
    Gx = [helpFunction(sigma, i) for i in range(-3 * int(sigma), 3 * int(sigma) + 1)]
    x = [i for i in range(-3*int(sigma), 3*int(sigma)+1)]
    return Gx, x



def helpFunction(sigma, i):
    return 1 / (math.sqrt(2*math.pi) * sigma) * math.exp(-float(i)**2/(2*sigma**2))


"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
    
    #...
    size = int(sigma**2)
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = helpFunction(sigma, kernel_1D[i])
    
    kernel_2D = np.outer(kernel_1D[::-1], kernel_1D[::-1])
    #img = ndimage.gaussian_filter(img, sigma=(5, 5, 0), order=0)    
    return conv2(img, kernel_2D)



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

    #...
    
    #return Dx, x
    pass



def gaussderiv(img, sigma):

    #...
    
     #return imgDx, imgDy
    pass

