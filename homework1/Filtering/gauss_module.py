# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2

sqrt = math.sqrt
pi = math.pi
e = math.exp

"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
    
    x = [i for i in range(-3*int(sigma), 3*int(sigma)+1)] # [-3s, ...., -1, 0, 1, ...., 3s]
    return np.array([mathGauss(i, sigma) for i in range(-3*int(sigma), 3*int(sigma)+1)]), x


def mathGauss(i, sigma):
    # ritorna il valore gaussiano per la posizione i
    return 1 / (sigma * sqrt(2 * pi)) * e(-float(i) ** 2 / (2*sigma**2))


"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""

def gaussianfilter_2d(img, sigma):
    kernel_1D = gauss(sigma)[0]
    kernel = np.outer(kernel_1D, kernel_1D) # crea il kernel 2D per eseguire la convolution 
    return conv2(img, kernel, mode='valid') # convolution img & kernel 2D.

def gaussianfilter(img, sigma):    
    kernel_1D = gauss(sigma)[0] # kernel 1D da applicare alle righe e poi alle colonne    
    newImage = np.zeros((len(img), len(img[0]))) # immagine da restituire         
    newImage = convolutionImage(newImage, img, kernel_1D) # convolution sulle righe
    newImage = convolutionImage(newImage, img, kernel_1D, row=False) # convolution sulle colonne    
    return newImage

def convolutionImage(new, old, kernel, row=True):
    if row:
        for r in range(len(old)):
            new[r,:] = np.convolve(old[r], kernel, mode='same')
    else:
        for c in range(len(old[0])):
            new[:,c] = np.convolve(new[:,c], kernel, mode='same')
    return new
"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):
    x = [i for i in range(-3*int(sigma), 3*int(sigma)+1)]
    return np.array([mathGaussDx(i, sigma) for i in range(-3*int(sigma), 3*int(sigma)+1)]), x

def mathGaussDx(x, sigma):
    return 1 / (sigma**3 * sqrt(2*pi)) * x * e(-float(x)**2/(2*sigma**2))

def gaussderiv(img, sigma):
    kernel_1D = gaussdx(sigma)[0]
    
    imgDx = np.zeros((len(img), len(img[0])))
    imgDy = np.zeros((len(img), len(img[0])))
    imgDx = convolutionImage(imgDx, img, kernel_1D) # convolution sulle x
    imgDy = convolutionImage(imgDy, img, kernel_1D, row=False) # convolution sulle y
    return imgDx, imgDy
