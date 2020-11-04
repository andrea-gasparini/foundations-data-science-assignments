import numpy as np
from numpy import histogram as hist



#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)
import gauss_module



#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    size = 255/num_bins # bins' size
    img_gray = img_gray/size # normalize the image by the bins' size.
    flattened_img_gray = img_gray.flatten() # flat the matrix
    hists = np.bincount(flattened_img_gray.astype(int), None, num_bins) / flattened_img_gray.size
    
    bins = np.linspace(0, 255, num_bins + 1)
    return hists, bins


#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'


    #... (your code here)


    #Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    
    size = 255/num_bins
    
    red_channel = np.floor(img_color_double[:,:,0].flatten()/size).astype(np.int)
    green_channel = np.floor(img_color_double[:,:,1].flatten()/size).astype(np.int)
    blue_channel = np.floor(img_color_double[:,:,2].flatten()/size).astype(np.int)
    
    # Loop for each pixel i in the image 
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        #... (your code here)
        hists[red_channel[i], blue_channel[i], green_channel[i]] += 1
    #Normalize the histogram such that its integral (sum) is equal 1
    #... (your code here)
    
    hists = hists / (128*128) # images are 128x128
    
    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists



#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'


    #... (your code here)
    
    size = 255/num_bins


    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    red_channel = np.floor(img_color_double[:,:,0].flatten()/size).astype(np.int)
    green_channel = np.floor(img_color_double[:,:,1].flatten()/size).astype(np.int)
    
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        #... (your code here)
        hists[red_channel[i], green_channel[i]] += 1
    #... (your code here)

    hists = hists / (128*128) # images are 128x128
    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    return hists




#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'
    
    dx, dy = gauss_module.gaussderiv(img_gray, 3.0)
    generatedbins = np.linspace(-6, 6, num=num_bins + 1)

    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    dxdy = np.clip(np.stack((dx, dy), axis=2), -6, 6) # cap the values in the range [-6, 6] with np.clip function
    fimg = dxdy.reshape(dxdy.shape[0]*dxdy.shape[1], 2) # reshape the arrays
    for i in range(dxdy.shape[0]*dxdy.shape[1]):

        
        dx_bin = np.where(generatedbins <= fimg[i, 0])[0][-1] if fimg[i, 0] != 6 else generatedbins.size - 2 # choose bins.size - 2 if current pixel is != 6
        dy_bin = np.where(generatedbins <= fimg[i, 1])[0][-1] if fimg[i, 1] != 6 else generatedbins.size - 2
        hists[dx_bin, dy_bin] += 1

    # Normalize the histogram such that its integral (sum) is equal 1
    hists = hists / hists.sum()

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists



def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
  if hist_name == 'grayvalue':
    return normalized_hist(img, num_bins_gray)
  elif hist_name == 'rgb':
    return rgb_hist(img, num_bins_gray)
  elif hist_name == 'rg':
    return rg_hist(img, num_bins_gray)
  elif hist_name == 'dxdy':
    return dxdy_hist(img, num_bins_gray)
  else:
    assert False, 'unknown distance: %s'%hist_name

