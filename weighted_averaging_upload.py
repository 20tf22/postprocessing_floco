import numpy as np
from scipy.stats.mstats import gmean
import cv2


def weighted_averaging(input_image, kernel_size = 23, power_weights = 4):    
    """Use weighted averaging algorithm
    
    Parameters
    ----------
    input_image : 4-dimensional np-array of shape (height x width x 
        3 (diffusion directions) x 4 (repetitions)) 
    kernel_size : adjustable parameter ks_1 (default 23) 
    power_weights : adjustable parameter n (default 4) 
    
    Returns
    ----------
    edited image
    """     

    # apply Gaussian Filter to all 12 images                      
    blurred_images = np.array([[cv2.GaussianBlur(input_image[:,:,dirs,avs], 
         ksize = (kernel_size, kernel_size), sigmaX = 0, borderType = 
         cv2.BORDER_DEFAULT) for dirs in range(3)] for avs in range(4)])
    
    # re-arrange shape to (height x width x diffusion directions x repetitions)
    weight_maps = np.moveaxis(blurred_images,[0, 1], [-1, -2])
    
    # set weights which are zero to small positive values to allow weighted 
    # averaging
    weight_maps[weight_maps == 0] = 0.00000001
    
    # weighted averaging with weights = weight_maps ** power   
    images_xyz = np.array([np.average(input_image[:,:,xyz,:], axis = 2, 
         weights = weight_maps[:,:,xyz,:]**power_weights) 
         for xyz in range(3)])
    
    # geometric averaging    
    return gmean(images_xyz, axis = 0)
