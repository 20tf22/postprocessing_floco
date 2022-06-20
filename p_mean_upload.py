import numpy as np
from scipy.stats.mstats import gmean

def p_mean(input_image, power = 4.5):    
    """Use p-mean algorithm
    
    Parameters
    ----------
    input_image : 4-dimensional np-array of shape (height x width x 
        diffusion directions x repetitions) 
    power : adjustable parameter p (default 4.5) 
    
    Returns
    ----------
    edited image
    """             
    
    # images ** power             
    image_1 = np.power(input_image, power)
    
    # arithmetic mean over repetitions
    image_2 = np.mean(image_1, axis = 3)
    
    # image_2 ** (1/power)
    image_3 = image_2 ** (1/power)
    
    # geometric mean over diffusion directions
    return gmean(image_3, axis = 2)
