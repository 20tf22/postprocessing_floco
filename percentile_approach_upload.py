import numpy as np

def percentile_approach(input_image, percentile = 75):    
    """Use percentile algorithm
    
    Parameters
    ----------
    input_image : 4-dimensional np-array of shape (height x width x 
        diffusion directions x repetitions) 
    percentile : adjustable parameter q (default 75) 
    
    Returns
    ----------
    edited image
    """            
    
    return np.percentile(input_image, percentile, axis = (2,3))
                 