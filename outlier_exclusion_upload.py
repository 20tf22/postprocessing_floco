import numpy as np
from scipy.stats.mstats import gmean
import cv2


def new_gmean(a_temp, weights1):
    """Bulky method for fast computation of weighted geometric mean"""
    
    indices_of_nans = np.where(np.isnan(a_temp))
    indices_of_one_nan = np.where(np.sum(np.isnan(a_temp),axis = 2) == 1) #find coordinates where there is only one nan
    
    indices_of_one_nan_array = np.zeros(a_temp.shape[0:2]) # initialize array, same size as image
    indices_of_one_nan_array[indices_of_one_nan[0], indices_of_one_nan[1]] = np.where(np.isnan(a_temp[indices_of_one_nan]))[1] # write 3rd dim. index of nan for each coordinate 
    y,x = np.meshgrid(range(a_temp.shape[1]), range(a_temp.shape[0])) # create index array
    mask = np.ones(a_temp.shape, dtype = bool) # initialize true mask
    mask[x,y,indices_of_one_nan_array.astype(int)] = False # set elements to be excluded to false
    a_temp_one_nan_removed = np.reshape(a_temp[mask], (a_temp.shape[0], a_temp.shape[1], a_temp.shape[2]-1)) # remove elements specified in mask
    
    weights_one_nan_removed = np.reshape(weights1[mask], (a_temp.shape[0], a_temp.shape[1], a_temp.shape[2]-1))
    weights_one_nan_removed[np.where(np.sum(weights_one_nan_removed == 0, axis = 2) == 2)[0],np.where(np.sum(weights_one_nan_removed == 0, axis = 2) == 2)[1],:] = 1
        
    
    indices_of_one_rem_value = np.where(np.sum(~np.isnan(a_temp),axis = 2) == 1) #find coordinates of only one remaining value
    indices_of_one_rem_value_array = np.zeros(a_temp.shape[0:2])
    indices_of_one_rem_value_array[indices_of_one_rem_value[0], indices_of_one_rem_value[1]] = np.where(~np.isnan(a_temp[indices_of_one_rem_value]))[1]
    mask = np.zeros(a_temp.shape, dtype = bool) # initialize false mask
    mask[x,y,indices_of_one_rem_value_array.astype(int)] = True # set elements to be kept to true
    a_temp_two_nans_removed = np.reshape(a_temp[mask], (a_temp.shape[0], a_temp.shape[1])) # remove elements specified in mask
       
    
    weights1[np.where(np.sum(weights1 == 0, axis = 2) == 3)[0],np.where(np.sum(weights1 == 0, axis = 2) == 3)[1],:] = 1 # make sure that normal gmean can be calculated, even if it is not used
    a_tracew = np.where(np.sum(~np.isnan(a_temp), axis = 2) == 3, \
                   gmean(a_temp, axis = 2, weights = weights1), \
                   np.where(np.sum(~np.isnan(a_temp), axis = 2) == 2, gmean(a_temp_one_nan_removed, axis = 2, weights = weights_one_nan_removed), \
                   a_temp_two_nans_removed))
    return a_tracew



def outlier_exclusion(input_image, std_threshold = 0.3, iterations = 8, kernel_size = 41):                        
    """Use outlier exclusion algorithm
    
    Parameters
    ----------
    input_image : 4-dimensional np-array of shape (height x width x 
        3 (diffusion directions) x 4 (repetitions)) 
    std_threshold : adjustable parameter thr (default 0.3) 
    iterations : adjustable parameter k (default 8) 
    kernel size : adjustable parameter ks_2 (default 41) 
    
    Returns
    ----------
    edited image
    """           
    
    # calculate "normal" trace-weighted image
    S = input_image                      
    S_tracew_temp = np.mean(S, axis = 3)
    S_tracew = gmean(S_tracew_temp, axis = 2)   
                    
    # sorted indices of images with lowest signal
    min_idx = np.array([[np.argsort(S[x,y,:,:], axis = None) for y in range(S.shape[1])] for x in range(S.shape[0])])
    
    S_corrected = np.copy(S).astype(float)
    
    for k in range(iterations):                                                       
        # arithmetic mean
        S_corrected_tracew_temp = np.nanmean(S_corrected , axis = 3)       
        
        # weighted geometric mean
        weights_temp = np.sum(~np.isnan(S_corrected), axis = 3)                   
        S_corrected_tracew = new_gmean(S_corrected_tracew_temp, weights_temp)
                        
        # calculate standard deviation map                                            
        std_map = np.nanstd(S_corrected, axis = (2,3))
                        
        # replace single zeros in S_corrected_tracew. They would lead to an
        # infinite relative standard deviation.        
        S_corrected_tracew_without_holes = np.copy(S_corrected_tracew)
        S_corrected_tracew_blurred = cv2.GaussianBlur(S_corrected_tracew_without_holes, ksize = (3,3), sigmaX = 4, borderType = cv2.BORDER_DEFAULT) #0
        S_corrected_tracew_without_holes[S_corrected_tracew==0] = S_corrected_tracew_blurred[S_corrected_tracew==0]
        
        # calculate relative standard deviation map
        std_map_rel = std_map/S_corrected_tracew_without_holes                          
               
        # replace Inf with 0 to make Gaussian filtering possible
        std_map_rel[np.isinf(std_map_rel)] = 0
       
        # correct very high standard deviations to make Gaussian filtering more
        # reliable
        std_map_rel[std_map_rel > 1] = 1
        
        # set standard deviation to zero at those positions, where it was nan
        # due to 0-by-0 division
        std_map_rel[np.logical_and(std_map == 0, S_corrected_tracew_without_holes == 0)] = 0
                        
        # Gaussian filtering
        std_map_rel_gauss2 = cv2.GaussianBlur(std_map_rel, ksize = (kernel_size, kernel_size), sigmaX = 0, borderType = cv2.BORDER_DEFAULT)     
                                   
        # set Inf and NaN values to 0
        std_map_rel_gauss2[np.logical_or(np.isnan(std_map_rel_gauss2), np.isinf(std_map_rel_gauss2))] = 0
        std_map_final = np.copy(std_map_rel_gauss2)
        
        badness_map = std_map_final > std_threshold 
            
        # replace currently lowest element by nan        
        temp_a, temp_b = np.meshgrid(np.arange(S.shape[1]), np.arange(S.shape[0]))       
        S_corrected_with_all_replacements = np.copy(S_corrected)
        S_corrected_with_all_replacements[temp_b, temp_a, (np.floor(min_idx[:,:,k]/4)).astype(int), min_idx[:,:,k]%4] = np.nan
                           
        # try to exclude highest element instead of lowest element
        if k == 0:
            # compare std(without min) and std(without max) 
            S_corrected_with_max_excluded = np.copy(S_corrected)
            S_corrected_with_max_excluded[temp_b, temp_a, (np.floor(min_idx[:,:,11-k]/4)).astype(int), min_idx[:,:,11-k]%4] = np.nan
            std_map_max_excluded = np.nanstd(S_corrected_with_max_excluded, axis = (2,3))
            std_map_min_excluded = np.nanstd(S_corrected_with_all_replacements, axis = (2,3))
            max_excluding_map = cv2.GaussianBlur(std_map_max_excluded, ksize = (7,7), sigmaX = 4, borderType = cv2.BORDER_DEFAULT)<0.8*cv2.GaussianBlur(std_map_min_excluded, ksize = (7,7), sigmaX = 4, borderType = cv2.BORDER_DEFAULT)  #0
            
            # make sure that both max and min are excluded when necessary
            S_corrected_max_and_min_excluded = np.copy(S_corrected_with_all_replacements)
            S_corrected_max_and_min_excluded[temp_b, temp_a, (np.floor(min_idx[:,:,11-k]/4)).astype(int), min_idx[:,:,11-k]%4] = np.nan    
            
        # apply only to "bad" regions
        S_corrected[badness_map] = S_corrected_with_all_replacements[badness_map]
        if k == 0:
            S_corrected[max_excluding_map] = S_corrected_max_and_min_excluded[max_excluding_map]                
                  
        # arithmetic averaging        
        S_corrected_tracew_temp = np.nanmean(S_corrected , axis = 3)
        
        weights_temp = np.sum(~np.isnan(S_corrected), axis = 3)
        S_corrected_tracew = new_gmean(S_corrected_tracew_temp, weights_temp)            

    
    return S_corrected_tracew
