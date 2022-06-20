import numpy as np
from itertools import chain , combinations
from scipy.stats.mstats import gmean
from collections import Counter
import warnings

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




def inverse_powerset ( iterable , maxout =6) :
    "Takes the list S and creates a list with possible subsets S-S_j as \
        entries. Subsets are of type tuple. Maxout denotes the exact number \
        of outliers."
    
    return list(combinations(iterable, 12-maxout))

def get_winningsets_and_sfs (arr , maxout =6) :
    " Takes array of shape (12, height, width) and returns array of shape \
        (height, width), where each entry contains the set S-S_j with the \
        highest SF of the respective pixel. \
        Also returns the highest SF of each pixel in an array of the same \
        shape. Maxout denotes the exact number of outliers."
    
    # calculate sets S-S_j and respective Smoothing Factors
    setsinv = np.apply_along_axis(inverse_powerset, 0, arr, (maxout)) 
    variances_lists = np.var(arr, axis =0) 
    variances_sets = np. var(setsinv, axis =1) 
    smoothing_factors = (arr.shape [0] - maxout )*(variances_lists - variances_sets) 
    
    # find best smoothing factors and respective sets S-S_j
    winning_sfs = np.amax(smoothing_factors, axis =0)
    indices = np.expand_dims(np.expand_dims(np.argmax(smoothing_factors , axis=0) , axis =0) , axis =0)
    winningsets = np.take_along_axis(setsinv, indices, axis =0) 
    return(winningsets[0], winning_sfs)

def remove_set_from_array_all (arr , maxout =6) :
    "Takes array of shape (12, height, width) and returns array of same shape, where \
        the outliers have been replaced by NaNs.  Maxout denotes the exact number \
        of outliers."
    
    # copy array to allow replacement with NaNs
    arr2 = arr.copy().astype ("float")
    
    # calculate sets S-S_j with the highest Smoothing Factors
    winningsets, winsfs = get_winningsets_and_sfs(arr2, maxout) 
   
    # replace outliers by NaNs 
    for i in range(arr2.shape[1]) : 
        for j in range(arr2.shape[2]) :
            c1 = Counter(arr2[:,i,j]) 
            c2 = Counter(winningsets[:,i,j])
            diff = c1-c2 
            to_remove = list(diff.elements())           
            for elem in to_remove:
                indices = np.where(arr2[:,i,j] == elem)
                arr2[indices [0][0] ,i,j]= np. nan
    return (arr2, winsfs)
            
def plain_to_trace(arr):
    "Takes array of shape (12, height, width) and averages arithmetically over \
        the four repetitions and geometrically over the 3 directions. Returns \
        array of shape (height, width)."

    # reshape
    arr = np.reshape(arr, [4,3, arr.shape[1] , arr.shape[2]])
    arr = np.moveaxis(arr, [0,1,2,3], [3, 2, 0, 1])
    
    # arithmetic mean
    with warnings.catch_warnings():
          warnings.filterwarnings("ignore", category=RuntimeWarning)  
          means = np.nanmean(arr, axis = 3)
          
    # geometric mean
    weights = np.sum(~np.isnan(arr), axis = 3)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        temp = new_gmean(means, weights)
    return(temp)

def algorithmus (arr , maxout =6) :
    "Takes array of shape (12, height, width) and applies the algorithm for a \
        fixed number of outliers. Maxout denotes the exact number of outliers. \
        Returns array of shape (height, width) which contains the values of the \
        new trace-weighted image."
    
    winarr, winsfs = remove_set_from_array_all(arr, maxout)
    return(plain_to_trace(winarr), winsfs)

def exception_set (input_image , maxout = 1) :
    """Use exception set algorithm
    
    Parameters
    ----------
    input_image : 4-dimensional np-array of shape (height x width x 
        3 (diffusion directions) x 4 (repetitions)) 
    maxout : adjustable parameter maxout (default 1) 
        
    Returns
    ----------
    edited image
    """    
    
    #reshape array
    image = np.moveaxis(input_image, [0, 1, 2, 3], [2,3,0,1])
    arr = np.reshape(image, [12, image.shape[2], image.shape[3]], order = 'F')       
        
    list_of_sfs =[]
    list_afteralg = []
    
    # calculate Smoothing Factors and trace-weighted images for up to maxout 
    # outliers
    for i in range (maxout +1) :
        winarr, winsfs = algorithmus(arr ,i)
        list_of_sfs.append(winsfs)
        list_afteralg.append(winarr)
    arr_of_sfs = np.array(list_of_sfs)
    arr_of_afteralg = np.array(list_afteralg)
    
    # finds the index of highest SF among the different number of outliers
    indices = np.expand_dims(np.argmax(arr_of_sfs, axis = 0), axis =0) 
    
    # final image after exclusion of the best-suited outlier set
    finalres = np.take_along_axis(arr_of_afteralg, indices, axis =0) 
    return(finalres[0])