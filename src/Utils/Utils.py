''' This module holds generic utility functions. '''
import numpy as np
import time



''' generic function to make a binary string from array '''
def BinaryStr(subset)
		'''
		Make a nice binary string of the form '1010' from an input
		array_like of binary values.
		:param subset: array_like of binary values
		:return strbinary: string of the binary values
		'''
		return ''.join([str(int(flg)) for flg in subset])


''' generic function to perform weighted random selection '''
def RandomWeightedSelect(keys, wats, randSeed=None):
    '''
    Randomly select an item from a list, according to a set of
    specified weights.
    :param keys: array-like of items from which to select
    :param wats: array-like of weights associated with the input
        keys; must be sorted in descending weight
    :param randSeed: optional random seed for np.random; if no
        randomization is desired, pass 0
    :return selection: selected item
    :return randSeed: random seed used
    '''
    
    # ranodmize, perhaps
    if randSeed != 0:
        if randSeed is None:
            randSeed = int(str(time.time()).split('.')[1])
        np.random.seed(randSeed)
    
    # ensure weights sum to 1
    totWats = sum(wats)
    if totWats != 1:
        wats = [v/totWats for v in wats]
    
    # get the cumulative weights
    cumWats = np.cumsum(wats)
    # get the indices of where the random [0,1] is < the cum weight
    rnd = np.random.rand()
    seld = rnd < cumWats
    
    return [k for (k,s) in zip(keys, seld) if s][0], randSeed