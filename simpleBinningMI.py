import numpy as np
from scipy.sparse import csr_matrix

# Mutual Information on 2 Continuous Random Variables
# Simple Binning:
#
#   identify range as min(X, Y) to max(X, Y)
#   put instances of X into range (indicator matrix)
#   put instances of Y into range (indicator matrix)
#   Shannon's entropy of X: H(X)
#   Shannon's entropy of Y: H(Y)
#   Shannon's entropy of joint probability of X, Y: H(X,Y)
#   MI = H(X) + H(Y) - H(X,Y)



def convert2IndicatorMatrix(variable, maxVal):
    '''Convert 1D np array variable (k instances) into k-by-bins indicator matrix'''
    bins = np.arange(maxVal+1) # increase max a bit so max won't be in a separate bin
    pointBinID = np.digitize(variable,bins) - 1 # convert to 0 based binID
    return csr_matrix((pointBinID[:,np.newaxis] == bins).astype(int))
    
def entropy(indicator_matrix, dim=1):
    '''Use indicator matrix of instance-by-bin to calculate MI.'''
    if dim==1:
        probability = indicator_matrix.sum(axis = 0)/indicator_matrix.sum()
        logP = np.ma.log2(probability).filled(0)
    else:
        probability = indicator_matrix.data/indicator_matrix.sum()
        logP = np.ma.log2(probability).filled(0)
    return np.sum(np.multiply(-probability, logP))

def MI(X,Y):
    '''Calculate mutual information of variables X and Y.'''
    X = np.round(X)
    Y = np.round(Y)
    minVal = min(np.min(X),np.min(Y))
    X = X-minVal
    Y = Y-minVal
    maxVal = max(np.max(X),np.max(Y))
    indicator_X = convert2IndicatorMatrix(X, maxVal)
    indicator_Y = convert2IndicatorMatrix(Y, maxVal)
    indicator_XY = np.dot(indicator_X.T,indicator_Y)
    
    H_X = entropy(indicator_X)
    H_Y = entropy(indicator_Y)
    H_XY = entropy(indicator_XY, dim=2)
    mutualInfo = H_X + H_Y - H_XY
    return max(mutualInfo, 0)