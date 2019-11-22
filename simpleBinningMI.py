import numpy as np

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



def convert2IndicatorMatrix(variable, minVal, maxVal):
    '''Convert 1D np array variable (k instances) into k-by-bins indicator matrix'''
    binBoundary = np.arange(minVal, maxVal+1) # increase max a bit so max won't be in a separate bin
    pointBinID = np.digitize(variable,binBoundary)
    binsOrder = np.arange(1,len(binBoundary)+1)
    return (pointBinID[:,np.newaxis] == binsOrder).astype(int)
    
def entropy(indicator_matrix, dim=1):
    '''Use indicator matrix of instance-by-bin to calculate MI.'''
    if dim==1:
        probability = np.sum(indicator_matrix, axis = 0)/np.sum(indicator_matrix)
    else:
        probability = indicator_matrix/np.sum(indicator_matrix)
    logP = np.ma.log2(probability).filled(0)
    return np.sum(-probability * logP)

def MI(X,Y):
    '''Calculate mutual information of variables X and Y.'''
    X = np.round(X-np.min(X))
    Y = np.round(Y-np.min(Y))
    minVal = min(np.min(X),np.min(Y))
    maxVal = max(np.max(X),np.max(Y))
    indicator_X = convert2IndicatorMatrix(X, minVal, maxVal)
    indicator_Y = convert2IndicatorMatrix(Y, minVal, maxVal)
    indicator_XY = np.dot(indicator_X.T,indicator_Y)
    
    H_X = entropy(indicator_X)
    H_Y = entropy(indicator_Y)
    H_XY = entropy(indicator_XY, dim=2)
    mutualInfo = H_X + H_Y - H_XY
    return max(mutualInfo, 0)

# TODO: sparse matrix
# TODO: test cases