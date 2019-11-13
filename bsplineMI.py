import numpy as np

def x2z(X):
    '''Convert raw data onto z range that can be evaluated by bsplineBasis.
    X is a list/numpy array.'''
    return (np.array(X)-np.min(X))/(np.max(X)-np.min(X))

def knotVector(nbins, order):
    '''Internal knots from 1 to nbins-degree, with order preceding and trailing knots.
    Total number of knots are nbins + order.'''
    internal_end = nbins-order+1
    knotVector = [[0]*order, range(1,internal_end),[internal_end]*order]
    return [group/internal_end for groups in knotVector for group in groups]

def bsplineBasis(i, k, U, u, nbins):
    '''Cox de Boor recursion to find b-spline basis function.
    i: knots ID
    k: basis function order
    U: knot vector
    u: value to be evaluated
    '''
    if k == 1:
        if (u >= U[i] and u < U[i+1]) or (u-U[i+1]==0 and i+1 == nbins):
            return 1
        else:
            return 0
    else:
        numer1 = u-U[i]
        denom1 = U[i+k-1]-U[i]
        numer2 = U[i+k]-u
        denom2 = U[i+k]-U[i+1]
        
        if denom1 == 0 and denom2 == 0:
            return 0
        elif denom1 != 0 and denom2 == 0:
            return numer1/denom1 * bsplineBasis(i,k-1,U,u,nbins)
        elif denom1 ==0 and denom2 != 0:
            return numer2/denom2 * bsplineBasis(i+1,k-1,U,u,nbins)
        else:
            return numer1/denom1 * bsplineBasis(i,k-1,U,u,nbins) + numer2/denom2 * bsplineBasis(i+1,k-1,U,u,nbins)

def probabilityTable(z, k, U, nbins):
    '''
    z: data mapped to z range
    k: order of basis function
    U: knot vector
    nbins: number of bins
    '''
    for i_sample in range(len(z)):
        sample = z[i_sample]
        sample_weights = []
        for binID in range(nbins):
            sample_weights.append(bsplineBasis(binID, k, U, sample, nbins))
        if i_sample == 0:
            probTable = np.array(sample_weights).reshape((1,-1))
        else:
            probTable = np.concatenate((probTable,np.array(sample_weights).reshape((1,-1))), axis=0)
    return probTable
        
def entropy(probTable, dim=1):
    '''If dim=1, probTable=nsamples x nbins;
    else, probTable=nbins x nbins x ...'''
    if dim == 1:
        probabilityPerBin = np.sum(probTable,axis=0)/np.sum(probTable)
    else:
        probabilityPerBin = probTable/np.sum(probTable)
    return np.sum(-probabilityPerBin*np.ma.log2(probabilityPerBin).filled(0))

def MI(probTable1, probTable2):
    '''Input probTable of each variables.
    Samples as rows and bins as columns.'''
    return entropy(probTable1) + entropy(probTable2) - entropy(np.dot(probTable1.T,probTable2),dim=2)

def wrapper(X,Y, nbins, order=3):
    '''Wrapper function to find mutual information of X and Y.
    X, Y are lists/numpy arrays.'''
    zx = x2z(X)
    zy = x2z(Y)
    U = knotVector(nbins,order)
    probTablex = probabilityTable(zx,order,U,nbins)
    probTabley = probabilityTable(zy,order,U,nbins)
    return MI(probTablex,probTabley)