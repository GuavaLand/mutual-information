import numpy as np

def knotVector(nbins, degree):
    knotVector = []
    for i in range(1,nbins+1):
        if i < degree:
            knotVector.append(0)
        elif  i>=degree and i<=nbins-1:
            knotVector.append(i-degree+1)
        elif i>nbins-1:
            knotVector.append(nbins-1-degree+2)
    return knotVector

def bsplineBasis(i, p, U, u):
    '''Cox de Boor recursion to find b-spline basis function.
    i: knots ID
    p: degree
    U: knot vector
    u: value to be evaluated
    N_i,p(u) = 1 if u_i<=u<u_i+1;
                0 otherwise
    N_i,p(u) = (u-u_i)/(u_i+p-u_i) * N_i,p-1(u) +
                (u_i+p+1-u)/(u_i+p+1-u_i+1) * N_i+1,p-1(u)
    '''
    if p == 0:
        if u >= U[i] and u < U[i+1]:
            return 1
        else:
            return 0
    else:
        return (u-U[i])/(U[i+p]-U[i]) * bsplineBasis(i,p-1,U,u) +\
                (U[i+p+1]-u)/(U[i+p+1]-U[i+1]) * bsplineBasis(i+1,p-1,U,u)