import numpy as np



# checks that input data does not exceed maximum number of dimensions (i.e., entries)
def maxdims(dims, d=2):
    assert len(dims) <= d, 'operation not supported for input of dimension '+str(d)+' or greater'


# checks that input data X,Y have the same shape
def shape(X,Y):
    assert X.shape==Y.shape, 'Input data must have the same shape'
    return X.shape


# checks that input matrix X is a similarity matrix
def sim(X):
    symmetric(X)
    unit_sup(X)
    return X


# checks that input matrix X is symmetric
def symmetric(X):
    assert np.allclose(X, X.T), 'Input should be a symmetric matrix'


# checks that input matrix X has supremum norm = 1
def unit_sup(X):
    assert np.all(np.abs(X) <= 1), 'Entries of input matrix should be between -1 and 1'


# checks that input value x is a whole number
def whole(x):
    assert np.isclose(x, int(x)), 'Input value must be an integer'
