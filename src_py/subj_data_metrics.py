import check
import numpy as np


############################################################################################################################################
def p_simdist(R, p=2):
    R_normed = np.power(np.abs(R - 1e-12), p)
    D = np.power(1 - R**p, 1/p)
    return D
############################################################################################################################################


############################################################################################################################################
# implements the (normalized) self-dualizing innper product on the vector space S+ (symmetric positive semidefinite matrices),
# which is given by <A,B> = Tr(AB), 
# giving its unit normalization (i.e., cosine similarity) as <A,B>_cos = Tr(AB)/sqrt[Tr(A^2)*Tr(B^2)]
def spd_cos(X,Y):
    check.shape(X,Y)
    X = symmetrize(X)
    Y = symmetrize(Y)

    check.unit_sup(X)
    check.unit_sup(Y)

    spd_cos = np.trace(X @ Y) / np.sqrt( np.sum(np.abs(X)**2) * np.sum(np.abs(Y)**2)  )

    return spd_cos 
############################################################################################################################################


############################################################################################################################################
# computes Pearson similarity of z-transformed Pearson similarity samples
def ztrans_psim(X,Y):
    n = check.shape(X,Y)
    # if vector of coefficients given, they are assumed to be vectorizations of the upper right triangle; otherwise, this is computed
    check.unit_sup(X)
    check.unit_sup(Y)
    if len(n) > 1:
        check.maxdims(n,d=2)
        n = n[0]
        X = X[np.triu_indices(n,1)]
        Y = Y[np.triu_indices(n,1)]

    Xz = _ztrans(X)
    Yz = _ztrans(Y)
    corr_simmtx = np.corrcoef(Xz, Yz)
    corr_sim = corr_simmtx[0,1]

    return corr_sim

# Fisher's z-transformation
def _ztrans(X, noise_term=1e-12):
    Xz = np.arctanh(X - noise_term)
    return Xz
############################################################################################################################################


############################################################################################################################################
# computes the geodesic distance between X and Y on the positive-definite cone (NOTE: only feasible for very small spd matrices X and Y, n<~50 ??)
def geodesic(X,Y):
    check.shape(X,Y)
    X = symmetrize(X)
    Y = symmetrize(Y)

    from pyriemann.utils.distance import distance_riemann
    spd_dist = distance_riemann(X,Y)
    return spd_dist
############################################################################################################################################


############################################################################################################################################
# checks that input X is either a symmetric matrix or the vectorized upper triangle of one; returns a symmetric matrix of X
def symmetrize(X):
    dims = X.shape
    check.maxdims(dims, d=2)
    if len(dims) < 2:
        X = _symmtx(X)

    check.symmetric(X)
    return X


# recomputes a symmetric matrix (assumed to have constant diagonal) from its vectorized upper triangle
def _symmtx(V, diag_val=1):
    k = len(V)
    n = 1/2 * np.sqrt(8*k + 1) + 1/2        # computes n satisfying k = (n choose 2)
    check.whole(n)
    n = round(n)

    M = np.zeros((n,n))
    M[np.triu_indices(n,1)] = V
    M = M + M.T + diag_val*np.eye(n)

    return M
############################################################################################################################################
