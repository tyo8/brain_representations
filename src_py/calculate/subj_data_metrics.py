import check
import numpy as np


############################################################################################################################################
def p_simdist(R, p=2):
    np.fill_diagonal(R, 1)
    D = np.power(1 - R**p, 1/p)
    return D
############################################################################################################################################


############################################################################################################################################
# implements the (normalized) self-dualizing inner product on the vector space S+ (symmetric positive semidefinite matrices),
# which is given by <A,B> = Tr(A^T B) = Tr(AB), where last equality holds because A is symmetric, 
# giving its unit normalization (i.e., cosine similarity) as <A,B>_cos = Tr(AB)/sqrt[Tr(A^2)*Tr(B^2)]
def spd_cos(X,Y, sym=True, normed=True):
    check.shape(X,Y)

    if sym:
        X = _symmetrize(X)
        Y = _symmetrize(Y)

    # note that we are overloading the signifier "normed" and using it in two distinct senses here: one refers to the magnitude of entries
    # in X and Y, and the other refers to our decision on whether or not to normalize by product of sqrt(var).
    if normed:
        check.unit_sup(X)
        check.unit_sup(Y)
        spd_cos = np.trace(X.T @ Y) / np.sqrt( np.sum(np.abs(X)**2) * np.sum(np.abs(Y)**2)  )
    else:
        if len(X.shape) > 1:
            spd_cos = np.trace(X.T @ Y)
        else:
            spd_cos = X.T @ Y

    return spd_cos

def inner(X,Y):
    x = X.flatten()
    y = Y.flatten()
    inner = spd_cos(x,y, sym=False, normed=False)
    return inner

def Frob_dist(X,Y):
    check.symmetric(X)
    check.symmetric(Y)
    D = X - Y
    dist = np.power( np.sum( np.power(D.flatten(), 2) ), 1/2)   # equal to Tr(D.T @ D) since X and Y are both symmetric
    return dist
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
def _ztrans(X):
    try:
        Xz = np.arctanh(X)
    except:
        print(X)
    return Xz
############################################################################################################################################


############################################################################################################################################
# computes the geodesic distance between X and Y on the positive-definite cone (NOTE: only feasible for very small spd matrices X and Y, n<~50 ??)
def geodesic(X,Y, tol=1e-6):
    check.shape(X,Y)
    X = _symmetrize(X)
    Y = _symmetrize(Y)

    if np.allclose(X, Y):
        return 0
    else:
        from pyriemann.utils.distance import distance_riemann
        try:
            spd_dist = distance_riemann(X,Y)
        except (np.linalg.LinAlgError, FloatingPointError) as err:
            print(f"Error: {err}")
            from pyriemann_addons import _regularize_SPD_mtx
            spd_dist = distance_riemann(
                    _regularize_SPD_mtx(X, verbose=False, tol=tol), 
                    _regularize_SPD_mtx(Y, verbose=False, tol=tol)
                    )
            print(f"Riemannian SPD distance: {spd_dist}")
        return spd_dist
############################################################################################################################################


## HELPER FUNCTIONS
############################################################################################################################################
# checks that input X is either a symmetric matrix or the vectorized upper triangle of one; returns a symmetric matrix of X
def _symmetrize(X, diag_val=1):
    dims = X.shape
    check.maxdims(dims, d=2)
    if len(dims) < 2:
        # X = _symmtx(X, diag_val=diag_val)
        from scipy.spatial.distance import squareform
        X = squareform(X)
        np.fill_diagonal(X, diag_val)

    check.symmetric(X)
    return X


# recomputes a symmetric matrix (assumed to have constant diagonal) from its vectorized upper triangle
def _symmtx(V, diag_val=1):
    k = len(V)
    n = 1/2 * np.sqrt(8*k + 1) + 1/2        # computes n satisfying k = (n choose 2)
    check.whole(n)
    n = int(n)

    M = np.zeros((n,n))
    M[np.triu_indices(n,1)] = V
    M = M + M.T
    np.fill_diagonal(M, diag_val)

    return M

def _regularize_sym(X, eps=1e-6):
    check.symmetric(X)
    n = X.shape[0]

    spectrum = np.linalg.eigvals(X)
    max_l = max(abs(spectrum))

    X = X + max_l*eps*np.eye(n)
    return X

# checks if input matrix X has trace approximately 0 (after max-normalizing entries)
def _trace0(X, tol=1e-6):
    X = X/np.max(np.abs(X))
    return np.abs(np.trace(X)) < tol
############################################################################################################################################
