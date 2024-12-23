## notes: package "pyriemann" must be pip-installed (not available via conda channels) and currently exists only in "neuro" and "Stats" conda envs
import os
import sys
import numpy as np
from pyriemann.utils.mean import mean_ale



# uses regularized mean_ale (AJD-based log-Euclidean mean of spd matrices algorithm from [1]) to approximate geodesic mean of input list of symmetric positive-definite (spd) matrices
def reg_mean_ale(spd_list, options=None, verbose=False):
    """
    References
    ----------
    .. [1] M. Congedo, B. Afsari, A. Barachant, M. Moakher, 'Approximate Joint
        Diagonalization and Geometric Mean of Symmetric Positive Definite
        Matrices', PLoS ONE, 2015
    """
    assert np.all([np.allclose(i,i.T) for i in spd_list]), 'Input matrices must be symmetric (and should be approximately positive semidefinite)'

    spd_stack = np.stack(spd_list, axis=0)

    if not options:
        options = ale_options()

    if verbose:
        print(f"Approximate mean of SPD matrix stack computed with following options set: \n{options}")

    try:
        spd_mean = mean_ale(spd_stack,
            tol=options.tol,
            maxiter=options.maxiter,
            sample_weight=options.weight)

    except (ValueError, FloatingPointError) as err:
        if verbose:
            print("Error: {err}")
            print("At least one matrix in the average fails positive definite criteria; attempting regularization...")
        spd_list = [_regularize_SPD_mtx(i, verbose=verbose) for i in spd_list]
        spd_stack = np.stack(spd_list, axis=0)

        spd_mean = mean_ale(spd_stack,
            tol=options.tol,
            maxiter=options.maxiter,
            sample_weight=options.weight)

    return spd_mean



class ale_options:
    def __init__(self, tol=1e-6, maxiter=50, weight=None):
        self.tol = tol
        self.maxiter = maxiter
        self.weight = weight


# regularizes symmetric input matrix M to guarantee positive definiteness
def _regularize_SPD_mtx(M, tol=1e-6, verbose=True):

    # assume M is symmetric (but may not be *strictly* Hermitian due to machine noise)
    assert np.allclose(M, M.T, rtol=1e-6), 'Matrix is not (sufficiently) symmetric'
    spectrum = np.linalg.eigvalsh(M)
    trace = np.sum(spectrum)

    max_eig = np.amax(spectrum)
    min_eig = np.amin(spectrum)
    if verbose:
        # NOTE that for permuted data, we are in a regularization setting on shaky ground: often, min_eig ~ -0.5 and max_eig ~ 2.5. 
        # Nevertheless, it's noise data. We do what we can.
        print(f"Min/Max eigenvalues: ({min_eig}, {max_eig})")

    # pseudo-condition number: divides largest eigenvalue by largest-magnitude negative eigenvalue (coincides with condition number if M is spd)
    cond_sgn = max_eig/np.abs(min_eig)

    if cond_sgn > 1/tol:
        delta = max_eig*tol
        if min_eig < 0:
            delta = delta + np.abs(min_eig)
    elif min_eig > 0:
        if verbose:
            print("Symmetric metric is well-conditioned positive definite; no regularization required. See spectrum histogram:")
            print(np.histogram(spectrum))
        delta = 0
    elif min_eig < 0:
        if verbose:
            print(f"Symmetric matrix is not (even approximately) positive semidefinite: \"condition number\" is {cond_sgn}. See spectrum histogram:")
            print("Performing simple scaled-idenity regularization anyways.")
            print(np.histogram(spectrum))
        delta = min(max_eig, 1.0)*tol + np.abs(min_eig)
            

    if np.abs(trace) > tol:
        if min_eig < 0:
            assert cond_sgn > 0, 'Input matrix is negative (semi-)definite: max eigenvalue is: '+str(max_eig)
            # assert trace*cond_sgn >= 1/tol, 'Non-positive eigenvalues are too large for simple regularization: unsigned \"condition number\" is '+str(trace*cond_sgn)
        if verbose:
            print(f"pseudo \"condition number\" is {cond_sgn}")
            print(f"value of \"delta\" in expression: delta={delta} \n1/(1 + delta) * (M + delta * np.eye(M.shape[0]))")
        M = 1/(1 + delta) * (M + delta * np.eye(M.shape[0]))
    else:
        # if M is a traceless matrix, send it to Pn by the exponential map
        if verbose:
            print("Matrix is traceless Hermitian! Exponentiate it to the positive definite cone; note that geodesic(expA, expB) = Frobnorm(A - B) by definition of the Lie algebra (for symmetric A, B).")

    if verbose:
        print(f"Trace of matrix is {trace}")
    
    ### debugging code ###
    if verbose:
        spectrum = np.linalg.eigvalsh(M)
        print(f"Min/Max eigenvalues (w reg): {(min(spectrum), max(spectrum))}")
    ### debugging code ###


    return M
