## notes: package "pyriemann" must be pip-installed (not available via conda channels) and currently exists only in "neuro" and "Stats" conda envs
import os
import sys
import numpy as np
from pyriemann.utils.mean import mean_ale



# uses regularized mean_ale (AJD-based log-Euclidean mean of spd matrices algorithm from [1]) to approximate geodesic mean of input list of symmetric positive-definite (spd) matrices
def reg_mean_ale(spd_list, options=None):
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

    try:
        spd_mean = mean_ale(spd_stack,
            tol=options.tol,
            maxiter=options.maxiter,
            sample_weight=options.weight)

    except ValueError:
        spd_list = [_regularize_SPD_mtx(i) for i in spd_list]
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
def _regularize_SPD_mtx(M, tol=1e-6):
    # assume M is symmetric (but may not be *strictly* Hermitian due to machine noise)
    assert np.allclose(M, M.T), 'Matrix is not (sufficiently) symmetric'
    spectrum = np.linalg.eigvalsh(M)

    max_eig = np.amax(spectrum)
    min_eig = np.amin(spectrum)

    # pseudo-condition number: divides largest eigenvalue by largest-magnitude negative eigenvalue (coincides with condition number if M is spd)
    cond_sgn = max_eig/np.abs(min_eig)

    assert cond_sgn > 0, 'Input matrix is negative definite: max eigenvalue is: '+str(max_eig)
    assert cond_sgn >= 1/tol, 'Non-positive eigenvalues are too large for simple regularization: signed \"condition number\" is '+str(cond_sgn)


    if cond_sgn > 1/tol:
        delta = max_eig*tol
        if min_eig < 0:
            delta = delta + np.abs(min_eig)
    else:
        delta = 0

    M = 1/(1 + delta) * (M + delta * np.eye(M.shape[0]))

    return M
