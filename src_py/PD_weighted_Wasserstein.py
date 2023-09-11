import numpy as np

def weighted_Wasserstein_loss(
        X1, X2, w1=None, w2=None, 
        wtfn_type='diff', p=2, ot_imp="POT",
        verbose=False):

    # default to the assumption of perfectly prevalent generators
    if not np.any(w1):
        w1 = np.ones(X1.shape[0],)
    if not np.any(w2):
        w2 = np.ones(X2.shape[0],)

    # compute transport cost matrix (given PD coordinates, p-norm power, and weight application function)
    cost_mtx = Wp_weighted_cost_mtx(X1, X2, w1, w2, wtfn_type=wtfn_type, p=p)

    # diagrams have discrite uniform measures D1=\mu' and D2=\nu'.
    # diagram transport problem is on measures \mu=\mu'+R\nu' and \nu=\nu'+R\mu',
    # where R is the "projection measure" of an arbitrary PD measure to the diagonal.
    signature_1 = np.append(np.ones(w1.shape), len(w2)).flatten()
    signature_2 = np.append(np.ones(w2.shape), len(w1)).flatten()

    if verbose:
        ### debug code ###
        print(f"signature 1 has shape {signature_1.shape}") # and takes the following values:", signature_1)
        print(f"signature 2 has shape {signature_2.shape}") # and takes the following values:", signature_2)
        ### debug code ###

    # checks which optimal transport library user specifies for computation backend
    if ot_imp=="POT":
        import ot
        # uses the python optimal transport library backend 
        Wp_dist = ot.emd2(
                signature_1, 
                signature_2, 
                cost_mtx
                )
    elif ot_imp=="OpenCV":
        import cv2
        # uses the python wrapper for OpenCV
        Wp_dist, _, Wp_flow = cv2.EMD(
                np.array(signature_1, dtype=np.float32), 
                np.array(signature_2, dtype=np.float32), 
                distType = cv2.DIST_USER, 
                cost = np.array(cost_mtx, dtype=np.float32)
                )
    else:
        raise Exception("optimal transport implementation specification not recognized")

    if verbose:
        print(f"Weighted PD Wasserstein-{p} Distance:", Wp_dist)
        print("Optimal Flow Matrix:")
        print(Wp_flow)
    return Wp_dist

def Wp_weighted_cost_mtx(X1, X2, w1, w2, wtfn_type='diff', p=2, verbose=False):
    proj_cost1 = _get_proj_cost(X1, w1, p=p)
    proj_cost2 = _get_proj_cost(X2, w2, p=p)

    inner_cost = _get_inner_cost(X1, X2, w1, w2, wtfn_type=wtfn_type, p=p)
    
    cost_mtx = np.block( [[inner_cost, proj_cost1], [proj_cost2.T, 0] ] )

    if verbose:
        ### debug code ###
        print(f"Block cost matrix has shape {cost_mtx.shape} and takes the following values:", cost_mtx)
        ### debug code ###

    return cost_mtx

def _get_inner_cost(X1, X2, w1, w2, wtfn_type='diff', p=2):
    if wtfn_type == 'diff':

        # Assumes that X1.shape=(n1, d) and X2.shape=(n2, d); broadcasts to get 
        naive_inner_cost = np.linalg.norm( X1[:, None, :] - X2[None, :, :], ord=p, axis=-1 )
        
        # assumes w1, w2 are vectors/lists of scalars
        weight_mtx = np.abs(np.subtract.outer(w1, w2))    
        inner_cost = np.multiply(naive_inner_cost, weight_mtx)

    elif wtfn_type=='measure':
        raise Exception("POT sliced_wasserstein is preferred for this method; exiting computation")
    else:
        raise Exception("unsupported cost function type")
    return inner_cost

# length of perpendicular segment connecting X to its nearest point on y=x; note that 2D assumption (X \subset R2) is important here
def _get_proj_cost(X, w, p=2):
    assert len(X)==len(w), "There must be as many weight values as diagram points."

    # diagonal projection cost for all cycles assuming uniform prevalence scores of 1
    stable_proj_cost = np.abs(np.diff(X, axis=1))*np.power(2, (1-p)/p)

    # prevalence scales diagonal projection cost
    proj_cost = np.multiply(w.flatten(), stable_proj_cost.flatten())
    proj_cost = proj_cost.reshape(len(proj_cost),1)
    
    return proj_cost
