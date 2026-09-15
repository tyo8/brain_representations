import OrdinaryDiffEqSymplecticRK as ODESymp # KahanLi6
import OrdinaryDiffEqLowOrderRK as ODELow    # RK4
import OrdinaryDiffEqRKN as ODERKN           # DPRKN6, ERKN4
import OrdinaryDiffEq as ODE
import ForwardDiff as FDiff
import ReverseDiff as RDiff
import LinearAlgebra as LA
import Plots

### ASSUMPTIONS
# 1. Initial attractive/repulsive forces correspond to initialization in discrete metric (e.g., placed on non-origin vertices of standard (n+1)-simplex), dynamics initially stationary
# 2. Dynamic data is fully determined by nxn matrix C of pairwise "charges" between objects: (-) --> attractive, (+) --> repulsive, 0 = neutral
#    ==> important implication: "convergent" --> (-), "divergent" --> (+), "incomparable" --> ~0
#    ==> extends the n-body problem "mass matrix" to signed version
# 3. Attractive/repuslive forces add (i.e., superposition of fields) and scales w/ "charges" and inverse square of distance: |F_ij(r)| ~propto~ |C/r^2|
#    ==> conservative forces, so potential energy is well-defined (can use Hamiltonian mechanics formulation)
#
## Governing Lagrangian/Hamiltonian:
#       T(q, dq/dt) = 1/2 * sum_i g_q(dq/dt, dq/dt),
#       V(q) = sum_{i < j} c_{ij} / d_JS(q_i, q_j),
#   where g_q is the Riemannian (in this case, Fisher information) metric at q \in M and d_JS is the Jensen-Shannon distance

metric_list = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalsneath", "sqeuclidean", "yule"]

####################################################################################################################
# main program body
function main(args):

    C, options = loadin(args)
    h_system = hamsys_spec(options, C=C)

    if options.backend == "pyhamsys":
        solution = solve_pyHamSys(h_system, options)
    else:
        raise Exception(f"Hamiltonian system solver not implemented for backend \'{backend}\'")

    write_sol_object(solution, args.outdir, backend=options.backend, verbose=args.verbose)
    ### dimension-reduction stuff (UMAP probably)
    # ????

    ### display/movie part (animation? idk)
    # ????

function loadin(args, debug=False):
    if debug:
        print("loading in args:")
        pprint.pprint(vars(args))

    pij_left = loadtxt(args.left_pvals_fpath)
    pij_right = loadtxt(args.right_pvals_fpath)
    
    C = pairCharge_fromPvals(pij_left, pij_right)

    if debug:
        print(f"Left p-values matrix of size {pij_left.shape}:\n{pij_left}")
        print(f"Right p-values matrix of size {pij_right.shape}:\n{pij_right}")
        print(f"Hamiltonian system is determined by charge matrix of shape {C.shape}:\n{C}")

    solver_params = init_params(
            backend = args.backend, 
            tol = args.tolerance,
            max_iter = args.max_iter
            )
    options = Options(
            backend = args.backend,
            atol = args.tolerance, 
            rtol = args.tolerance * prod(C.shape), 
            avg_method = args.avg_method,
            sys_shape = C.shape,
            int_params=solver_params
            )

    if args.verbose:
        print("\nHamiltonian system is initialized with the following options:")
        pprint.pprint(vars(options))
    return C, options
####################################################################################################################


####################################################################################################################
# Hamiltonian solver functions

# solve a given Hamiltonian system (pyHamSys version)
function solve_pyHamSys(h_system, options, params=None, cmd=None):
    err_str = f"integrator parameters must be specified to use pyHamSys solver(s): \rams=\n{params}\n \noptions.int_params=\n{options.int_params}"
    if options.int_params is None:
        assert params is not None, err_str 
    elif params is None:
        assert options.int_params is not None, err_str 
        params = options.int_params

    X0, Xdot0 = init_sysvars(shape=options.sys_shape, atol=options.atol)

    y0 = ylike_from_varslike(Xdot0, X0)

    dt = params.step
    t_max = params.max_iter * dt
    t_eval = linspace(0, t_max, int(options.eval_len))

    if options.cmd is None:
        if cmd is None:
            sol = h_system.integrate(z0=y0, t_eval=t_eval, params=params)
        else:
            sol = h_system.integrate(z0=y0, t_eval=t_eval, params=params, command=cmd)
    else:
        sol = h_system.integrate(z0=y0, t_eval=t_eval, params=params, command=options.cmd)

    return sol


function solve_jlHamSys(h_system, options, params=None):
    err_str = f"integrator parameters must be specified to use pyHamSys solver(s): \rams=\n{params}\n \noptions.int_params=\n{options.int_params}"
    if options.int_params is None:
        assert params is not None, err_str 
    elif params is None:
        assert options.int_params is not None, err_str 
        params = options.int_params

    ode, _ = jl_load('OrdinaryDiffEq', 'OrdinaryDiffEqDefault')
    ode_symp = jl_load('OrdinaryDiffEqSymplecticRK')
    sol = ode.solve(h_system, ode_symp.KahanLi6(), dt = options.int_params.step);
    return sol


# specify the Hamiltonian system
function hamsys_spec(options, C=None, debug=False):
  err_str = f"integrator parameters are required to specify Hamiltonian system in Julia: \rams=\n{params}\n \noptions.int_params=\n{options.int_params}"
  function ham_eval(X, Xdot, pars):
    return H_xxdot(X, Xdot)

  dep = jl_load('DiffEqPhysics')
  X0, Xdot0 = init_sysvars(shape=options.sys_shape, atol=options.atol)

  assert options.int_params is not None, err_str

  max_time = options.int_params.step * options.int_params.max_iter

  system = dep.HamiltonianProblem(H, p0, q0, (0,max_time))

  return system
end
####################################################################################################################


####################################################################################################################
# Hamiltonian system (definition) functions

# 1. Initial position on non-origin vertices of standard (n+1)-simplex, assumed at rest, with small random perturbations
function init_sysvars(shape=None, N=None, atol=1e-9):
    assert (N is not None) or (shape is not None), "At least one of 'N' and 'shape' must be specified; shape takes priority over N."
    # "particle" "positions":
    if shape is not None:
        assert len(shape) <= 2, "system must be (locally) vector- or matrix-valued"
        X0 = eye(*shape)
    elif N is not None:
        X0 = eye(N)/sqrt(2)
    X0 = X0 + atol * random.rand(*X0.shape)
    Xdot0 = atol * sign(random.rand(*X0.shape))
    return X0, Xdot0

# sets pair interactions from p-value pairs; only its own function because I might decide to do something different with it later, idk.
# A left-significant pair cannot be right-significant (and vice versa). 
function pairCharge_fromPvals(Pij_left, Pij_right, avg_method="log", atol=1e-9, rtol=1e-7):
    # left pvals produce attractive forces
    # right pvals produce repulsive forces
    if avg_method == "log":
        C = log10( abs( safe(Pij_left, atol=atol, rtol=rtol))) - log10( abs( safe(Pij_right, atol=atol, rtol=rtol)))
    elif avg_method == "reciprocal":
        C = 1 / safe(Pij_left) - 1 / safe(Pij_right)
    return C

# assume X and Xdot are non-normalized position and velocity vectors (respectively)
function PQ_of_XdotX(Xdot, X, pnorm=1, atol=1e-9, rtol=1e-7):
    Q, pnormX_i = normvec(X, p=pnorm)
    Q = safe(Q, atol=atol, rtol=rtol, fast=False)
    pnormX_i = safe(pnormX_i, atol=atol, rtol=rtol, fast=False)

    Qbar = mean(Q, axis=1).reshape(-1,1)
    Aq = Qbar * sign(Q) - 1/Q.shape[1]
    U = Qbar * (Xdot - 1/safe(Q, atol=atol, rtol=rtol)) - X * sum( (Xdot * Q * Aq) , axis=1).reshape(-1,1)

    # p_i = u_i / avg(u_i)
    P = U / sum( U, axis=1 ).reshape(-1,1)
    P = safe(P, atol=atol, rtol=rtol, fast=False)
    return P, Q, pnormX_i
end

# kinetic term of Hamiltonian; given as 0.5*g(Qdot, Qdot), where g is the Riemannian metric associated to (inverse) coordinates Qi(Xi) = Xi / ||Xi||
function Kinetic(P, Q, normX, Pbar=None, Qbar=None, atol=1e-9, rtol=1e-7):
    if Qbar is None:
        Qbar = mean(Q, axis=1).reshape(-1,1)
    if Pbar is None:
        Pbar = mean(P, axis=1).reshape(-1,1)

    T_i = 1 / (2 * normX) * ( Pbar**2 / (Qbar*Q.shape[1]) + sum( P**2 / Q , axis=1) )
    T = sum(T_i)
    return T
end

# potential term of Hamiltonian
function Potential(C, Q, atol=1e-9, rtol=1e-7, dist_type="JS", avg_method="signed_geometric", pnorm=1):
    n = Q.shape[0]

    dist_fn = get_dist_fn(dist_type, atol=atol, rtol=rtol, avg_method=avg_method, pnorm=pnorm)
    if not dist_type == 'JS':
        dist_mtx = safe(scidist.squareform( scidist.pdist(Q, metric=dist_fn) ), atol=atol, rtol=rtol)
    else:
        dist_mtx = fast_genmean_JSdist(Q, atol=atol, rtol=rtol, dists_only=True)

    V_ij = - C / dist_mtx
    fill_diagonal(V_ij, 0)
    return V_ij.sum()
end

# total Hamiltonian function, including options
# here "param" is a stand-in for C (as 
function get_Hpq_fn(options, C = None):
    if C is None:
        # substitutes a random dummy set of charge-pair interaction coefficients when none is provided
        raise Warning("No charge matrix provided to \'get_Hpg_fn\': creating random pairwise charge matrix.")
        n = Q.shape[0]
        p0_ij = random.rand(n,n)/20
        p_ij = p0_ij + p0_ij.T
        C = pairCharge_fromPvals(p_ij, 1-p_ij, avg_method="log", atol=options.atol, rtol=options.rtol)

    # v = generalized momenta (p) dummy var
    # u = generalized position (q) dummy var
    T_pq = (lambda v, u, norm_val : Kinetic(
        v, 
        u,
        norm_val,
        atol=options.atol, 
        rtol=options.rtol) 
               )

    # u = generalized position (q) dummy var
    # c = array of pairwise charges (C) dummy var
    V_q = (lambda u, c : Potential(
        u,
        c,
        pnorm=options.pnorm,
        atol=options.atol, 
        rtol=options.rtol, 
        dist_type=options.dist_type,
        avg_method=options.avg_method)
                 )

    function H(X, Xdot, charges):
        P, Q, pnormX = PQ_of_XdotX(
                Xdot,
                X,
                pnorm=options.pnorm,
                atol=options.atol,
                rtol=options.rtol,
                )
        return T_pq(P, Q, pnormX) + V_q(Q, charges)
    return lambda x, xdot : H(x, xdot, C)
end

####################################################################################################################


####################################################################################################################
# analytic derivatives of Hamiltonian (for first-order implementation)

# analytic form of Qdot: assumed not to depend on potential function, so Qdot = get_dHdP = dTdP.
function get_dHdP(P, Q, normX, Pbar=None, Qbar=None, atol=1e-9, rtol=1e-9):
    if Qbar is None:
        Qbar = safe(mean(Q, axis=1).reshape(-1,1), atol=atol, rtol=rtol)
    if Pbar is None:
        Pbar = safe(mean(P, axis=1).reshape(-1,1), atol=atol, rtol=rtol)

    Qdot = P * (Pbar * Q + Qbar) / (Q * Qbar * normX)
    return Qdot
end

# Q-derivative of potential function
# *somewhat* flexible to different potential functions (e.g., euclidean dist or arithmetic mixture case for JS dist)
function get_dVdQ(Q, C=None, Qbar=None, atol=1e-9, rtol=1e-9, dist_type="JS", avg_method="signed_geometric", pnorm=1, debug=False):

    n = len(Q)
    dVdQij = zeros(Q.shape + (n,))

    if Qbar is None:
        Qbar = safe(mean(Q, axis=1).reshape(-1,1), atol=atol, rtol=rtol)
    if C is None:
        # if no pairwise "charge" array is provided, assume universal symmetric attraction
        C = 1 - eye(Q.shape[0])

    if dist_type == "JS":
        Q_pairs, A_pairs, M_pairs, A_norm, M_norm, dist_pairs, JSterm_pairs, Mterm_pairs = fast_genmean_JSdist(
                Q, 
                atol=atol, 
                rtol=rtol, 
                avg_method=avg_method, 
                pnorm=pnorm,
                debug=debug
                )
        # returns JSdists_pairs, (Qi, Qj), A_pairs, M_pairs, A_norm, M_norm, JSterm_pairs, Mterm_pairs

        Qi = Q_pairs[:,0,:]
        Qj = Q_pairs[:,1,:]

        # takes (Qi, Qj, atol=1e-9, rtol=1e-7, KLdiv=None JSdiv=None, M=None, A=None, M_norm=None, A_norm=None, pnorm=1, avg_method="signed_geometric")
        dDiv_dQij = get_dJSdiv_dQi(
                Qi,
                Qj,
                KLdiv = Mterm_pairs,
                JSdiv = JSterm_pairs,
                A = A_pairs,
                A_norm = A_norm,
                M = M_pairs,
                M_norm = M_norm,
                atol=atol, 
                rtol=rtol, 
                avg_method=avg_method, 
                pnorm=pnorm,
                debug=debug
                )

        # derivative is not (generally) symmetric in Qi, Qj its!
        dDiv_dQji = get_dJSdiv_dQi(
                Qj,
                Qi,
                KLdiv = Mterm_pairs,
                JSdiv = JSterm_pairs,
                A = A_pairs,
                A_norm = A_norm,
                M = M_pairs,
                M_norm = M_norm,
                atol=atol, 
                rtol=rtol, 
                avg_method=avg_method, 
                pnorm=pnorm,
                debug=debug
                )

        ij_idx = [p for p in itertools.combinations(range(n),2)]
        I = array([p[0] for p in ij_idx]).astype(int) 
        J = array([p[1] for p in ij_idx]).astype(int)
        dist_mtx = scidist.squareform(dist_pairs.ravel())

        # NOTE: both C and dist_pairs are symmetric matrices
        dVdQij[I,:,J] = dDiv_dQij/2 * C[I,J].reshape(-1,1) / (dist_mtx[I,J]**3).reshape(-1,1)
        dVdQij[J,:,I] = dDiv_dQji/2 * C[J,I].reshape(-1,1) / (dist_mtx[J,I]**3).reshape(-1,1)

    else:
        dist_fn = get_dist_fn(
                dist_type,
                atol=atol,
                rtol=rtol,
                avg_method=avg_method,
                pnorm=pnorm
                )

        dist_mtx = safe(scidist.squareform(scidist.pdist(Q, metric=dist_fn)), atol=atol, rtol=rtol)

        dDiv_dQi = get_div_deriv(
                dist_type,
                atol=atol,
                rtol=rtol,
                avg_method=avg_method,
                pnorm=pnorm
                )

        # efficiency could be substantially improved by either (a) parallelizing or (b) one-stepping the computation below
        # (since [i,j] contributions are independent of one another)
        for i in range(len(Q)):
            for j in range(len(Q)):
                if i != j:
                    if debug:
                        print(f"Q[i] (i={i}) has shape {Q[i].shape}")
                        print(f"Q[j] (j={j}) has shape {Q[j].shape}")
                    # since dDiv_dQi(Qi,Qj) is not a symmetric function, must iterate over i=/=j, not i < j
                    dVdQij[i,:,j] = 1/2 * C[i,j] * dDiv_dQi(Q[i], Q[j]) / (dist_mtx[i,j]**3)

    dVdQi = sum(dVdQij, axis=2)

    if debug:
        print(f"dVdQi: shape={dVdQi.shape}, type={type(dVdQi)}")
    return dVdQi
end

# Q-derivative of kinetic energy function -- for stability reasons, want this to mostly be positive (so that Pdot ~ -|P|^2)
function get_dTdQ(P, Q, normX, Pbar=None, Qbar=None, atol=1e-9, rtol=1e-7, debug=False):
    dTdQ = zeros(P.shape)
    if Qbar is None:
        Qbar = safe(mean(Q, axis=1).reshape(-1,1), atol=atol, rtol=rtol)
    if Pbar is None:
        Pbar = safe(mean(P, axis=1).reshape(-1,1), atol=atol, rtol=rtol)

    if debug:
        print(f"P: shape {P.shape}, type={type(P)}")
        print(f"Q: shape {Q.shape}, type={type(Q)}")
        print(f"normX: shape {normX.shape}, type={type(normX)}")
        print(f"Pbar: shape {Pbar.shape}, type={type(Pbar)}")
        print(f"Qbar: shape {Qbar.shape}, type={type(Qbar)}")

    # maybe should be -1 times below? but then solution blows up in p (i.e., p=o(e^(p^2)))
    dTdQ = 1/normX * ( (1 + normX)*sign(Q)*Kinetic(
        P, 
        Q,
        normX,
        Pbar=Pbar,
        Qbar=Qbar,
        atol=atol,
        rtol=rtol
        ) + 0.5* ( Pbar**2 / (Qbar*Q.shape[0])**2 + P**2 * Q**2 ) )

    if debug:
        print(f"dTdQ: shape={dTdQ.shape}, type={type(dTdQ)}")
    return dTdQ
end
####################################################################################################################


####################################################################################################################
# Utility functions

# it: nxk matrix X (array of vectors), norm order, and flag
# output: array of unit-norm vectors, list of (original) vector norms
function normvec(X, p=1, return_X=True, return_norm=True, axis=1):
    if ndim(X) > 1:
        new_shape = list(X.shape)
        new_shape[axis] = 1
        pnorm_Xi = linalg.norm(X, ord=p, axis=axis).reshape(*new_shape)
        if max(pnorm_Xi.shape) == 1:
            pnorm_Xi = squeeze(pnorm_Xi)
    elif ndim(X) == 1:
        pnorm_Xi = linalg.norm(X, ord=p)
    else:
        pnorm_Xi = abs(X)


    if return_X:
        X_normed = X/pnorm_Xi
        if return_norm:
            return X_normed, pnorm_Xi
        else:
            return X_normed
    else:
        return None, pnorm_Xi
end

# for avoiding 0 in denominators in calculations
function safe(X, atol = 1e-9, rtol = 1e-7, pnorm=1, fast=True):
    if fast:
        eps_val = atol
        X_safe = where(abs(X) < eps_val, eps_val, X)
    else:
        if isinstance(X, ndarray):
            eps_val = min(rtol / len(X) * linalg.norm(X.ravel(), ord=pnorm), atol)
        elif isinstance(X, Number):
            eps_val = min(rtol * abs(X), atol)
        else:
            raise Warning(f"encountered it X of type \"{type(X)}\" -- defaulting to eps={atol} (atol value)")
            eps_val = atol

        X_small = abs(X) < eps_val
        X_neg = X < 0

        X_safe = where(X_small, eps_val, X)
        X_safe = where(X_small & X_neg, -eps_val, X)

    return X_safe
end

# switch statement for distance functions
function get_dist_fn(dist_type, atol=1e-9, rtol=1e-7, avg_method=None, pnorm=1):
    if (dist_type=="JS") and (avg_method=="arithmetic"):
        return "jensenshannon"
    elif dist_type=="JS":
        return lambda q1, q2 : genmean_JSdist(q1,q2,pnorm=pnorm, avg_method=avg_method)
    elif dist_type in metric_list:
        return dist_type
    else:
        raise Exception(f"Unrecogized distance type: {dist_type}")
end

# switch statement for the derivative of the (Bregman) divergence corresponding to the specified distance
function get_div_deriv(dist_type, atol=1e-9, rtol=1e-7, avg_method="signed_geometric", pnorm=1):
    deriv_err = Exception(f"Derivative not implemented for \ndistance type: {dist_type} \navg_method type: {avg_method}. Consider using an autodifferentiation method instead.")

    switch = {
            "JS": (lambda q1, q2 : get_dJSdiv_dQi(q1,q2,pnorm=pnorm, avg_method=avg_method)),
            "euclidean": (lambda q1, q2 : q1 - q2),
     }
    divderiv_fn = switch.get(dist_type, deriv_err)
    return divderiv_fn
end

# function pair for going between state vectors and phase variables
# note that "stack" and "asarray(split())" are inverse functions
# generalizes 'y = concat([p, q])' to arbitrary (shared) P,Q-shapes
function ylike_from_varslike(P, Q):
    y = stack([P, Q])
    return y.ravel()
end

# generalizes 'p, q = split(y, 2)' to arbitrary (shared) P,Q-shapes
function varslike_from_ylike(y, var_shape):
    y0 = y.reshape((2,) + var_shape)
    return y0[0], y0[1]
end

# utility class for computational options specification
class Options():
    function __init__(self, 
                 backend=None, 
                 atol=1e-9, 
                 rtol=1e-7, 
                 dist_type="JS", 
                 avg_method=None, 
                 pnorm=1, 
                 eval_len = 1e3,
                 sys_shape=None, 
                 int_params=None,
                 command=None
                 ):
        self.atol = atol
        self.rtol = rtol
        self.backend = backend
        self.dist_type = dist_type
        self.avg_method = avg_method
        self.pnorm = pnorm
        self.eval_len = eval_len
        self.sys_shape = sys_shape
        self.int_params = int_params
        self.cmd = command
end

# utility class for specification of Hamiltonian solver parameters 
function init_params(backend="pyhamsys", tol=1e-9, max_iter=1e4):
    if backend == "pyhamsys":
        params = pH.Parameters(
                step=1e-2, 
                tol=tol,
                max_iter=max_iter,
                extension=True, 
                check_energy=True, 
                projection="symmetric", 
                solver="BM6"
                )
    else:
        raise Exception(f"solver parameter initialization not implemented for backend \'{backend}\'")
    return params
end

# write Hamiltonian solution to file
function write_sol_object(solution_obj, outdir, backend="pyhamsys", verbose=True):
    if backend == "pyhamsys":
        # write object from the pyhamsys 'solution' class to (JSON) file
        sol_compat = solution_obj.copy()
        for i in solution_obj:
            if isinstance(solution_obj[i], ndarray):
                # converting key {i} with ndarray values of shape {sol[i].shape} to list:
                sol_compat[i] = solution_obj[i].tolist()
        date_str = datetime.datetime.now().strftime('%y_%h_%d_%H-%M-%S')
        outpath = os.path.join(outdir, f"pyHamSys_solution_{date_str}.json")
        with open(outpath, 'w') as fout:
            json.dump(sol_compat, fout, indent=4, sort_keys=True)
    else:
        raise Exception(f"Hamiltonian solution writer not implemented for backend \'{backend}\'")

    print(f"solution saved to: \n{outpath}")
end
####################################################################################################################


####################################################################################################################
# Statistical functions (i.e., JS divergence and generalized averaging functions)

# it: k-vectors q1 and q2 each of unit 1-norm (signed discrete probability distributions w/ same event cardinality)
# output: modified Jensen-Shannon distance between p1 and p2, where the mixture distribution M12 is the (signed) geometric (not arithmetic) mean distribution of q1 and q2.
# NOTE: uses extension of KL_div for non-normalized measures q1, q2: KL(q1,q2) = \int_Q q1 log(q1/q2) + q2 - q1 d\mu(q)
function fast_genmean_JSdist(Q, atol=1e-9, rtol=1e-7, avg_method=None, pnorm=1, dists_only=False, broadcast=True, debug=False):
    Q_pairs = asarray( [qij for qij in itertools.combinations(Q, 2)] )
    A_pairs = safe(mean( Q_pairs, axis=1 ), atol=atol, rtol=rtol)
    M_pairs = safe(gen_mean(Q_pairs, avg_method=avg_method, axis=1, atol=atol, rtol=rtol), atol=atol, rtol=rtol)

    Qi = Q_pairs[:,0,:]
    Qj = Q_pairs[:,1,:]

    JSdists_pairs, A_norm, M_norm, JSterm_pairs, Mterm_pairs = genmean_JSdist(Qi, Qj, A=A_pairs, M=M_pairs, atol=atol, rtol=rtol, axis=1)

    if dists_only:
        dist_mtx = scidist.squareform(JSdists_pairs)
        return dist_mtx

    varset = [A_pairs, M_pairs, A_norm, M_norm, JSdists_pairs, JSterm_pairs, Mterm_pairs]
    varsout = [squeeze(array(var)) for var in varset]
    if broadcast:
        varsout = [ var.reshape(-1,1) if ndim(var) == 1 else var for var in varsout ]

    if debug:
        print(f"Q pairs: shape = {Q_pairs.shape}, type={type(Q_pairs)}")
        print("other vars:")
        for var in varsout:
            print(f"shape = {var.shape} of type \'{type(var)}\'\n")

    return Q_pairs, *varsout 
end

# it: k-vectors q1 and q2 each of unit 1-norm (signed discrete probability distributions w/ same event cardinality)
# output: modified Jensen-Shannon distance between p1 and p2, where the mixture distribution M12 is the (signed) geometric (not arithmetic) mean distribution of q1 and q2.
# NOTE: uses extension of KL_div for non-normalized measures q1, q2: KL(q1,q2) = \int_Q q1 log(q1/q2) + q2 - q1 d\mu(q)
function genmean_JSdist(q1, q2, A=None, M=None, atol=1e-9, rtol=1e-7, avg_method=None, pnorm=1, axis=1, debug=False):

#   q1 = q1.reshape(1,-1)
#   q2 = q2.reshape(1,-1)
#   quot_vec = safe(q1, atol=atol, rtol=rtol) / safe(q2, atol=atol, rtol=rtol) 
    quot_vec = q1 / q2

    # if in regime where numexpr faster than numpy, use numexpr
    big_flag = prod(q1.shape) >= 1e4+1

    if avg_method is None:
        avg_method = "signed_geometric"

    if (A is None) or (M is None):
        q1q2 = squeeze(stack([q1, q2]))

        # arithmetic mean
        if A is None:
            A = gen_mean( q1q2, avg_method="arithmetic" ).reshape(1,-1)
        if M is None:
        # generalized quasi-arithmetic mean
            M = gen_mean( q1q2, avg_method=avg_method ).reshape(1,-1)

        if debug:
            print(f"[q1, q2] (of shape [{q1.shape}, {q2.shape}] --> {q1q2.shape}) = \n{q1q2}")
            print(f"averaging method = \'{avg_method}\'")
            print(f"Generalized mean vector (shape {M.shape}) = \n{M}")

    _, A_norm = normvec(A, axis=axis, p=pnorm, return_X=False)
    _, M_norm = normvec(M, axis=axis, p=pnorm, return_X=False)

    if debug:
        print(f"Arithmetic mean vector norm (shape {A_norm.shape}) = \n{A_norm}")
        print(f"Generalized mean vector norm (shape {M_norm.shape}) = \n{M_norm}")

    # NOTE: uses Nielsen's formula from following reference: JS_m(q1,q2) = JS(q1,q2) + KL(a,m) for aritmetic mean a and generalized mean m
    ##  [1] F. Nielsen, “On a generalization of the Jensen-Shannon divergence and the JS-symmetrization of distances relying on abstract means,” Entropy, vol. 21, no. 5, p. 485, May 2019, doi: 10.3390/e21050485.
    JS_term = (KL_div(q1, A/A_norm) + KL_div(q2, A/A_norm))/2 
    M_term = KL_div(A/A_norm, M/M_norm)

    # if in regime where numexpr faster than numpy, use numexpr
    if big_flag:
        JSdiv = ne.evaluate('JS_term + M_term')
    else:
        JSdiv = JS_term + M_term
    
    if debug:
        print(f"Jensen-Shannon divergence calculated as JSD = {JSdiv} \nfrom M_norm_term = {M_term}, \nand JSdiv_normless = {JS_term}")

    return sqrt(JSdiv), A_norm, M_norm, JS_term, M_term
end

# if q1, q2 have ndim > 1, takes KLdiv along axis ax=1
function KL_div(q1,q2, atol=1e-9, rtol=1e-7, axis=1, debug=False, big_flag=False):

    # if in regime where numexpr faster than numpy, use numexpr
    big_flag = prod(q1.shape) >= 1e4+1
    if big_flag:
        expr = "q1 * ( log( abs(q1) ) - log( abs(q2) ) )"
        KL_all = ne.evaluate(expr)
    else:
        logdiff = log( abs(q1) ) - log( abs(q2) )
        KL_all = q1 * logdiff

    if ndim(q1) > 1:
        kldiv = sum(KL_all, axis=axis)
    else:
        kldiv = sum(KL_all)

    if len(kldiv.shape) > 0:
        if max(kldiv.shape) == 1:
            kldiv = squeeze(kldiv)

    if debug:
        shapes_report = f"\n\tq1 shape = {q1.shape}\n, \tq2 shape = {q2.shape}\n, \tlogdiff shape = {logdiff.shape}\n"
        if (max(ndim(q1), ndim(logdiff)) <= 2) and (min(min(q1.shape), min(logdiff.shape)) == 1):
            q10 = squeeze(q1)
            logdiff0 = squeeze(logdiff)
            err_str = f"sum-computed KLdiv =\\= dot-computed KLdiv: \nsum_kldiv = {kldiv} \ndot_kldiv = {dot(q10,logdiff0)}"
            assert isclose(kldiv, dot(q10, logdiff0), atol=atol, rtol=rtol), shapes_report + err_str
        if big_flag:
            print("Issuing shape report for large its:")
        else:
            print(shapes_report)

    return kldiv
end

function get_dJSdiv_dQi(Qi, Qj, atol=1e-9, rtol=1e-7, KLdiv=None, JSdiv=None, M=None, A=None, M_norm=None, A_norm=None, pnorm=1, avg_method="signed_geometric", debug=False):
    if M is None:
        M = safe(gen_mean( stack([Qi, Qj]), avg_method=avg_method ), atol=atol, rtol=rtol)
    if M_norm is None:
        _, M_norm = normvec( M, p=pnorm, return_X=False)
    M_sgn = sign(M)

    if A is None:
        A = safe((Qi + Qj)/2, atol=atol, rtol=rtol)
    if A_norm is None:
        _, A_norm = normvec( A, p=pnorm, return_X=False)

    big_flag = prod(Qi.shape) >= 1e4+1

    M_sgn = sign(M)
    A_sgn = sign(A)
    Q_sgn = sign(Qi)
    dMdQi = get_dMijdQi(Qi, M, avg_method=avg_method)
    
    if ndim(A) >= 2:
        A_sum = ne.evaluate("sum(A, axis=1)")
    else:
        A_sum = sum(A)

    A_sum = A_sum.reshape(A_norm.shape)

    if JSdiv is None:
        JSdiv = genmean_JSdist(Qi, Qj, avg_method="arithmetic")**2
    if KLdiv is None:
        KLdiv = KL_div(A/A_norm, M/M_norm)

    if debug:
        print(f"Qi has shape: {Qi.shape}")
        print(f"A has shape: {A.shape}")
        print(f"A_sum has shape: {A_sum.shape}")
        print(f"A_norm has shape: {A_norm.shape}")
        print(f"dMdQi has shape: {dMdQi.shape}")
        print(f"JSdiv has shape: {JSdiv.shape}")
        print(f"KLdiv has shape: {KLdiv.shape}")

    if big_flag:
        logNa = ne.evaluate("log(A_norm)")
        sgnA_term = ne.evaluate("1/2 * A_sgn * ( JSdiv - KLdiv - 2 * A_norm * logNa - A_sum/A_norm )")
        sgnQi_term = ne.evaluate("Q_sgn * logNa * ( log(abs(Qi/A)) + 1 )")
        logterm = ne.evaluate("log(abs(A/M)*M_norm/A_norm)")
        dM_term = ne.evaluate("dMdQi * ( M_sgn * A_sum/M_norm - A / M )")
        
        dJSdQi = ne.evaluate( "( sgnA_term + sgnQi_term + 1/2 + logterm + dM_term ) / A_norm" )
    else:
        logNa = log(A_norm)
        sgnA_term = 1/2 * A_sgn * ( JSdiv - KLdiv - 2 * A_norm * logNa - A_sum/A_norm )
        sgnQi_term = Q_sgn * logNa * ( log(abs(Qi/A)) + 1 )
        logterm = log(abs(A/M)*M_norm/A_norm)
        dM_term = dMdQi * ( M_sgn * A_sum/M_norm - A / M )

        dJSdQi = squeeze( sgnA_term + sgnQi_term + 1/2 + logterm + dM_term ) / A_norm

        if ndim(Qi) == 1:
            assert all(Qi.shape == dJSdQi.shape), f"Qi has shape {Qi.shape}, which should (and does not) match {dJSdQi.shape}."

    return dJSdQi
end


## deprecated prior version of dJSdiv_dQi
#ef dJSdiv_dQi00(Qi, Qj, atol=1e-9, rtol=1e-7, M=None, M_norm=None, Qibar=None, Qjbar=None, avg_method="signed_geometric"):
#   if Qibar is None:
#       Qibar = safe(mean(Qi, axis=1).reshape(-1,1), atol=atol, rtol=rtol)
#   if Qjbar is None:
#       Qjbar = safe(mean(Qi, axis=1).reshape(-1,1), atol=atol, rtol=rtol)
#   if M_norm is None:
#       _, M_norm = normvec( gen_mean( stack([q1,q2]), avg_method=avg_method ), p=pnorm, return_X=False)
#
#   M_norm = safe(M_norm, atol=atol, rtol=rtol)
#   Qi = safe(Qi, atol=atol, rtol=rtol)
#   Qj = safe(Qj, atol=atol, rtol=rtol)
#   if avg_method == "signed_gemoetric" or avg_method == "Fisher":
#       # exactly coincides w/ the Jeffreys divergence in this case
#       dMijdQi = 1/2 * power(abs(Qj/Qi) , 1/2)
#       dJSdQi = (1 - Qj / Qi + log( abs(Qi/Qj) ))/4
#   else:
#       # assert avg_method == "arithmetic", "Jensen-Shanon derivatives analytically computed only for arithmetic and (signed) geometric means"
#       M, M_norm = normvec( gen_mean( stack([q1,q2]), avg_method=avg_method ), p=pnorm)
#       if avg_method == "arithmetic":
#           M = safe(M, atol=atol, rtol=rtol)
#           dMijdQi = 1/len(Qi)
#           dJSdQi = log( safe(Qi / M, atol=atol, rtol=rtol) ) - Qj / (M * len(Qj))
#       elif avg_method == "harmonic":
#           dMijdQi = 1/2 * M**2 / Qi**2
#           dJSdQi = 1/2 * ( 1 - Qi/Qj + log(2*(Qi + Qj)) - log(Qj) )
#       else:
#           raise ValueError(f"Averaging method '{avg_method}' not recognized.")
#
#   dNormij_dQi = math.log(M_norm) + len(Qi)/M_norm * (Qibar + Qjbar) * dMijdQi
#
#   return 0.5*dNormij_dQi + dJSdQi



# Generalized mean functions
# it: (n,)-shaped or (n,k)-shaped array of real numbers
# output: scalar or (1,k)-shaped array (respectively) giving the (signed) geometric mean of the its (resp., per-component signed geometric mean)
function gen_mean(X, avg_method=None, param=2, axis=0, atol=1e-12, rtol=1e-8, multidim_compat=True):
    if avg_method is None or avg_method=="arithmetic":
        M = mean(X, axis=axis)

    # slight generalization of geometric mean (param=1 => geometric mean)
    elif avg_method == "Fisher":
        M = exp( -1/param * mean(-param*log(abs(
            safe(X, atol=atol, rtol=rtol))), axis=axis) )

    elif avg_method == "signed_geometric":
        M0 = prod(X, axis=axis)
        M = sign(M0) * power(abs(M0), 1/X.shape[0])

    elif avg_method == "harmonic":
        M = 1 / sum( 1/safe(X, atol=atol, rtol=rtol), axis=axis )

    elif avg_method == "r-mean":
        M = power(1/param, mean( power(X, param), axis=axis) )

    # reshapes 1-dim arrays to correctly right-multiply when broadcast against matrices
    if (ndim(M) == 1) and multidim_compat:
        M = M.reshape(1,-1)
    
    return M
end


# compute derivative of averaging function M(Q1,...,Qn) at Qi (M=Mij --> n=2)
function get_dMijdQi(Qi, M, avg_method="signed_geometric", param=2, n=2):
    if avg_method == "arithmetic":
        dMdQi = zeros(Qi.shape) + 1/2

    # slight generalization of geometric mean (and has same dierivative)
    if (avg_method == "Fisher") or (avg_method == "signed_geometric"):
        dMdQi = M / (n * Qi) 

    elif avg_method == "harmonic":
        dMdQi = 1/n * M**2 / Qi**2

    elif avg_method == "r-mean":
        dMdQi = 1/n * power(Qi, param-1) / power(M, param-1)

    if (ndim(Qi) == 1) and (ndim(M) == 1) and multidim_compat:
        dMdQi = diag(dMdQi)
    
    return dMdQi
end
####################################################################################################################



####################################################################################################################
# parses it, saves output
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Show distributions of outputs from topological bootstrap"
    )
    parser.add_argument(
        "-L",
        "--left_pvals_fpath",
        type=str,
        default="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/dynamics_viz/dummy-pvals_left.csv",
        help="filepath containing grid of left pvals"
    )
    parser.add_argument(
        "-R",
        "--right_pvals_fpath",
        type=str,
        default="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/dynamics_viz/dummy-pvals_right.csv",
        help="filepath containing grid of right pvals"
    )
    parser.add_argument(
        "-o", 
        "--outdir", 
        default="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/dynamics_viz",
        type=str, 
        help="output directory"
    )
    parser.add_argument(
        "-a", 
        "--avg_method", 
        default="signed_geometric",
        type=str, 
        help="method of generalized averaging"
    )
    parser.add_argument(
        "-I", 
        "--max_iter", 
        default=1e4,
        type=int, 
        help="maximum number of solver integrations"
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=1e-9,
        help="system error tolerance"
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default="pyhamsys",
        help="specify Hamiltonian integrator package to use (options are \"pyhamsys\", \"diffeqpy\", or \"mici\")"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="toggle verbose output"
    )
    args = parser.parse_args()

    main(args)
