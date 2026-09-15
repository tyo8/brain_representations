import os
import math
import json
# import mici
import pprint
import argparse
import datetime
import itertools
import numpy as np
import pandas as pd
import numexpr as ne
import seaborn as sns
import pyhamsys as pH
import figutils as futils

import plotly.express as px

import umap
import umap.aligned_umap

from scipy import integrate as itg
from scipy import interpolate as itp
from numbers import Number
from matplotlib import pyplot as plt
from matplotlib import animation as anim
from scipy.spatial import distance as scidist
from diffeqpy import load_julia_packages as jl_load

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

metric_list = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalsneath", "sqeuclidean", "yule", "JS"]

####################################################################################################################
# main program body
def main(args):

    C, options = loadin(args)
    h_system = hamsys_spec(options, C=C)

    if options.backend == "pyhamsys":
        solution = solve_pyHamSys(h_system, options)
    else:
        solution = solve_jlHamSys(h_system, options)
        raise Exception(f"Hamiltonian system solver not implemented for backend \'{options.backend}\'")

    solution.C = C
    solution.options = options

    ### dimension-reduction and visualization (aligned UMAP)
    fname, outpath = get_fpath(options=options, outdir=args.outdir)
    solution = plot_sol_summary(solution=solution, fname=fname, options=options, outdir=args.outdir, show_figs=args.show_figs)

    ### GIF animation of solution
    if options.backend == "pyhamsys":
        sol_fig = animate_sol(Qt=solution.Qt_embed, time=solution.t)
    else:
        t0 = np.linspace(0,1, solution.Qt_embed.shape[-1])
        sol_fig = animate_sol(Qt=solution.Qt_embed, t=t0)

    if args.show_figs:
        sol_fig.show()

    futils._write_html(
            sol_fig, 
            outpath.replace('json', 'html'), 
            frame_duration = 1000/options.eval_len   # frame duration in milliseconds
            )
    # futils._write_gif(fig=sol_fig, outpath=outpath.replace('json','gif'))
    write_solution(solution, outpath, backend=options.backend, verbose=args.verbose)

    return None



def loadin(args, debug=False):
    if debug:
        print("loading in args:")
        pprint.pp(vars(args), sort_dicts=True)

    try:
        C = np.loadtxt(args.pairwise_fpath)
        prob_name = os.path.basename(args.pairwise_fpath).split('.')[0].replace("_C",'')
        if debug:
            print(f"Hamiltonian system parameterized by pairwise interaction matrix of shape {C.shape}:\n{C}")
    except FileNotFoundError:
        pij_left = np.loadtxt(args.left_pvals_fpath)
        pij_right = np.loadtxt(args.right_pvals_fpath)
        C = pairCharge_fromPvals(pij_left, pij_right)
        prob_name = os.path.basename(args.left_pvals_fpath).split('.')[0].replace("_left",'')

        if debug:
            print(f"Left p-values matrix of size {pij_left.shape}:\n{pij_left}")
            print(f"Right p-values matrix of size {pij_right.shape}:\n{pij_right}")
            print(f"Hamiltonian system parameterized by pairwise interaction matrix of shape {C.shape}:\n{C}")

    solver_params = init_params(
            backend = args.backend, 
            tol = args.tolerance,
            max_iter = args.max_iter
            )
    options = Options(
            problem_type = prob_name,
            backend = args.backend,
            dist_type = args.dist_type,
            atol = args.tolerance, 
            rtol = args.tolerance * np.prod(C.shape), 
            avg_method = args.avg_method,
            sys_shape = C.shape,
            renormalize = args.renorm,
            int_params = solver_params  # integrator parameters
            )

    min_dist = comp_mindist(
            shape=options.sys_shape, 
            dist_type=options.dist_type, 
            avg_method=options.avg_method, 
            atol=options.atol, 
            min_sep=options.min_dist
            )
    options.min_dist = min_dist

    if args.verbose:
        X0, Xdot0 = init_sysvars(shape=options.sys_shape, dist_type=options.dist_type, atol=options.atol)
        print("\nFree N-body Hamiltonian system is parameterized by the following initial data:")
        print(f"\t{C.shape} matrix C of interaction (potential) terms: \n{C}")
        print(f"\t{X0.shape} initial positions: \n{X0}")
        print(f"\t{Xdot0.shape} initial velocities: \n{Xdot0}\n\n")
        print("and the following set of options:")
        pprint.pp(vars(options), sort_dicts=True)
    return C, options
####################################################################################################################


####################################################################################################################
# Display and analysis functions

# pass to analysis from file
def fpath_plot_sol_summary(fpath):
    with open(fpath, 'r') as fin:
        d = json.load(fin)

    for key in d:
        val = d[key]
        if isinstance(val, list):
            d[key] = np.array(val)

    obj = dict_obj(d=d)
    return obj

# parent analysis function
def plot_sol_summary(solution=None, options=None, outdir='', fname='', target_dim=2, show_figs=True):
    fname = fname.split('.')[0]

    time = solution.t
    Qsol, Psol = np.split(solution.y, 2)

    ydot_t  = np.array([ y_dot(y, solution.C, options) for y in solution.y.transpose() ]).transpose()
    solution.ydot = ydot_t
    Qdot_sol, Pdot_sol = np.split(ydot_t, 2)

    total_shape = solution.options.sys_shape + (len(time),)
    Qt = Qsol.reshape(total_shape)
    Qdot_t = Qdot_sol.reshape(total_shape)
    Pt = Psol.reshape(total_shape)
    Pdot_t = Pdot_sol.reshape(total_shape)

    dist_type = solution.options.dist_type
    avg_method = solution.options.avg_method

    if show_figs:
        do_prelim_plots(val=Qt, time=time, avg_method=avg_method, meta=fname, vartype="Q", outdir=outdir, show=show_figs)
        do_prelim_plots(val=Qdot_t, time=time, avg_method=avg_method, meta=fname, vartype="Qdot", outdir=outdir, show=show_figs)
        do_prelim_plots(val=Pt, time=time, meta=fname, vartype="P", outdir=outdir, show=show_figs)
        do_prelim_plots(val=Pdot_t, time=time, meta=fname, vartype="Pdot", outdir=outdir, show=show_figs)

    # if natively solved in high-dim space, reduce to target_dim-d space for visualization:
    if Qt.shape[1] > target_dim:
        Qt_embed = umap_sol_redux(Qt, time, dist_type=dist_type, min_dist=solution.options.min_dist, avg_method=avg_method, target_dim=target_dim)
    else:
        Qt_embed = Qt

    solution.Qt_embed = Qt_embed

    return solution

# preliminary/unsophisticated plots
def do_prelim_plots(val, time, avg_method="arithmetic", outdir='', meta=None, vartype="Q", show=True):
    title = f"{vartype}: {meta} \n({avg_method} mean value)"
    fig, ax = plt.subplots()

    if np.ndim(val) > 2:
        avg_val = gen_mean(val, avg_method=avg_method)
    elif np.ndim(val) > 1:
        avg_val = gen_mean(val, avg_method=avg_method, multidim_compat=True)
    else:
        raise Exception("Input array expected to be >1-dimensional. Instead, input.shape={val.shape} and time.shape={t.shape}.")

    for i,y in enumerate(avg_val):
        sns.lineplot(x=time, y=y, ax=ax, label=f"Particle {i} avg")

    ax.set_xlabel("Time")
    ax.set_ylabel(vartype)
    ax.set_title(title)

    fname = f"avgplot_{meta}_{vartype}.png"
    outpath = os.path.join(outdir, fname)

    if show:
        plt.show()
    
    futils._write_img(fig=fig, outpath=os.path.join(outdir, f"{fname}".replace('json','png')))

    return None 

# use Aligned_UMAP to (semi-continuously) project dynamics into visualizable dimensions
def umap_sol_redux(Qt, time, dist_type="JS", avg_method="euclidean", min_dist=1e-3, pnorm=1, window_size=3, target_dim=2, verbose=True):
    N = Qt.shape[0]

    # UMAP 'metric="precomputed"' assumes/requires square distance matrix, rather than upper-right-triangle vector
    tdist_arr = np.array([ scidist.squareform(get_distvals(Q, dist_type, avg_method=avg_method, min_dist=min_dist, pnorm=pnorm)) for Q in Qt])

    tdist_windowed, tdist_relations = make_aumap_relations(tdist_arr, win_sz=window_size)

    if verbose:
        print("Running aligned UMAP (window_size={win_sz}) to reduce (D={N})->(d={target_dim}) on problem of size {Qt.shape}, nt={len(t)}...")

    aligned_mapper = umap.aligned_umap.AlignedUMAP(
            metric="precomputed",
            n_neighbors=round(N/5),
            n_components=target_dim,
            alignment_regularisation=0.1,
            alignment_window_size=window_size,
            n_epochs=200,
            random_state=42,
            ).fit(tdist_windowed, relations=tdist_relations)
    
    return aligned_mapper.embeddings_

def make_aumap_relations(tdist_arr, win_sz=3, debug=True):
    # sliding dist
    spdims = tdist_arr.shape[:-1]
    nt = tdist_arr.shape[-1]

    if debug:
        print(f"input array of shape {tdist_arr.shape}, window size={win_sz}")

    # produces array of dimensions (win_sz, *spdims, nt)
    tdist_windowed = [ tdist_arr[..., i:i+win_sz] for i in range(nt - win_sz) ]
    window_shape = tdist_windowed[0].shape

    ### umap relations mapping convention is 'target: source'
    # identifies distances between x_i and x_j in window w_k at times 1:win_sz
    # with distances between x_i and x_j in window w_k+1 at times 0:win_sz-1
    idx_arr = np.array(list(range(np.prod(window_shape)))).reshape(window_shape)
    rln_dict = { int(pair[0]):int(pair[1]) for pair in zip(idx_arr[...,1:].ravel(), idx_arr[...,:-1].ravel()) }

    tdist_relations = [ rln_dict.copy() for entry in tdist_windowed ]

    return tdist_windowed, tdist_relations

# maybe redo with plotly??
def animate_sol(Qt, time, cmap="Spectral", marker_size=25, zoom=0.95):
    ndim = Qt.shape[1]

    names = ["particle", "coord_set", "time"]
    idx_list = [ [f"q_{i}" for i in range(Qt.shape[0])], ['x','y','z'][0:ndim], time ]
    idx = pd.MultiIndex.from_product(idx_list, names=names)
    Df = pd.DataFrame({'Qt': Qt.flatten()}, index=idx)['Qt']; 
    df = Df.unstack("coord_set").reset_index()

    df["mk_sz"] = marker_size
    xwin = list(_get_window(df['x'], zoom=zoom))
    ywin = list(_get_window(df['y'], zoom=zoom))

    if ndim == 2:
        fig = px.scatter(
                data_frame=df, color="particle", size="mk_sz",
                x='x', y='y', range_x=xwin, range_y=ywin, 
                animation_frame="time"
                )
    else:
        zwin = list(_get_window(df['z'], zoom=zoom))
        assert ndim==3, f"Animations only supported for 2d and 3d datasets; instead, found ndim3={ndim}. Note that smooth interpolation is only available in 2d."
        fig = px.scatter_3d(
                data_frame=df, color="particle", size="mk_sz",
                x='x', y='y', z='z', 
                range_x=xwin, range_y=ywin, range_z=zwin,
                animation_frame="time"
                )
    return fig

def _get_window(values, zoom=0.95):
    win = np.array([min(values), max(values)]).flatten()
    width = np.diff(win)/2
    center = np.mean(win)
    new_win = np.array([center - width/zoom, center + width/zoom]).flatten()
    return new_win

####################################################################################################################


####################################################################################################################
# Hamiltonian solver functions

# solve a given Hamiltonian system (pyHamSys version)
def solve_pyHamSys(h_system, options, params=None, cmd=None):
    err_str = f"integrator parameters must be specified to use pyHamSys solver(s): \nparams=\n{params}\n \noptions.int_params=\n{options.int_params}"
    if options.int_params is None:
        assert params is not None, err_str 
    elif params is None:
        assert options.int_params is not None, err_str 
        params = options.int_params

    X0, Xdot0 = init_sysvars(shape=options.sys_shape, dist_type=options.dist_type, atol=options.atol)

    y0 = ylike_from_varslike(X0, Xdot0)

    dt = params.step
    t_max = options.eval_len * dt
    t_eval = np.linspace(0, t_max, int(options.eval_len))

    if options.cmd is None:
        if cmd is None:
            sol = h_system.integrate(z0=y0, t_eval=t_eval, params=params)
        else:
            sol = h_system.integrate(z0=y0, t_eval=t_eval, params=params, command=cmd)
    else:
        sol = h_system.integrate(z0=y0, t_eval=t_eval, params=params, command=options.cmd)

    return sol


def solve_jlHamSys(h_system, options, params=None):
    err_str = f"integrator parameters must be specified to use pyHamSys solver(s): \nparams=\n{params}\n \noptions.int_params=\n{options.int_params}"
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
def hamsys_spec(options, C=None, debug=False):
    H_xxdot = get_Hpq_fn(options, C=C)

    if options.backend == "pyhamsys":
        def ham_eval(t, y):
            X, Xdot = varslike_from_ylike(y, options.sys_shape)
            Energy = H_xxdot(X, Xdot)
            if debug:
                print(f"\n\n*** Total System Energy ***")
                print(f"{Energy}, \t{type(Energy)}")
                print(f"*** Total System Energy ***\n\n")
                exit()
            return Energy

        # (assume) Hamiltonian carries no explicit time dependence
        k_dot = lambda t, y: 0

        # y_dot(y) *also* carries no explicit time dependence
        y_dot_local = lambda t, y: y_dot(y, C, options)

        system = pH.HamSys(
                hamiltonian = ham_eval,
                y_dot = y_dot_local,
                ndof = options.sys_shape[0],
                k_dot = k_dot
                )
        return system
    elif options.backend == "julia":
        err_str = f"integrator parameters are required to specify Hamiltonian system in Julia: \noptions.int_params=\n{options.int_params}"
        assert options.int_params is not None, err_str
        
        def ham_eval(X, Xdot, pars):
            return H_xxdot(X, Xdot)

        dep = jl_load('DiffEqPhysics')
        X0, Xdot0 = init_sysvars(shape=options.sys_shape, dist_type=options.dist_type, atol=options.atol)

        system = dep.HamiltonianProblem(ham_eval, X0, Xdot0, (0,options.int_params.max_time))

        return system

    else:
        raise Exception(f"Hamiltonian system definition not implemented for backend \'{options.backend}\'")


def y_dot(y, C, options, ignore_dTdQ=False, debug=False):
    if options.renormalize:
        X, Xdot = varslike_from_ylike(y, options.sys_shape)
        Q, P, pnormX = QP_of_XXdot(
                X,
                Xdot,
                dist_type=options.dist_type,
                pnorm=options.pnorm,
                atol=options.atol,
                rtol=options.rtol,
                )
    else:
        Q, P = varslike_from_ylike(y, options.sys_shape)
        _, pnormX = normvec(Q, return_X=False)

    Qbar = safe(np.mean(Q, axis=1).reshape(-1,1), atol=options.atol, rtol=options.rtol)
    Pbar = safe(np.mean(P, axis=1).reshape(-1,1), atol=options.atol, rtol=options.rtol)

    if debug:
        print(f"Q: \n{Q}")
        # print(f"Q, P: \n{Q, P}")
        # print(f"Q: shape={Q.shape}, type={type(Q)}")
        # print(f"P: shape={P.shape}, type={type(P)}")
        # print(f"pnormX: shape={pnormX.shape}, type={type(pnormX)}")

    dVdQ, dDivdQ, dist_mtx = get_dVdQ(
            Q, C=C, Qbar=Qbar,
            atol = options.atol,
            rtol = options.rtol,
            dist_type = options.dist_type, 
            avg_method = options.avg_method,
            pnorm = options.pnorm,
            min_dist = options.min_dist
            )

    P = collision_corrector(P, dist_mtx, dDivdQ, min_dist=options.min_dist)

    Qdot = get_dHdP(
            Q, P, pnormX, 
            Pbar=Pbar, Qbar=Qbar,
            dist_type=options.dist_type,
            atol=options.atol, rtol=options.rtol
                )

    # if non-Euclidean, Riemannian metric (potentially) depends on Q! then dTdQ=dgdQ(Pi,Pj)
    # (where g is the Riemannian metric; vanishes if grad_Q is Levi-Civita connection)
    if ignore_dTdQ:
        Pdot = - dVdQ
    else:
        dTdQ = get_dTdQ(
                Q, P, pnormX, 
                dist_type = options.dist_type,
                Pbar=Pbar, Qbar=Qbar, 
                atol=options.atol, rtol=options.rtol
                )
        Pdot = - dVdQ - dTdQ

    if debug:
        print(f"Qdot: \n{Qdot}")
        # print(f"Qdot, Pdot: \n{Qdot, Pdot}")
        # print(f"Qdot: shape={Qdot.shape}, type={type(Qdot)}")
        # print(f"Pdot: shape={Pdot.shape}, type={type(Pdot)}")

    return ylike_from_varslike(Qdot, Pdot)

####################################################################################################################


####################################################################################################################
# Hamiltonian system (definition) functions

# 1. Initial position on non-origin vertices of standard (n+1)-simplex, assumed at (relative, internally) rest, with small random perturbations
def init_sysvars(shape=None, dist_type="JS", N=None, atol=1e-9):
    assert (N is not None) or (shape is not None), "At least one of 'N' and 'shape' must be specified; shape takes priority over N."
    # "particle" "positions":
    if shape is not None:
        assert len(shape) <= 2, "system must be (locally) vector- or matrix-valued"
        # X0 = 2*np.eye(*shape) - 1
        if N is None:
            N = shape[0]
    elif N is not None:
        shape = (N,N)
    
    base = np.eye(*shape); fbase = np.flip(base, axis=1)
    # X0 = base
    # X0 = 2*base - 1
    X0 = np.flip(np.triu(fbase), axis=1) - np.tril(fbase)

    # Xdot0 = base
    # Xdot0 = 1 - X0
    # Xdot0 = np.flip(X0, axis=1)
    Xdot0 = np.full(X0.shape, 0)
    # Xdot0 = np.full(X0.shape, 1)
    # Xdot0 = np.full(X0.shape, -atol)
    # Xdot0 = atol * np.sign(np.random.rand(*X0.shape))
    # Xdot0 = ([(-1)**i for i in range(N)] * np.ones(X0.shape)).T
    # Xdot0 = ([(-N/2)+i for i in range(N)] * np.ones(X0.shape)).T
    if dist_type=="JS":
        return X0/N, Xdot0/N
    else:
        return X0, Xdot0


def comp_mindist(shape=None, N=None, dist_type="JS", avg_method=None, pnorm=1, atol=1e-9, min_sep=5e-4):
    if shape is not None:
        if N is None:
            N = shape[0]
    elif N is not None:
        shape = (N,N)

    q1 = np.ones(N)
    q2 = q1.copy()
    q2[0] = 1 - min_sep    # minimum displacement separating two spatial vectors
    Q = np.vstack([q1,q2])
    dist = get_distvals(Q, dist_type=dist_type, avg_method=avg_method, pnorm=pnorm, min_dist=atol)
    if np.ndim(dist) > 0:
        dist = dist[0]     # should be single scalar value
    return float(dist)


def _gate_dists(dists, min_dist=1e-3):
    dists = np.where( dists < min_dist, min_dist, dists )
    return dists


# sets pair interactions from p-value pairs; only its own function because I might decide to do something different with it later, idk.
# A left-significant pair cannot be right-significant (and vice versa). 
def pairCharge_fromPvals(Pij_left, Pij_right, weight_method="log", atol=1e-9, rtol=1e-7):
    # left pvals produce attractive forces
    # right pvals produce repulsive forces
    if weight_method == "log":
        C = np.log10( np.abs( safe(Pij_left, atol=atol, rtol=rtol))) - np.log10( np.abs( safe(Pij_right, atol=atol, rtol=rtol)))
    elif weight_method == "reciprocal":
        C = 1 / safe(Pij_left) - 1 / safe(Pij_right)

    np.fill_diagonal(C, 0)
    return C



# assume X and Xdot are non-normalized position and velocity vectors (respectively)
def QP_of_XXdot(X, Xdot, dist_type="JS", pnorm=1, atol=1e-9, rtol=1e-7):
    if dist_type == "JS":
        Q, pnormX_i = normvec(X, p=pnorm)
        Q = safe(Q, atol=atol, rtol=rtol, fast=False)
        pnormX_i = safe(pnormX_i, atol=atol, rtol=rtol, fast=False)

        Qbar = np.mean(Q, axis=1).reshape(-1,1)
        Aq = Qbar * np.sign(Q) - 1/Q.shape[1]
        U = Qbar * (Xdot - 1/safe(Q, atol=atol, rtol=rtol)) - X * np.sum( (Xdot * Q * Aq) , axis=1).reshape(-1,1)

        # p_i = u_i / avg(u_i)
        P = U / np.sum( U, axis=1 ).reshape(-1,1)
        # P = safe(P, atol=atol, rtol=rtol, fast=False)
        P = safe(P, atol=atol, rtol=rtol, fast=True)
    elif dist_type == "euclidean":
        Q = X
        P = Xdot
        pnormX_i = normvec(X, p=pnorm, return_X=False)
    return Q, P, pnormX_i

# kinetic term of Hamiltonian; given as 0.5*g(Qdot, Qdot), where g is the Riemannian metric associated to (inverse) coordinates Qi(Xi) = Xi / ||Xi||
def Kinetic(Q, P, normX, dist_type="JS", Pbar=None, Qbar=None, atol=1e-9, rtol=1e-7):
    if Qbar is None:
        Qbar = np.mean(Q, axis=1).reshape(-1,1)
    if Pbar is None:
        Pbar = np.mean(P, axis=1).reshape(-1,1)

    if dist_type=="JS":
        T_i = 1 / (2 * normX) * ( Pbar**2 / (Qbar*Q.shape[1]) + np.sum( P**2 / Q , axis=1) )
    elif "euclidean" in dist_type:
        T_i = 1 / 2 * P**2
    else:
        T_i = 1 / (2 * normX) * P**2

    T = np.sum(T_i)
    return T

# potential term of Hamiltonian
# NOTE: implement minimum distance?
def Potential(Q, C, atol=1e-9, rtol=1e-7, dist_type="JS", avg_method="signed_geometric", pnorm=1, min_dist=1e-3, debug=False):
    n = Q.shape[0]

    d_ij = get_distvals(Q, dist_type, atol=atol, rtol=rtol, avg_method=avg_method, min_dist=min_dist, pnorm=pnorm)
    n_collision = np.sum(d_ij <= min_dist)

    # np.fill_diagonal(dist_mtx, 1)

    ### SIGN CONVENTION: C_ij > 0 => i&j repulse, C_ij < 0 => i&j attract 
    if debug:
        print(f"pairwise interaction matrix (for potential computation): {C}")
        print(f"distance matrix (for potential computation): {dist_mtx}")

    # upper right triangle vector of (symmetric matrix) C
    C_ij = scidist.squareform(C)

    # sum over i<j; collision events change the potential energy function
    V = -np.sum(C_ij/d_ij) + max(np.abs(C_ij))*n_collision/min_dist
    return V

# total Hamiltonian function, including options
# here "param" is a stand-in for C (as 
def get_Hpq_fn(options, C = None):
    if C is None:
        # substitutes a random dummy set of charge-pair interaction coefficients when none is provided
        raise Warning("No charge matrix provided to \'get_Hpg_fn\': substituting uniform pairwise attraction matrix.")
        C = 1 - np.eye(Q.shape[0])
#       n = Q.shape[0]
#       p0_ij = np.random.rand(n,n)/20
#       p_ij = p0_ij + p0_ij.T
#       C = pairCharge_fromPvals(p_ij, 1-p_ij, avg_method="log", atol=options.atol, rtol=options.rtol)

    # u = generalized position (q) dummy var
    # v = generalized momenta (p) dummy var
    T_pq = (lambda u, v, norm_val : Kinetic(
        u,
        v, 
        norm_val,
        dist_type=options.dist_type,
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
        min_dist=options.min_dist,
        avg_method=options.avg_method)
                 )

    def H(X, Xdot, charges):
        Q, P, pnormX = QP_of_XXdot(
                X,
                Xdot,
                dist_type=options.dist_type,
                pnorm=options.pnorm,
                atol=options.atol,
                rtol=options.rtol,
                )
        return T_pq(Q, P, pnormX) + V_q(Q, charges)
    return lambda x, xdot : H(x, xdot, C)

####################################################################################################################


####################################################################################################################
# analytic derivatives of Hamiltonian (for first-order implementation)

# analytic form of Qdot: assumed not to depend on potential function, so Qdot = get_dHdP = dTdP.
def get_dHdP(Q, P, normX, dist_type="JS", Pbar=None, Qbar=None, atol=1e-9, rtol=1e-9):
    if dist_type == "JS":
        if Qbar is None:
            Qbar = safe(np.mean(Q, axis=1).reshape(-1,1), atol=atol, rtol=rtol)
        if Pbar is None:
            Pbar = safe(np.mean(P, axis=1).reshape(-1,1), atol=atol, rtol=rtol)

        Qdot = P * (Pbar * Q + Qbar) / (Q * Qbar * normX)
    elif dist_type == "euclidean":
        # unit mass assumption! (implies T = 1/2 * P**2)
        Qdot = P
    else:
        raise ValueError(f"dHdP is not computed for distance metric of type \'{dist_type}\'")
    return Qdot

# Q-derivative of potential function
# *somewhat* flexible to different potential functions (e.g., euclidean dist or arithmetic mixture case for JS dist)
def get_dVdQ(Q, C=None, Qbar=None, atol=1e-9, rtol=1e-9, dist_type="JS", avg_method="signed_geometric", pnorm=1, min_dist=1e-3, debug=False):

    n = len(Q)
    dVdQij = np.full(shape = Q.shape+(n,), fill_value=0)
    dDivdQ = np.full(shape = Q.shape+(n,), fill_value=0)
    ij_idx = [p for p in itertools.combinations(range(n),2)]
    I = np.array([p[0] for p in ij_idx]).astype(int) 
    J = np.array([p[1] for p in ij_idx]).astype(int)

    if Qbar is None:
        Qbar = safe(np.mean(Q, axis=1).reshape(-1,1), atol=atol, rtol=rtol)
    if C is None:
        # if no pairwise "charge" array is provided, assume universal symmetric attraction
        C = 1 - np.eye(Q.shape[0])

    # vectorize JS calculation for speedup
    if dist_type == "JS":
        Q_pairs, A_pairs, M_pairs, A_norm, M_norm, dist_pairs, JSterm_pairs, Mterm_pairs = fast_genmean_JSdist(
                Q, 
                atol=atol, 
                rtol=rtol, 
                avg_method=avg_method, 
                pnorm=pnorm,
                dists_only=False,
                debug=debug
                )
        # returns JSdists_pairs, (Qi, Qj), A_pairs, M_pairs, A_norm, M_norm, JSterm_pairs, Mterm_pairs
        d_ij = dist_pairs.ravel()

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

        # derivative is not (generally) symmetric in Qi, Qj inputs!
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

        dist_mtx = scidist.squareform(d_ij)

        # NOTE: both C and dist_pairs are symmetric matrices
        dVdQij[I,:,J] = dDiv_dQij/2 * C[I,J].reshape(-1,1) / (dist_mtx[I,J]**3).reshape(-1,1)
        dVdQij[J,:,I] = dDiv_dQji/2 * C[J,I].reshape(-1,1) / (dist_mtx[J,I]**3).reshape(-1,1)
    
    # vectorize euclidean calculation for speedup
    elif dist_type == "euclidean":
        Q_pairs = np.asarray( [qij for qij in itertools.combinations(Q, 2)] )
        Q_diff = np.squeeze(np.diff(Q_pairs, axis=1), axis=1)
        d_ij = _gate_dists(np.sqrt(np.sum(Q_diff**2, axis=1)), min_dist=min_dist)
        C_ij = scidist.squareform(C)

        dVdQij_0 = [scidist.squareform( dq * C_ij / d_ij**3 / 2 ) for dq in Q_diff.T ]
        
        # antisymmetrize
        dVdQij = [ np.triu(dv) - np.tril(dv) for dv in dVdQij_0 ]

        # permute to standardize layout
        dVdQij = np.permute_dims(np.array(dVdQij), (2, 0, 1))

        # standardize for collision inputs
        dist_mtx = scidist.squareform(d_ij)
        dDiv_dQij = Q_diff
        dDiv_dQji = -Q_diff

    else:
        d_ij = get_distvals(
                Q,
                dist_type,
                atol=atol,
                rtol=rtol,
                avg_method=avg_method,
                min_dist=min_dist,
                pnorm=pnorm
                )

        dist_mtx = scidist.squareform(d_ij)

        dDiv_dQi = get_div_deriv(
                dist_type,
                atol=atol,
                rtol=rtol,
                avg_method=avg_method,
                pnorm=pnorm
                )
        dDiv_dQij = dDiv_dQi(Q[I], Q[J])
        dDiv_dQji = dDiv_dQi(Q[J], Q[I])

        # more efficient (when possible) to (a) parallelize or (b) vectorize the computation below (implemented for special cases above)
        for i in range(len(Q)):
            for j in range(len(Q)):
                if i != j:
                    dVdQij[i,:,j] = 1/2 * C[i,j] * dDiv_dQi(Q[i], Q[j]) / (dist_mtx[i,j]**3)
#                   if debug:
#                       print(f"Q[i] (i={i}) has shape {Q[i].shape}")
#                       print(f"Q[j] (j={j}) has shape {Q[j].shape}")
                    # since dDiv_dQi(Qi,Qj) is not a symmetric function, must iterate over i=/=j, not i < j

    dVdQi = -np.sum(dVdQij, axis=2)

    dDivdQ[I,:,J] = dDiv_dQij
    dDivdQ[J,:,I] = dDiv_dQji


    if debug:
        print(f"dVdQi = \n{dVdQi}") 
        print(f"dVdQij = \n{dVdQij}") 
        # print(f"dVdQi: shape={dVdQi.shape}, type={type(dVdQi)}")
    return dVdQi, dDivdQ, dist_mtx


def collision_corrector(P, dist_mtx, dDivdQ, min_dist=5e-4):

    idx_collide = np.where(dist_mtx <= min_dist)
    I = idx_collide[0]
    J = idx_collide[1]


# Q-derivative of kinetic energy function -- for stability reasons, want this to mostly be positive (so that Pdot ~ -|P|^2)
def get_dTdQ(Q, P, normX, dist_type="JS", Pbar=None, Qbar=None, atol=1e-9, rtol=1e-7, debug=False):

    if dist_type == "JS":
        if Qbar is None:
            Qbar = safe(np.mean(Q, axis=1).reshape(-1,1), atol=atol, rtol=rtol)
        if Pbar is None:
            Pbar = safe(np.mean(P, axis=1).reshape(-1,1), atol=atol, rtol=rtol)

        if debug:
            print(f"P: shape {P.shape}, type={type(P)}")
            print(f"Q: shape {Q.shape}, type={type(Q)}")
            print(f"normX: shape {normX.shape}, type={type(normX)}")
            print(f"Pbar: shape {Pbar.shape}, type={type(Pbar)}")
            print(f"Qbar: shape {Qbar.shape}, type={type(Qbar)}")

        # maybe should be -1 times below? but then solution blows up in p (i.e., p=o(e^(p^2)))
        dTdQ = 1/normX * ( (1 + normX)*np.sign(Q)*Kinetic(
            P, 
            Q,
            normX,
            dist_type = dist_type,
            Pbar=Pbar,
            Qbar=Qbar,
            atol=atol,
            rtol=rtol
            ) + 0.5* ( Pbar**2 / (Qbar*Q.shape[0])**2 + P**2 * Q**2 ) )
    elif dist_type == "euclidean":
        # dTdQ=0 if and only if the Riemannian metric is constant (as a function of coordinates Q) 
        dTdQ = np.full(shape=P.shape, fill_value=0)
    else:
        raise ValueError(f"dTdQ is not computed for distance metric of type \'{dist_type}\'")

    if debug:
        print(f"dTdQ: shape={dTdQ.shape}, type={type(dTdQ)}")
    return dTdQ
####################################################################################################################


####################################################################################################################
# Utility functions

# input: nxk matrix X (array of vectors), norm order, and flag
# output: array of unit-norm vectors, list of (original) vector norms
def normvec(X, p=1, return_X=True, return_norm=True, axis=1):
    if np.ndim(X) > 1:
        new_shape = list(X.shape)
        new_shape[axis] = 1
        pnorm_Xi = np.linalg.norm(X, ord=p, axis=axis).reshape(*new_shape)
        if max(pnorm_Xi.shape) == 1:
            pnorm_Xi = np.squeeze(pnorm_Xi)
    elif np.ndim(X) == 1:
        pnorm_Xi = np.linalg.norm(X, ord=p)
    else:
        pnorm_Xi = np.abs(X)


    if return_X:
        X_normed = X/pnorm_Xi
        if return_norm:
            return X_normed, pnorm_Xi
        else:
            return X_normed
    else:
        return None, pnorm_Xi

# for avoiding 0 in denominators in calculations
def safe(X, atol = 1e-9, rtol = 1e-7, pnorm=1, fast=True):
    if fast:
        eps_val = atol
        X_safe = np.where(np.abs(X) < eps_val, eps_val, X)
    else:
        if isinstance(X, np.ndarray):
            eps_val = min(rtol / len(X) * np.linalg.norm(X.ravel(), ord=pnorm), atol)
        elif isinstance(X, Number):
            eps_val = min(rtol * np.abs(X), atol)
        else:
            raise Warning(f"encountered input X of type \"{type(X)}\" -- defaulting to eps={atol} (atol value)")
            eps_val = atol

        X_small = np.abs(X) < eps_val
        X_neg = X < 0

        X_safe = np.where(X_small, eps_val, X)
        X_safe = np.where(X_small & X_neg, -eps_val, X)

    return X_safe

# flexibly-defined distance function
def get_distvals(inp_arr, dist_type, atol=1e-9, rtol=1e-8, avg_method=None, min_dist=1e-3, pnorm=1, gate=True):
    if dist_type == 'JS':
        d_ij = fast_genmean_JSdist(inp_arr, atol=atol, rtol=rtol, dists_only=True, avg_method=avg_method)
    elif dist_type in metric_list:
        d_ij = scidist.pdist(inp_arr, metric=dist_type)

    if gate:
        d_ij = _gate_dists(d_ij, min_dist=min_dist)

    return d_ij


# switch statement for the derivative of the (Bregman) divergence corresponding to the specified distance
def get_div_deriv(dist_type, atol=1e-9, rtol=1e-7, avg_method="signed_geometric", pnorm=1):
    deriv_err = Exception(f"Derivative not implemented for \ndistance type: {dist_type} \navg_method type: {avg_method}. Consider using an autodifferentiation method instead.")

    switch = {
            "JS": (lambda q1, q2 : get_dJSdiv_dQi(q1,q2,pnorm=pnorm, avg_method=avg_method)),
            "jensenshannon": (lambda q1, q2 : get_dJSdiv_dQi(q1,q2,pnorm=pnorm, avg_method=avg_method)),
            "euclidean": (lambda q1, q2 : q1 - q2),
     }
    divderiv_fn = switch.get(dist_type, deriv_err)
    return divderiv_fn

# function pair for going between state vectors and phase variables
# note that "stack" and "np.asarray(np.split())" are inverse functions
# generalizes 'y = concat([p, q])' to arbitrary (shared) P,Q-shapes
def ylike_from_varslike(Q, P):
    y = np.stack([Q, P])
    return y.ravel()

# generalizes 'p, q = np.split(y, 2)' to arbitrary (shared) P,Q-shapes
def varslike_from_ylike(y, var_shape):
    y0 = y.reshape((2,) + var_shape)
    return y0[0], y0[1]

# utility class for computational options specification
class Options(object):
    def __init__(self, 
                 problem_type=None,
                 backend=None, 
                 min_dist=1e-3,
                 atol=1e-9, 
                 rtol=1e-7, 
                 dist_type="JS", 
                 avg_method=None, 
                 pnorm=1, 
                 eval_len = 5e2,
                 sys_shape=None, 
                 renormalize=False,
                 int_params=None,
                 command=None
                 ):
        self.atol = atol
        self.min_dist = min_dist
        self.rtol = rtol
        self.backend = backend
        self.dist_type = dist_type
        if dist_type == "JS":
            self.avg_method = avg_method
        else:
            # averaging method is only relevant when using the Jensen-Shannon distance
            self.avg_method = None
        self.pnorm = pnorm
        self.eval_len = eval_len
        self.sys_shape = sys_shape
        self.renormalize = renormalize
        self.int_params = int_params
        self.problem_type = problem_type
        self.cmd = command


class Parameters(object):
    pass


# recasts a dictionary as a Class object
class dict_obj:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


# utility class for specification of Hamiltonian solver parameters 
def init_params(backend="pyhamsys", step=5e-3, tol=1e-9, max_iter=1e1):
    if backend == "pyhamsys":
        params = pH.Parameters(
                step=step, 
                tol=tol,
                max_iter=max_iter,
                extension=True, 
                check_energy=True, 
                projection=None, 
                solver="BM6"
                )
    else:
        max_time = step * max_iter

        params = Parameters()
        params.step=step
        params.max_iter=max_iter
        params.max_time=max_time

    return params

def get_fpath(options=None, outdir=None):
    if options is not None:
        a = options.avg_method
        d = options.dist_type
        if a is not None:
            d = f"{d}-({a})"
        backend = options.backend
        sz = options.sys_shape
        name = options.problem_type
        if name is None:
            fname = f"{backend}Sol_{d}_{sz[0]}x{sz[1]}.json"
        else:
            fname = f"{name}_{backend}Sol_{d}_{sz[0]}x{sz[1]}.json"
    else:
        date_str = datetime.datetime.now().strftime('%y_%h_%d_%H-%M-%S')
        fname = f"{backend}Sol_{date_str}.json"
        
    outpath = os.path.join(outdir, fname)

    return fname, outpath


# write Hamiltonian solution to file
def write_solution(solution_obj, outpath, options=None, backend="pyhamsys", verbose=True):
    if backend == "pyhamsys":
        # write object from the pyhamsys 'solution' class to (JSON) file
        if solution_obj.options is not None:
            options = solution_obj.options

        if options is not None:
            options.int_params.logger = None
            if options.int_params is not None:
                options.int_params = options.int_params.__dict__
            # json.dump(vars(options), fout, indent=4, sort_keys=True)

        if isinstance(options, dict):
            solution_obj.options = options
        else:
            solution_obj.options = vars(options)

        for i in solution_obj:
            if isinstance(solution_obj[i], np.ndarray):
                # converting key {i} with ndarray values of shape {sol[i].shape} to list:
                solution_obj[i] = solution_obj[i].tolist()

        sol_out = dict(solution_obj)
        with open(outpath, 'w') as fout:
            json.dump(sol_out, fout, indent=4, sort_keys=True)
            # json.dump(sol_compat, fout, indent=4, sort_keys=True)
    else:
        raise Exception(f"Hamiltonian solution writer not implemented for backend \'{backend}\'")

    print(f"solution saved to: \n{outpath}")
    return None 

# verify that the given distance function is computable and allowed
def _validate_dist(value):
    if not isinstance(value, str):
        raise ValueError()
    if not value in metric_list:
        raise ValueError()
    return value
####################################################################################################################


####################################################################################################################
# Statistical functions (i.e., JS divergence and generalized averaging functions)

# input: k-vectors q1 and q2 each of unit 1-norm (signed discrete probability distributions w/ same event cardinality)
# output: modified Jensen-Shannon distance between p1 and p2, where the mixture distribution M12 is the (signed) geometric (not arithmetic) mean distribution of q1 and q2.
# NOTE: uses extension of KL_div for non-normalized measures q1, q2: KL(q1,q2) = \int_Q q1 log(q1/q2) + q2 - q1 d\mu(q)
def fast_genmean_JSdist(Q, atol=1e-9, rtol=1e-7, avg_method=None, pnorm=1, dists_only=False, broadcast=True, debug=False):
    Q_pairs = np.asarray( [qij for qij in itertools.combinations(Q, 2)] )
    A_pairs = safe(np.mean( Q_pairs, axis=1 ), atol=atol, rtol=rtol)
    M_pairs = safe(gen_mean(Q_pairs, avg_method=avg_method, axis=1, atol=atol, rtol=rtol), atol=atol, rtol=rtol)

    Qi = Q_pairs[:,0,:]
    Qj = safe(Q_pairs[:,1,:], atol=atol, rtol=rtol)

    JSdists_pairs, A_norm, M_norm, JSterm_pairs, Mterm_pairs = genmean_JSdist(Qi, Qj, A=A_pairs, M=M_pairs, atol=atol, rtol=rtol, axis=1)

    if dists_only:
        d_ij = JSdists_pairs
        return d_ij

    varset = [A_pairs, M_pairs, A_norm, M_norm, JSdists_pairs, JSterm_pairs, Mterm_pairs]
    varsout = [np.squeeze(np.array(var)) for var in varset]
    if broadcast:
        varsout = [ var.reshape(-1,1) if np.ndim(var) == 1 else var for var in varsout ]

    if debug:
        print(f"Q pairs: shape = {Q_pairs.shape}, type={type(Q_pairs)}")
        print("other vars:")
        for var in varsout:
            print(f"shape = {var.shape} of type \'{type(var)}\'\n")

    return Q_pairs, *varsout


# input: k-vectors q1 and q2 each of unit 1-norm (signed discrete probability distributions w/ same event cardinality)
# output: modified Jensen-Shannon distance between p1 and p2, where the mixture distribution M12 is the (signed) geometric (not arithmetic) mean distribution of q1 and q2.
# NOTE: uses extension of KL_div for non-normalized measures q1, q2: KL(q1,q2) = \int_Q q1 log(q1/q2) + q2 - q1 d\mu(q)
def genmean_JSdist(q1, q2, A=None, M=None, atol=1e-9, rtol=1e-7, avg_method=None, pnorm=1, axis=1, debug=False):

#   q1 = q1.reshape(1,-1)
#   q2 = q2.reshape(1,-1)
#   quot_vec = safe(q1, atol=atol, rtol=rtol) / safe(q2, atol=atol, rtol=rtol) 
    quot_vec = q1 / q2

    # if in regime where numexpr faster than numpy, use numexpr
    big_flag = np.prod(q1.shape) >= 1e4+1

    if avg_method is None:
        avg_method = "signed_geometric"

    if (A is None) or (M is None):
        q1q2 = np.squeeze(np.stack([q1, q2]))

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

    return np.sqrt(JSdiv), A_norm, M_norm, JS_term, M_term

# if q1, q2 have ndim > 1, takes KLdiv along axis ax=1
def KL_div(q1,q2, atol=1e-9, rtol=1e-7, axis=1, debug=False, big_flag=False):

    # if in regime where numexpr faster than numpy, use numexpr
    big_flag = np.prod(q1.shape) >= 1e4+1
    if big_flag:
        expr = "q1 * ( log( abs(q1) ) - log( abs(q2) ) )"
        KL_all = ne.evaluate(expr)
    else:
        logdiff = np.log( np.abs(q1) ) - np.log( np.abs(q2) )
        KL_all = q1 * logdiff

    if np.ndim(q1) > 1:
        kldiv = np.sum(KL_all, axis=axis)
    else:
        kldiv = np.sum(KL_all)

    if len(kldiv.shape) > 0:
        if max(kldiv.shape) == 1:
            kldiv = np.squeeze(kldiv)

    if debug:
        shapes_report = f"\n\tq1 shape = {q1.shape}\n, \tq2 shape = {q2.shape}\n, \tlogdiff shape = {logdiff.shape}\n"
        if (max(np.ndim(q1), np.ndim(logdiff)) <= 2) and (min(min(q1.shape), min(logdiff.shape)) == 1):
            q10 = np.squeeze(q1)
            logdiff0 = np.squeeze(logdiff)
            err_str = f"sum-computed KLdiv =\\= dot-computed KLdiv: \nsum_kldiv = {kldiv} \ndot_kldiv = {np.dot(q10,logdiff0)}"
            assert np.isclose(kldiv, np.dot(q10, logdiff0), atol=atol, rtol=rtol), shapes_report + err_str
        if big_flag:
            print("Issuing shape report for large inputs:")
        else:
            print(shapes_report)

    return kldiv


def get_dJSdiv_dQi(Qi, Qj, atol=1e-9, rtol=1e-7, KLdiv=None, JSdiv=None, M=None, A=None, M_norm=None, A_norm=None, pnorm=1, avg_method="signed_geometric", debug=True):
    if M is None:
        M = safe(gen_mean( np.stack([Qi, Qj]), avg_method=avg_method ), atol=atol, rtol=rtol)
    if M_norm is None:
        _, M_norm = normvec( M, p=pnorm, return_X=False)
    M_sgn = np.sign(M)

    if A is None:
        A = safe((Qi + Qj)/2, atol=atol, rtol=rtol)
    if A_norm is None:
        _, A_norm = normvec( A, p=pnorm, return_X=False)

    big_flag = np.prod(Qi.shape) >= 1e4+1

    M_sgn = np.sign(M)
    A_sgn = np.sign(A)
    Q_sgn = np.sign(Qi)
    dMdQi = get_dMijdQi(Qi, M, avg_method=avg_method)
    
    if np.ndim(A) >= 2:
        A_sum = ne.evaluate("sum(A, axis=1)")
    else:
        A_sum = np.sum(A)

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
        logNa = np.log(A_norm)
        sgnA_term = 1/2 * A_sgn * ( JSdiv - KLdiv - 2 * A_norm * logNa - A_sum/A_norm )
        sgnQi_term = Q_sgn * logNa * ( np.log(np.abs(Qi/A)) + 1 )
        logterm = np.log(np.abs(A/M)*M_norm/A_norm)
        dM_term = dMdQi * ( M_sgn * A_sum/M_norm - A / M )

        dJSdQi = np.squeeze( sgnA_term + sgnQi_term + 1/2 + logterm + dM_term ) / A_norm

        if np.ndim(Qi) == 1:
            assert np.all(Qi.shape == dJSdQi.shape), f"Qi has shape {Qi.shape}, which should (and does not) match {dJSdQi.shape}."

    if debug:
        print(f"dJSdQi has shape {dJSdQi.shape}")

    return dJSdQi


## deprecated prior version of dJSdiv_dQi
#ef dJSdiv_dQi00(Qi, Qj, atol=1e-9, rtol=1e-7, M=None, M_norm=None, Qibar=None, Qjbar=None, avg_method="signed_geometric"):
#   if Qibar is None:
#       Qibar = safe(np.mean(Qi, axis=1).reshape(-1,1), atol=atol, rtol=rtol)
#   if Qjbar is None:
#       Qjbar = safe(np.mean(Qi, axis=1).reshape(-1,1), atol=atol, rtol=rtol)
#   if M_norm is None:
#       _, M_norm = normvec( gen_mean( np.stack([q1,q2]), avg_method=avg_method ), p=pnorm, return_X=False)
#
#   M_norm = safe(M_norm, atol=atol, rtol=rtol)
#   Qi = safe(Qi, atol=atol, rtol=rtol)
#   Qj = safe(Qj, atol=atol, rtol=rtol)
#   if avg_method == "signed_gemoetric" or avg_method == "Fisher":
#       # exactly coincides w/ the Jeffreys divergence in this case
#       dMijdQi = 1/2 * np.power(np.abs(Qj/Qi) , 1/2)
#       dJSdQi = (1 - Qj / Qi + log( np.abs(Qi/Qj) ))/4
#   else:
#       # assert avg_method == "arithmetic", "Jensen-Shanon derivatives analytically computed only for arithmetic and (signed) geometric means"
#       M, M_norm = normvec( gen_mean( np.stack([q1,q2]), avg_method=avg_method ), p=pnorm)
#       if avg_method == "arithmetic":
#           M = safe(M, atol=atol, rtol=rtol)
#           dMijdQi = 1/len(Qi)
#           dJSdQi = np.log( safe(Qi / M, atol=atol, rtol=rtol) ) - Qj / (M * len(Qj))
#       elif avg_method == "harmonic":
#           dMijdQi = 1/2 * M**2 / Qi**2
#           dJSdQi = 1/2 * ( 1 - Qi/Qj + np.log(2*(Qi + Qj)) - np.log(Qj) )
#       else:
#           raise ValueError(f"Averaging method '{avg_method}' not recognized.")
#
#   dNormij_dQi = math.log(M_norm) + len(Qi)/M_norm * (Qibar + Qjbar) * dMijdQi
#
#   return 0.5*dNormij_dQi + dJSdQi



# Generalized mean functions
# input: (n,)-shaped or (n,k)-shaped array of real numbers
# output: scalar or (1,k)-shaped array (respectively) giving the (signed) geometric mean of the inputs (resp., per-component signed geometric mean)
def gen_mean(X, avg_method=None, param=2, axis=0, atol=1e-12, rtol=1e-8, multidim_compat=True):
    if avg_method is None or avg_method=="arithmetic":
        M = np.mean(X, axis=axis)

    # slight generalization of geometric mean (param=1 => geometric mean)
    elif avg_method == "Fisher":
        M = np.exp( -1/param * np.mean(-param*np.log(np.abs(
            safe(X, atol=atol, rtol=rtol))), axis=axis) )

    elif avg_method == "signed_geometric":
        M0 = np.prod(X, axis=axis)
        M = np.sign(M0) * np.power(np.abs(M0), 1/X.shape[0])

    elif avg_method == "harmonic":
        M = 1 / np.sum( 1/safe(X, atol=atol, rtol=rtol), axis=axis )

    elif avg_method == "r-mean":
        M = np.power(1/param, np.mean( np.power(X, param), axis=axis) )

    # reshapes 1-dim arrays to correctly right-multiply when broadcast against matrices
    if (np.ndim(M) == 1) and multidim_compat:
        M = M.reshape(1,-1)
    
    return M


# compute derivative of averaging function M(Q1,...,Qn) at Qi (M=Mij --> n=2)
def get_dMijdQi(Qi, M, avg_method="signed_geometric", param=2, n=2):
    if avg_method == "arithmetic":
        dMdQi = np.full(shape=Qi.shape, fill_value=1/2)

    # slight generalization of geometric mean (and has same dierivative)
    if (avg_method == "Fisher") or (avg_method == "signed_geometric"):
        dMdQi = M / (n * Qi) 

    elif avg_method == "harmonic":
        dMdQi = 1/n * M**2 / Qi**2

    elif avg_method == "r-mean":
        dMdQi = 1/n * np.power(Qi, param-1) / np.power(M, param-1)

    if (np.ndim(Qi) == 1) and (np.ndim(M) == 1) and multidim_compat:
        dMdQi = np.diag(dMdQi)
    
    return dMdQi
####################################################################################################################

# pleft_default="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/dynamics_viz/sml-dummy-pvals_left.csv"
# pleft_default="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/dynamics_viz/dummy-pvals_left.csv"
# pright_default="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/dynamics_viz/sml-dummy-pvals_right.csv"
# pright_default="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/dynamics_viz/dummy-pvals_right.csv"

# C_default="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/dynamics_viz/sml-dummy-pvals_C.csv"
# C_default="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/dynamics_viz/dummy-pvals_C.csv"
# C_default="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/dynamics_viz/coul_three_body_C.csv"
# C_default="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/dynamics_viz/grav_three_body_C.csv"
C_default="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/dynamics_viz/grav_two_body_C.csv"

####################################################################################################################
# parses input, calls main
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Show distributions of outputs from topological bootstrap"
    )
    parser.add_argument(
        "-L",
        "--left_pvals_fpath",
        type=str,
        # default=pleft_default,
        default='',
        help="filepath containing grid of left pvals"
    )
    parser.add_argument(
        "-R",
        "--right_pvals_fpath",
        type=str,
        # default=pleft_default,
        default='',
        help="filepath containing grid of right pvals"
    )
    parser.add_argument(
        "-C",
        "--pairwise_fpath",
        type=str,
        default=C_default,
        help="filepath to pairwise interaction terms"
    )
    parser.add_argument(
        "-o", 
        "--outdir", 
        default="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/dynamics_viz",
        type=str, 
        help="output directory"
    )
    parser.add_argument(
        "-d", 
        "--dist_type", 
        default="JS",
        type=str, 
        help="(Riemannian) distance function between spatial coordinates"
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
        "-s",
        "--show_figs",
        default=False,
        action="store_true",
        help="toggle showing solution summary figures"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="toggle verbose output"
    )
    parser.add_argument(
        "-r",
        "--renorm",
        default=True,
        action="store_false",
        help="renormalize Q upon extraction from y/ydot"
    )
    args = parser.parse_args()

    main(args)
