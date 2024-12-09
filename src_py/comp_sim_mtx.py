import os
import csv
import ast
import argparse
import permuteHCP
import numpy as np
import HCP_utils as hutils
import subj_data_metrics as sdm
from sklearn.metrics import pairwise_distances


# receives path to .csv file containing list of subject data file names
# outputs a subject-by-subject matrix of Pearson correlation coefficients
def comp_sim_mtx(
        fname_in, 
        fname_out, 
        method='Psim', 
        return_dist=False,
        permute=False,
        perm_type="subject",
        perm_set=None,
        rng_seed=0
        ):
    with open(fname_in,newline='') as fin:
        subj_list = fin.read().split()   # list of strings of subject filenames
    
    too_big = _check_datasize(subj_list)

    if too_big:
        sim_mtx = comp_sim_from_list(subj_list, method=method, perm_pars=[permute, perm_type, None, rng_seed])
    else:
        data_mtx = np.asarray([hutils._parse_fname(i) for i in subj_list])

        ### debugging code ###
        _validate_data(data_mtx)
        ### debugging code ###

        if permute:
            ### debugging code ###
            print(f"original data: \n{data_mtx}")
            ### debugging code ###
            data_mtx = permuteHCP.permute(data_mtx, perm_type=perm_type, perm_set=perm_set, rng_seed=rng_seed)
            ### debugging code ###
            print(f"permuted data: \n{data_mtx}")
            ### debugging code ###

        sim_mtx = comp_sim_from_mtx(np.double(data_mtx), method=method)

    if method=='geodesic':
        np.fill_diagonal(sim_mtx, 0)
        dist_mtx = sim_mtx
    else:
        np.fill_diagonal(sim_mtx, 1)
        dist_mtx = sdm.p_simdist(sim_mtx, p=2)
        np.fill_diagonal(dist_mtx, 0)
    
    ### debugging code ###
    _validate_distmtx(dist_mtx)
    ### debugging code ###


    if return_dist:
        np.savetxt(fname_out.replace('_sims.txt','_dists.txt'), dist_mtx)
        return dist_mtx
    else:
        np.savetxt(fname_out, sim_mtx)
        return sim_mtx

# estimate if dataset is too large to fit in memory
def _check_datasize(subj_list):
    n_subj = len(subj_list)
    numel = n_subj*hutils._parse_fname(subj_list[0]).size

    GBmem = numel * 4/2**30     # assumes data is a float array
    GBmem = round(GBmem, -int(np.floor(np.log10(GBmem))) + 3)
    
    print(f"Estimated dataset size is {numel} elements or ~{GBmem}GB.") 

    too_big = (GBmem > 100)

    if too_big:
        print("Data may be too large to fit in memory; opting for sequential data loading (slow).")

    return too_big


# computes similarity (or geodesic distance) between pairwise-loaded subject data (according to specified method)
def comp_simval(data1,data2,method='Psim'):
    if method=='Psim':
        sim_mtx = np.corrcoef(data1,data2)
        simval = np.abs(sim_mtx[0,1])
#   elif method=='dcor':
#       simval = dcor.distance_correlation(data1, data2)
    elif method=='Psim_ztrans':
        simval = sdm._ztrans(data1, data2)
    elif method=='spd_cos':
        simval = sdm.spd_cos(data1, data2)
    elif method=='inner':
        simval = sdm.inner(data1, data2)
    elif method=='geodesic':
        simval = sdm.geodesic(data1, data2)

    return simval


# computes pairwise similarity (or geodesic distance) between group-loaded subject data (according to specified method)
def comp_sim_from_mtx(data_mtx, method='Psim', p=2):
    n_subj = data_mtx.shape[0]
    if method=='Psim':
        sim_mtx = np.corrcoef(data_mtx)
#    elif method=='dcor':
#        sim_mtx = pairwise_distances(data_mtx, metric=dcor.distance_correlation)
    elif method=='Psim_ztrans':
        sim_mtx = np.corrcoef(sdm._ztrans(data_mtx))
    elif method=='spd_cos':
        sim_mtx = pairwise_distances(data_mtx, metric=sdm.spd_cos)
    elif method=='inner':
        sim_mtx = pairwise_distances(data_mtx, metric=sdm.inner)
        sim_mtx = sim_mtx/sim_mtx.max()
    elif method=='geodesic':
        sim_mtx = pairwise_distances(data_mtx, metric=sdm.geodesic)
        sim_mtx = sim_mtx/sim_mtx.max()

    assert sim_mtx.shape == (n_subj, n_subj),'sim_mtx is not a symmetric n_subj x n_subj matrix.'

    return sim_mtx


# sequentially loads data in pairs; very slow, but can handle jobs that are too large to fit in memory
## NOTE: can this be re-written to be more efficient? we may need in the case of the dense connectome -- can we try to "traverse" the upper triangle in a more efficient way?
def comp_sim_from_list(subj_list, method='Psim', perm_pars=[None]):
    n_subj = len(subj_list)
    sim_mtx = np.zeros((n_subj,n_subj))

    if perm_pars[0]:
        if perm_pars[1]=="subject":
            raise Exception("Full dataset is too large for subject-type permutations. Exiting.")
        elif perm_pars[1]=="feature":
            rng = np.random.default_rng(perm_pars[2])
        else:
            raise ValueError(f"Unrecognized permutation type \"{perm_pars[1]}\".")

    for i in range(n_subj):
        subj1_fname = subj_list[i]
        data1 = hutils._parse_fname(subj1_fname)
        for j in range(i+1,n_subj):
            subj2_fname = subj_list[j]
            data2 = hutils._parse_fname(subj2_fname)
            if perm_pars[0]:
                if perm_pars[1]=="feature":
                    data2 = rng.permutation(data2)
            simval = comp_simval(data1,data2, method=method)
            sim_mtx[i,j] = simval

    if np.amax(np.abs(sim_mtx)) > 1:
        sim_mtx = sim_mtx/sim_mtx.max()

    sim_mtx = sim_mtx + sim_mtx.T + np.eye(n_subj)

    return sim_mtx

############################### debugging function ###############################  
def _validate_data(data):
    print("Initial data has shape (n_subj, n_feats)=" + str(data.shape))
    n_subj = data.shape[0]
    try:
        nan_data = np.isnan(data)
    except TypeError:
        ### debugging code ###
        print(f"data could not safely be checked for nan values because it is of type {type(data)}")
        print(f"data shape: {data.shape}")
        print(f"unique data shapes: {set([i.shape for i in data])}")
        ### debugging code ###
    nanflag = nan_data.any()

    subj_vals = np.sum(data, axis=1)
    if nanflag:
        nan_els = np.count_nonzero(nan_data)
        nan_subjs = np.count_nonzero(np.isnan(subj_vals))

        nan_feats = [np.count_nonzero(np.isnan(data[i,:])) 
                for i in range(n_subj) if np.isnan(subj_vals[i])]

    print("NaN values found: " + str(nanflag))
    if nanflag:
        print("Number of NaN elements: " + str(nan_els))
        print("Number of subjects with NaN data: " + str(nan_subjs))
        print("Number of NaN features per subject with NaN data: ")
        print(np.histogram(nan_feats))
        raise ValueError("Input data validation failed. Exiting to avoid further NaN-corrupted calculations.")
############################### debugging function ###############################  


############################### debugging function ###############################  
def _validate_distmtx(dmtx):
    n_subj = dmtx.shape[0]
    nan_data = np.isnan(dmtx)
    nanflag = nan_data.any()

    subj_vals = dmtx[np.triu_indices(n_subj,1)]
    if nanflag:
        nan_els = np.count_nonzero(subj_vals)

    print("NaN values found in dist_mtx: " + str(nanflag))
    if nanflag:
        print("Number of NaN elements: " + str(nan_els), f"({nan_els/len(subj_vals)*100}%)")
        raise ValueError("Distance matrix validation failed. Exiting to avoid further NaN-corrupted calculations.")
############################### debugging function ###############################  


# input parsing and verbose options output for logs
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Generate family structure-respecting permutations of HCP data"
    )
    parser.add_argument(
        "-i", "--subj_datalist_fname", type=str, help="input filepath to saved exchangeability blocks"
    )
    parser.add_argument(
        "-o", "--fname_out", type=str, help="output directory for permutation sets"
    )
    parser.add_argument(
        "-m", "--method", type=str, help="output directory for permutation sets"
    )
    parser.add_argument(
        "-D", "--return_dist", default=False, action="store_true", help="if flag given, then output distance (rather than similarity) matrix"
    )
    parser.add_argument(
        "-P", "--permute", default=False, action="store_true", help="if flag given, then output distance (rather than similarity) matrix"
    )
    parser.add_argument(
        "-t", "--perm_type", default="subject", type=str, help="permutation type: \"subject\" or  \"feature\" (the distribution of values is preserved across the unchosen)"
    )
    parser.add_argument(
        "-s", "--perm_set", default=None, help="Array of permutations or path to an array of permutations"
    )
    parser.add_argument(
        "-r", "--rng_seed", default=0, type=int, help="RNG seed (not used for \'subject\'-type permutations)"
    )
    args = parser.parse_args()

    if args.return_dist:
        args.fname_out = args.fname_out.replace('_sims.txt', '_dists.txt')

    print(f"reading data from {str(args.subj_datalist_fname)}")
    print(f"sending data to {str(args.fname_out)}")
    print(f"computing similarity according to {args.method}")
    print("computing/saving/returning \"distance\" instead of similarity matrices? ",  args.return_dist)
    print("conducting permutation testing?", args.permute)
    if args.permute:
        if args.perm_type=="feature":
            args.perm_set = ast.literal_eval(args.perm_set)     # interpret read-in as integer for feature-type permutations
        print("")
        print(f"Permutation type: {args.perm_type}")
        print(f"Permutation set specified: {args.perm_set} (of type {type(args.perm_set)})")
        print(f"Permutation random generator seed (not relevant to subject-type permutations): {args.rng_seed}")
        print("")

    comp_sim_mtx(
            args.subj_datalist_fname, 
            args.fname_out, 
            method = args.method, 
            return_dist = args.return_dist,
            permute = args.permute,
            perm_type = args.perm_type,
            perm_set = args.perm_set,
            rng_seed = args.rng_seed
            )
