import os
import sys
import csv
import numpy as np
import HCP_utils as hutils
import subj_data_metrics as sdm
from sklearn.metrics import pairwise_distances


# receives path to .csv file containing list of subject data file names
# outputs a subject-by-subject matrix of Pearson correlation coefficients
def comp_sim_mtx(fname_in, fname_out, method='Psim', return_dist=False):
    with open(fname_in,newline='') as fin:
        subj_lists = list(csv.reader(fin))          # list of lists of subject filenames
        subj_list = list(map(''.join,subj_lists))   # list of strings of subject filenames
    
    too_big = _check_datasize(subj_list)

    if too_big:
        sim_mtx = comp_sim_from_list(subj_list, method=method)
    else:
        data_mtx = np.asarray([hutils._parse_fname(i) for i in subj_list])

        ### debug code ###
        _validate_data(data_mtx)
        ### debug code ###

        sim_mtx = comp_sim_from_mtx(np.double(data_mtx), method=method)

    if method=='geodesic':
        np.fill_diagonal(sim_mtx, 0)
        dist_mtx = sim_mtx
    else:
        np.fill_diagonal(sim_mtx, 1)
        dist_mtx = sdm.p_simdist(sim_mtx, p=2)
        np.fill_diagonal(dist_mtx, 0)
    
    ### debug code ###
    _validate_distmtx(dist_mtx)
    ### debug code ###


    if return_dist:
        np.savetxt(fname_out.replace('_sims.txt','_dists.txt'), dist_mtx)
        return dist_mtx
    else:
        np.savetxt(fname_out, sim_mtx)
        return sim_mtx

# estimate if dataset is too large to fit in memory
def _check_datasize(subj_list):
    n_subj = len(subj_list)
    samp_data = hutils._parse_fname(subj_list[0])
    data_size = samp_data.size

    numel = n_subj*data_size

    GBmem = numel * 32/91282**2
    GBmem = round(GBmem, -int(np.floor(np.log10(GBmem))) + 3)
    
    print("Estimated data size is " + str(n_subj*data_size) + " elements or ~" + str(GBmem) + "GB.") 

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
def comp_sim_from_list(subj_list, method='Psim'):
    n_subj = len(subj_list)
    sim_mtx = np.zeros((n_subj,n_subj))

    for i in range(n_subj):
        subj1_fname = subj_list[i]
        data1 = hutils._parse_fname(subj1_fname)
        for j in range(i+1,n_subj):
            subj2_fname = subj_list[j]
            data2 = hutils._parse_fname(subj2_fname)
            simval = comp_simval(data1,data2, method=method)
            sim_mtx[i,j] = simval

    if np.amax(np.abs(sim_mtx)) > 1:
        sim_mtx = sim_mtx/sim_mtx.max()

    sim_mtx = sim_mtx + sim_mtx.T + np.eye(n_subj)

    return sim_mtx

############################### debug function ###############################  
def _validate_data(data):
    print("Initial data has shape (n_subj, n_feats)=" + str(data.shape))
    n_subj = data.shape[0]
    nan_data = np.isnan(data)
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
############################### debug function ###############################  


############################### debug function ###############################  
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
############################### debug function ###############################  


# input parsing and verbose options output for logs
if __name__=="__main__":
    subj_datalist_fname = sys.argv[1]
    fname_out = sys.argv[2]
    method = sys.argv[3]
    if len(sys.argv) > 4:
        return_dist = sys.argv[4]
    else:
        return_dist = False

    if return_dist:
        fname_out = fname_out.replace('_sims.txt', '_dists.txt')

    print("reading data from " + str(subj_datalist_fname))
    print("sending data to " + str(fname_out))
    print("computing similarity according to " + method)
    print("computing/saving/returning \"distance\" instead of similarity matrices? " + return_dist)
    comp_sim_mtx(subj_datalist_fname, fname_out, method=method, return_dist=return_dist)
