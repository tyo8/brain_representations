import os
import numpy as np


# computes a list of admissible permutations given a path to a file defining Exchangeability Blocks [1] on presumed dataset
## [1] Winkler et al. (2015) https://doi.org/10.1016/j.neuroimage.2015.05.092

def perms_from_EBs(EB_fname,n_shuffles):
    EB_data = np.genfromtxt(EB_fname,delimiter=",")
    block_nums = np.unique(EB_data)

    perm_list = np.zeros((len(EB_data),n_shuffles))

    for i in block_nums:
        idx_i = np.where(EB_data == i)[0]
        perm_list[idx_i,:] = shuffle_in_EB(idx_i,n_shuffles)

    return perm_list


def shuffle_in_EB(EB_idx,n_shuffles):
    n_idx = len(EB_idx)

    EB_shuffles = np.zeros((n_idx,n_shuffles))

    for i in range(n_shuffles):
        EB_shuffles[:,i] = EB_idx[np.random.permutation(n_idx)]

    return EB_shuffles
