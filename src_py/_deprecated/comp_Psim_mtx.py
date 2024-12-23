import os
import sys
import csv
import numpy as np
import parse_neurodata as parse


def main(argvals):
    comp_dist_mtx(argvals[1],argvals[2])


# receives path to .csv file containing list of subject data file names
# outputs a subject-by-subject matrix of Pearson correlation coefficients
def comp_dist_mtx(fname_in,fname_out):
    with open(fname_in,newline='') as fin:
        subj_lists = list(csv.reader(fin))      # list of lists of subject filenames

    subj_list = list(map(''.join,subj_lists))   # list of strings of subject filenames
    Psim_mtx = comp_Psim_vals(subj_list)
    Psim_mtx = Psim_mtx + np.identity(len(subj_list)) + np.matrix.transpose(Psim_mtx)
    Psim_dmtx = 1 - np.abs(Psim_mtx)
    np.savetxt(fname_out,Psim_dmtx,delimiter=",")
    return Psim_dmtx


def comp_Psim(data1,data2):
    sim_mtx = np.corrcoef(data1,data2)
    simval = sim_mtx[0,1]
    return simval


def comp_Psim_from_mtx(data_mtx):
    n_subj = data_mtx.shape[0]
    Psim_mtx = np.corrcoef(data_mtx)

    assert Psim_mtx.shape == (n_subj, n_subj),'Psim_mtx is not a symmetric n_subj x n_subj matrix.'

#     for i in range(n_subj):
#         data_i = data_mtx[i,:]
#         for j in range(i+1,n_subj):
#             data_j = data_mtx[j,:]

#             Psim_mtx[i,j] = comp_Psim(data_i, data_j)

#     Psim_mtx = Psim_mtx + np.transpose(Psim_mtx) + np.identity(n_subj)
    return Psim_mtx

# the function comp_Psim_vals could be rewritten to recursively check and split lists that try to load more than 1TB of data
def comp_Psim_vals(subj_list):
    n_subj = len(subj_list)
    Psim_mtx = np.zeros((n_subj,n_subj))

    if n_subj > 10**4:
        for i in range(n_subj):
            subj1_fname = subj_list[i]
            data1 = parse.parse_fname(subj1_fname)
            for j in range(i+1,n_subj):
                subj2_fname = subj_list[j]
                data2 = parse.parse_fname(subj2_fname)
                simval = comp_Psim(data1,data2)
                Psim_mtx[i,j] = simval
    else:
        data_list = ['']*n_subj
        for i in range(n_subj):
            data_list[i] = parse.parse_fname(subj_list[i])

        for i in range(n_subj):
            for j in range(i+1,n_subj):
                simval = comp_Psim(data_list[i],data_list[j])
                Psim_mtx[i,j] = simval

    return Psim_mtx

if __name__=="__main__":
    main(sys.argv)
